package com.riis.autelevoiidemo.tensorflow

import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.RectF
import android.os.Trace
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.*
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.*
import kotlin.collections.HashMap

class TFLiteObjectDetectionAPIModel : Classifier {

    // Only return this many sheep (maximally).
    private var isModelQuantized: Boolean = false
    // Float model
    private val IMAGE_MEAN = 128.0f
    private val IMAGE_STD = 128.0f
    // Config values.
    private var inputSize: Int = 0

    var processing = false

    // Pre-allocated buffers.
    private val labels = Vector<String>()
    private var intValues: IntArray? = null
    // outputLocations: array of shape [Batchsize, NUM_DETECTIONS,4]
    // contains the location of detected boxes
    private var outputLocations: Array<Array<FloatArray>> = arrayOf()
    // outputClasses: array of shape [Batchsize, NUM_DETECTIONS]
    // contains the classes of detected boxes
    private var outputClasses: Array<FloatArray>? = null
    // outputScores: array of shape [Batchsize, NUM_DETECTIONS]
    // contains the scores of detected boxes
    private var outputScores: Array<FloatArray>? = null
    // numDetections: array of shape [Batchsize]
    // contains the number of detected boxes
    private var numDetections: FloatArray? = null

    private var imgData: ByteBuffer? = null

    private var tfLite: Interpreter? = null

    companion object {
        var xScaler = 1.0f
        var yScaler = 1.0f
        private const val NUM_DETECTIONS = 10
        // Number of threads in the java app
        private const val NUM_THREADS = 4

        /** Memory-map the model file in Assets.  */
        @Throws(IOException::class)
        private fun loadModelFile(assets: AssetManager, modelFilename: String): MappedByteBuffer {
            val fileDescriptor = assets.openFd(modelFilename)
            val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
            val fileChannel = inputStream.channel
            val startOffset = fileDescriptor.startOffset
            val declaredLength = fileDescriptor.declaredLength
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
        }

        /**
         * Initializes a native TensorFlow session for classifying images.
         *
         * @param assetManager The asset manager to be used to load assets.
         * @param modelFilename The filepath of the model GraphDef protocol buffer.
         * @param labelFilename The filepath of label file for classes.
         * @param inputSize The size of image input
         * @param isQuantized Boolean representing model is quantized or not
         */
        fun create(
        assetManager: AssetManager,
        modelFilename: String,
        labelFilename: String,
        inputSize: Int,
        isQuantized: Boolean
        ): Classifier {
            val d = TFLiteObjectDetectionAPIModel()
            var line: String

            val labelsInput: InputStream? = assetManager.open(labelFilename)
            val br: BufferedReader? = BufferedReader(InputStreamReader(labelsInput))

            br?.let{
                it.forEachLine { line ->
                    d.labels.add(line)
                }
                it.close()
            }

            d.inputSize = inputSize

            try {
                d.tfLite = Interpreter(loadModelFile(assetManager, modelFilename))
            } catch (e: Exception) {
                throw RuntimeException(e)
            }

            d.isModelQuantized = isQuantized
            // Pre-allocate buffers.
            val numBytesPerChannel: Int
            numBytesPerChannel = if (isQuantized) {
                1 // Quantized
            } else {
                4 // Floating point
            }
            d.imgData =
                ByteBuffer.allocateDirect(1 * d.inputSize * d.inputSize * 3 * numBytesPerChannel)
            d.imgData!!.order(ByteOrder.nativeOrder())
            d.intValues = IntArray(d.inputSize * d.inputSize)

            d.tfLite!!.setNumThreads(NUM_THREADS)
            d.outputLocations = Array(1) { Array(NUM_DETECTIONS) { FloatArray(4) } }
            d.outputClasses = Array(1) { FloatArray(NUM_DETECTIONS) }
            d.outputScores = Array(1) { FloatArray(NUM_DETECTIONS) }
            d.numDetections = FloatArray(1)
            return d
        }
    }

    fun setOriginalImageSize(xSize: Float, ySize: Float) {
        xScaler = xSize / inputSize
        yScaler = ySize / inputSize
    }

    override fun recognizeImage(bitmap: Bitmap): List<Classifier.Result> {
        processing = true
        if (bitmap == null) return ArrayList()

        // Log this method so that it can be analyzed with systrace.
        Trace.beginSection("recognizeImage")

        Trace.beginSection("preprocessBitmap")
        // Preprocess the image data from 0-255 int to normalized float based
        // on the provided parameters.

        val bitmap = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, false)
        bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)

        imgData?.rewind()
        for (i in 0 until inputSize) {
            for (j in 0 until inputSize) {
                val pixelValue = intValues?.get(i * inputSize + j)
                if (isModelQuantized) {
                    // Quantized model
                    if (pixelValue != null) {
                        imgData?.put((pixelValue shr 16 and 0xFF).toByte())
                        imgData?.put((pixelValue shr 8 and 0xFF).toByte())
                        imgData?.put((pixelValue and 0xFF).toByte())
                    }
                } else { // Float model
                    if (pixelValue != null) {
                        imgData?.putFloat(((pixelValue shr 16 and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
                        imgData?.putFloat(((pixelValue shr 8 and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
                        imgData?.putFloat(((pixelValue and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
                    }
                }
            }
        }
        Trace.endSection() // preprocessBitmap

        // Copy the input data into TensorFlow.
        Trace.beginSection("feed")
        outputLocations = Array(1) { Array(NUM_DETECTIONS) { FloatArray(4) { 0f }}}
        outputClasses = Array(1) { FloatArray(NUM_DETECTIONS) }
        outputScores = Array(1) { FloatArray(NUM_DETECTIONS) }
        numDetections = FloatArray(1)

        val inputArray = imgData?.let { arrayOf<Any>(it) }
        val outputMap = HashMap<Int, Any>()
        outputMap[0] = outputLocations
        outputMap[1] = outputClasses!!
        outputMap[2] = outputScores!!
        outputMap[3] = numDetections!!


        Trace.endSection()


        // Run the inference call.
        Trace.beginSection("run")
        tfLite?.runForMultipleInputsOutputs(inputArray, outputMap)
        Trace.endSection()

        // Show the best detections.
        // after scaling them back to the input size.
        val recognitions = ArrayList<Classifier.Result>(NUM_DETECTIONS)
        for (i in 0 until NUM_DETECTIONS) {
            val detection = RectF(
                outputLocations[0][i][1] * inputSize.toFloat() * xScaler,
                outputLocations[0][i][0] * inputSize.toFloat() * yScaler,
                outputLocations[0][i][3] * inputSize.toFloat() * xScaler,
                outputLocations[0][i][2] * inputSize.toFloat() * yScaler
            )

            // SSD Mobilenet V1 Model assumes class 0 is background class
            // in label file and class labels start from 1 to number_of_classes+1,
            // while outputClasses correspond to class index from 0 to number_of_classes
            val labelOffset = 0

            recognitions.add(
                Classifier.Result(
                    "" + i,
                    labels[outputClasses!![0][i].toInt() + labelOffset],
                    outputScores!![0][i],
                    detection
                )
            )
        }
        Trace.endSection() // "recognizeImage"
        processing = false
        return recognitions
    }

    fun recognizeImageWithResolution(bitmap: Bitmap, w: Int, h: Int): List<Classifier.Result> {
        val origXScaler = xScaler
        val origYScaler = yScaler
        this.setOriginalImageSize(w.toFloat(), h.toFloat())
        while (processing); // block while processing and get in as soon as we can
        val results = recognizeImage(bitmap)
        xScaler = origXScaler
        yScaler = origYScaler
        return results
    }


    override fun close() { }
    override fun getStatString(): String { return "" }
    override fun enableStatLogging(debug: Boolean) { }

}