package com.riis.autelevoiidemo.model

import android.content.res.AssetManager
import android.graphics.Bitmap
import com.riis.autelevoiidemo.tensorflow.TFLiteObjectDetectionAPIModel

class ImageClassifier(assetManager: AssetManager) {

    private val detectionModel = TFLiteObjectDetectionAPIModel.create(assetManager,

                                               "example.tflite",
                                                "example_labels.txt",
                                                    300, true)
    var processing = false

    /**
     * Runs tensorflow classification on the image and returns and unscaled list of bounding
     * boxes
     *
     * @post { box.confidence >= 0.55 | box \in result }
     */
    fun classifyImage(image: Bitmap): MutableList<BoundingBox> {
        processing = true

        val result = mutableListOf<BoundingBox>()
        val detections = (detectionModel as TFLiteObjectDetectionAPIModel)
            .recognizeImageWithResolution(image, image.width, image.height)

        detections.filter { detection -> detection.confidence >= 0.65f }
            .forEach { detection ->
                val box = BoundingBox(detection.location.left,
                    detection.location.top,
                    detection.location.right,
                    detection.location.bottom,
                    detection.title)
                result.add(box)
            }
        processing = false
        return result
    }

}