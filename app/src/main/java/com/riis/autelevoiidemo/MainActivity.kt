package com.riis.autelevoiidemo

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.View
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.drawToBitmap
import com.autel.common.CallbackWithNoParam
import com.autel.common.error.AutelError
import com.autel.sdk.Autel
import com.autel.sdk.AutelSdkConfig
import com.autel.sdk.ProductConnectListener
import com.autel.sdk.product.BaseProduct
import com.autel.sdk.widget.AutelCodecView
import com.google.android.material.floatingactionbutton.FloatingActionButton
import com.riis.autelevoiidemo.model.BoundingBox
import com.riis.autelevoiidemo.model.ImageClassifier
import com.riis.autelevoiidemo.view.OverlayView
import org.jetbrains.anko.doAsync


class MainActivity : AppCompatActivity() {
    // Drone
    private lateinit var currentProduct: BaseProduct

    // Tensorflow Classifier
    private lateinit var classifier: ImageClassifier

    // Views
    private lateinit var stream_view_overlay: OverlayView
    private lateinit var codec_view: AutelCodecView
    private lateinit var textView: TextView

    // Hnadler to capture view every 50 ms
    lateinit var mainHandler: Handler

    // Calls function every 50ms to capture view from drone stream
    private val updateCaptureLoop = object: Runnable {
        override fun run() {
            captureBitmapLoop()
            mainHandler.postDelayed(this, 50)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Initializes all relevant variables
        mainHandler = Handler(Looper.getMainLooper())
        classifier = ImageClassifier(application.assets)
        stream_view_overlay = findViewById(R.id.stream_view_overlay)
        codec_view = findViewById(R.id.codec_view)
        textView = findViewById(R.id.textView)

        // Creates SDK connection
        initSDK()
        // Connects the app to the drone
        initProduct()

    }

    private fun initSDK(){
        val config = AutelSdkConfig.AutelSdkConfigBuilder()
            .setAppKey("<SDK license should be input>")
            .setPostOnUi(true)
            .create()


        Autel.init(applicationContext, config, object: CallbackWithNoParam {
            override fun onSuccess() {
                Log.i("autelInit", "connected")
            }

            override fun onFailure(p0: AutelError?) {
                Log.i("autelInit", "${p0?.description}")
            }

        })
    }

    private fun initProduct() {
        Autel.setProductConnectListener(object: ProductConnectListener {
            override fun productConnected(p0: BaseProduct?) {
                // If drone is connected to the remote
                Log.i("productListener", "connected")
                if (p0 != null) {
                    // Hides the on screen text view displaying "Drone not connected."
                    textView.visibility = View.INVISIBLE
                    // Makes a toast displaying the name of the connected drone
                    Toast.makeText(applicationContext, getString(R.string.connected,  p0.type.toString()), Toast.LENGTH_LONG).show()
                    // Sets the drone reference returned from this function
                    currentProduct = p0


                }
            }

            // If drone is disconnected from the remote
            override fun productDisconnected() {
                Log.i("productListener", "disconnected")
            }

        })
    }

    private fun captureBitmapLoop(){
        // Checks if the codec view is working
        if(codec_view.isAvailable){
            // Captures the a bitmap from the view if possible
            codec_view.bitmap?.let {
                processBitmap(it)
            }

        }

    }

    private fun processBitmap(bitmap: Bitmap) {
        doAsync {
            if(!classifier.processing){
                // classifies the image and return the results
                val results = classifier.classifyImage(bitmap)

                // draws the boxes and title on the "stream_view_overlay" for user to see
                runOnUiThread {
                    drawBoxes(results)
                }

            }
        }

    }

    private fun drawBoxes(boxes: List<BoundingBox>) {
        stream_view_overlay.boxes = boxes
        stream_view_overlay.invalidate()

    }

    override fun onResume() {
        super.onResume()
        // Starts the function that is called every 50 ms
        mainHandler.post(updateCaptureLoop)
    }

    override fun onPause() {
        super.onPause()
        // Removes the handler callback to the function to
        // prevent errors when entering and exiting the app
        mainHandler.removeCallbacks(updateCaptureLoop)
    }
}