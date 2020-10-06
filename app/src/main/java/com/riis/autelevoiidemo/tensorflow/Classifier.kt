package com.riis.autelevoiidemo.tensorflow

import android.graphics.Bitmap
import android.graphics.RectF

/**
 * Generic interface for interacting with different recognition engines.
 */
interface Classifier {
    /**
     * An immutable result returned by a Classifier describing what was recognized.
     */
    data class Result(
        val id: String?,
        val title: String,
        val confidence: Float,
        val location: RectF
    ) {

        override fun toString(): String {
            var resultString = StringBuilder()


            id?.let { resultString = resultString.append("[$it] ") }
            title.let { resultString =  resultString.append("$it ") }
            confidence.let {resultString = resultString.append("%.2f".format(confidence * 100.0f)) }
            location.let { resultString = resultString.append("$it") }

            return resultString.toString()
        }
    }
    fun recognizeImage(bitmap: Bitmap) : List<Result>
    fun enableStatLogging(debug: Boolean)
    fun getStatString() : String
    fun close()
}