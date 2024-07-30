package com.example.cropyield

import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import androidx.activity.ComponentActivity
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel


class MainActivity : ComponentActivity() {
    private lateinit var interpreter: Interpreter
    private lateinit var medians: FloatArray
    private lateinit var means: FloatArray
    private lateinit var meanDB: FloatArray
    private lateinit var scaleDB: FloatArray

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Load the TensorFlow Lite model

        interpreter = Interpreter(loadModelFile())

        // Load the feature medians
        medians = loadMedians()
        means =loadMeans()
        // Load the scaler parameters
        loadScalerParams()

        val yearInput: EditText = findViewById(R.id.year_input)
        val locationIdInput: EditText = findViewById(R.id.location_id_input)
        val predictButton: Button = findViewById(R.id.predict_button)
        val predictionResult: TextView = findViewById(R.id.prediction_result)

        predictButton.setOnClickListener {
            val year = yearInput.text.toString().toFloat()
            val locationId = locationIdInput.text.toString().toFloat()

            // Prepare the input data
            val inputData = prepareInputData(locationId, year)

            // Perform inference
            val prediction = doInference(inputData)

            // Display the result
            predictionResult.text = "Predicted Yield: $prediction"
        }
    }

    private fun loadModelFile(): MappedByteBuffer {
        val assetFileDescriptor = assets.openFd("cnn_model.tflite")
        val fileInputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel = fileInputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    private fun loadMedians(): FloatArray {
        val inputStream = assets.open("feature_medians.npy")
        val mediansByteArray = inputStream.readBytes()
        val mediansByteBuffer = ByteBuffer.wrap(mediansByteArray).order(ByteOrder.nativeOrder())
        val medians = FloatArray(mediansByteBuffer.remaining() / 4)
        mediansByteBuffer.asFloatBuffer().get(medians)
        return medians
    }

    private fun loadMeans(): FloatArray {
        val inputStream = assets.open("feature_means.npy")
        val meansByteArray = inputStream.readBytes()
        val meansByteBuffer = ByteBuffer.wrap(meansByteArray).order(ByteOrder.nativeOrder())
        val means = FloatArray(meansByteBuffer.remaining() / 4)
        meansByteBuffer.asFloatBuffer().get(means)
        return means
    }

    private fun loadScalerParams() {
        val inputStream = assets.open("scaler_params.json")
        val size = inputStream.available()
        val buffer = ByteArray(size)
        inputStream.read(buffer)
        inputStream.close()
        val json = JSONObject(String(buffer))
        val meanJsonArray = json.getJSONArray("mean")
        val scaleJsonArray = json.getJSONArray("scale")

        meanDB = FloatArray(meanJsonArray.length())
        scaleDB = FloatArray(scaleJsonArray.length())

        for (i in 0 until meanJsonArray.length()) {
            meanDB[i] = meanJsonArray.getDouble(i).toFloat()
            scaleDB[i] = scaleJsonArray.getDouble(i).toFloat()
        }
    }

    private fun prepareInputData(locationId: Float, year: Float): ByteBuffer {

        val medians = loadMedians()
        // Create a ByteBuffer to hold the input data
        val byteBuffer = ByteBuffer.allocateDirect(394 * 4)  // Adjust based on your input size
        byteBuffer.order(ByteOrder.nativeOrder())

        // Prepare the input data
        val inputData = FloatArray(394)  // Adjust based on your input size

        // Fill the first two features with the provided values
        inputData[0] = (locationId - meanDB[0]) / scaleDB[0]
            inputData[1] = (year - meanDB[1]) / scaleDB[1]

        // Fill the remaining features with the median values
        for (i in 2 until inputData.size) {
            inputData[i] = (meanDB[i]) / scaleDB [i]
        }

        // Put the input data into the ByteBuffer
        for (value in inputData) {
            byteBuffer.putFloat(value)
        }

        return byteBuffer
    }

    private fun doInference(inputData: ByteBuffer): Float {
        // Allocate space for the output
        val outputData = Array(1) { FloatArray(1) }

        // Run inference
        interpreter.run(inputData, outputData)
        val rawOutput = outputData[0][0]

        // Format the output to 2 decimal places

        return String.format("%.2f", rawOutput).toFloat()
    }

    override fun onDestroy() {
        // Close the interpreter when done
        interpreter.close()
        super.onDestroy()
    }

}
