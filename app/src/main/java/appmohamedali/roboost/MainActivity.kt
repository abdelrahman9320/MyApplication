package appmohamedali.roboost

import android.graphics.Bitmap
import android.os.Bundle
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.core.content.res.ResourcesCompat
import androidx.core.graphics.drawable.toBitmap
import androidx.core.view.drawToBitmap
import appmohamedali.roboost.ml.Flowers
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.label.ImageLabelerOptionsBase
import com.google.mlkit.vision.label.ImageLabeling
import com.google.mlkit.vision.label.defaults.ImageLabelerOptions
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer


class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.main_activity)
        val processImage: ImageView = findViewById(R.id.processImage)
        val btn: Button = findViewById(R.id.processingbtn)
        var lable: TextView = findViewById(R.id.imageLable)

        var drawable = ResourcesCompat.getDrawable(resources, R.drawable.flower3, null)
        processImage.setImageDrawable(drawable)
        var bitmap: Bitmap = drawable!!.toBitmap(192,192)


        btn.setOnClickListener {
//            var labler = ImageLabeling.getClient(ImageLabelerOptions.DEFAULT_OPTIONS)
//            var input_image = InputImage.fromBitmap(bitmap, 0)
//            var text = ""
//            labler.process(input_image).addOnSuccessListener { lables ->
//                for (l in lables) {
//                    text += "${l.text}  ${l.confidence}"
//                }
//                lable.text = text
//            }
            val image=TensorImage.fromBitmap(bitmap)
            val inputImage=TensorImage.createFrom(image,DataType.FLOAT32)
            val model = Flowers.newInstance(this)
            val flower= arrayListOf<String>("daisy", "dandelion", "roses", "sunflowers", "tulips")
            var text = ""

// Creates inputs for reference.
            val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 192, 192, 3), DataType.FLOAT32)
            inputFeature0.loadBuffer(inputImage.buffer)

// Runs model inference and gets result.
            val outputs = model.process(inputFeature0)
            val outputFeature0 = outputs.outputFeature0AsTensorBuffer
            val percent=outputFeature0.floatArray

// Releases model resources if no longer used.
            model.close()
            for (i in percent.indices) {
                text += "${flower[i]}  ${percent[i]}\n"
            }
            lable.text = text



        }

    }


    fun flowerLabeler(bitmap:Bitmap){

    }

}

