# OCR_with_TFOD_and_EasyOCR
TFOD and EasyOCR for a robust OCR engine

<span align="left">
  <img width="600" heigt="300" src="https://github.com/Nnamaka/OCR_with_TFOD_and_EasyOCR/blob/main/annotating%20(1).gif">
</span>


# EasyOCR

<p align="left">
  <img width="300" heigt="300" src="https://github.com/Nnamaka/OCR_with_TFOD_and_EasyOCR/blob/main/easyocr.png">
</p>



<a href="https://github.com/JaidedAI/EasyOCR">EasyOCR</a> is a deep learning model trained for OCR(optical character recognition). It's code base is based on the pytorch framework. The model is able to recognize 83+ languages.
  
  
# Introduction
Optical character recognition is the conversion of images of typed, handwritten, or printed text into machine-encoded text, whether from a scanned document, a photo of a document, a scene photo, or subtitle text superimposed on an image.
The OCR application developed here combines TFOD and EasyOCR to create a robust OCR system.
 
<i>This README is a brief walkthrough of the major steps carried out to create this application. Refer to <a href="https://github.com/Nnamaka/OCR_with_TFOD_and_EasyOCR/blob/main/TFOD_and_EasyOCR.ipynb">TFOD_and_EasyOCR.ipynb</a> for the full procedures</i>



I used the <a href="https://github.com/tzutalin/labelImg">labellimg</a> tool to label and annotate my images.
My images are saved in the pascalVOC format and transformed to TFRecords to be fed into the TFOD pipeline.

# Steps

## step 1 - Download the TFOD repo and requirments
<pre>
if not os.path.exists(os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection')):
    !git clone https://github.com/tensorflow/models {paths['APIMODEL_PATH']}
</pre>
After that we:
* Install TFOD
* Install Dependencies
* Run Verification Script
* Creat Label map and TFRecords
* Train and Evaluate the model


## step 2 - Install EasyOCR and Import it to our enviroment
<pre>
!pip install easyocr
</pre>
<pre>
import easyocr
</pre>

## Step 3 - Filter the detections from our TFOD model
<pre>
scores = list(filter(lambda x: x >thresh, detections['detection_scores']))
boxes = detections['detection_boxes'][:len(scores)]
classes = detections['detection_classes'][:len(scores)]
</pre>

## step 4 - Make inference on the OCR Model
Now we loop throug the detection(s) to get our final text recognition.

<i>
<b>Note: We need to Renormalize the detection box:</b>
  
<p>The coordinates of the bounding box from the output of the TFOD pipeline needs to be renormalized in other to correspond with the original image size.
This is done because the image document fed into the TFOD model was pre-processed and transformed. This reduces the image size and now the final Output bounding box coordinates now reflects the size of the pre-processed image, which is not what we want.
</p>
</i>
<pre>
height, width = image_np_with_detections.shape[0], image_np_with_detections.shape[1]
</pre>



<pre>
for idx, box in enumerate(boxes):
  roi = box * [height, width, height, width]
  region = image_np_with_detections[int(roi[0]) : int(roi[2]), int(roi[1]) : int(roi[3])]
  ocr_result = reader.readtext(region)
  print(ocr_result)
</pre>
