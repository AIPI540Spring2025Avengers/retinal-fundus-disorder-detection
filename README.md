# retinal-fundus-disorder-detection
Computer vision project to detect and classify disorders from retinal fundus images.

Fundus Image Classification - Rule-Based Model
This repository contains a rule-based approach for classifying retinal fundus images using handcrafted features. The classification is performed by the classify_fundus_image() function in model.py.

📌 Function: classify_fundus_image(image)
🔹 Description
The function extracts image-based features from a given retinal fundus image and applies a set of rule-based conditions to classify it into one of several categories.

🔹 Input
image: A loaded OpenCV image object (cv2.imread() format).
The function expects an image object (not a file path).
🔹 Output
Returns: A predicted category (str) for the image, which could be one of the following:
"Dry AMD"
"Wet AMD"
"Mild DR"
"Moderate DR"
"Severe DR"
"Proliferate DR"
"Cataract"
"Glaucoma"
"Pathological Myopia"
"Normal Fundus"
"Uncertain - Further Review Needed"
Error Handling: Returns "Error - Unable to Load" if the image is invalid.