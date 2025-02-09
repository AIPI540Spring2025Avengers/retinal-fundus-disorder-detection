import cv2
import argparse
from model import classify_fundus_image

#Argument Parser for Image Path
parser = argparse.ArgumentParser(description="Test Fundus Image Classification Model")
parser.add_argument("image_path", type=str, help="Path to the fundus image")
args = parser.parse_args()

#Load Image
image = cv2.imread(args.image_path)

if image is None:
    print("Error: Unable to load image. Please check the file path.")
else:
    print(f"Predicted Category: {classify_fundus_image(image)}")
