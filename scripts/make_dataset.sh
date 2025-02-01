#!/bin/bash
curl -L -o ./data/raw/retinal-fundus-images.zip\
  https://www.kaggle.com/api/v1/datasets/download/kssanjaynithish03/retinal-fundus-images
unzip ./data/raw/retinal-fundus-images.zip
mv "Retinal Fundus Images"/* ./data/raw/
rm -rf "Retinal Fundus Images"
rm ./data/raw/retinal-fundus-images.zip
