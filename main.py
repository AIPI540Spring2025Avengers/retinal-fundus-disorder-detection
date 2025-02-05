import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Load the pretrained model
model = models.resnet50()
model.load_state_dict(torch.load('models/toy_resnet50.pth', map_location=torch.device('cpu')))
model.eval()

# Prepare the transformation pipeline
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Streamlit app
st.title('Image Classifier using Pretrained ResNet50')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)

    # Preprocess the image
    input_tensor = transform(image).unsqueeze(0)

    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_idx = torch.max(output, 1)

    # Load ImageNet class labels
    with open('scripts/toy/imagenet_classes.txt', 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    # Display prediction
    st.write(f'Prediction: {classes[predicted_idx]}')
