import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

# Define class names manually
class_names = ['Prapty', 'Rezwan']  # Replace with your actual class names

# Define transformations
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(class_names))
model = model.to(device)
model.load_state_dict(torch.load('model/poopy-or-paia-pytorch.pth', map_location=device))


# Define accuracy evaluation function
def evaluate_model(model, dataloader):
    model.eval()
    corrects = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == labels).item()
            total += labels.size(0)
    accuracy = corrects / total
    return accuracy

from torchvision import datasets

# Load the dataset from the 'dataset' folder
data_dir = 'E:/GitHub-rzn/poopy_or_paia/Dataset'
dataset = datasets.ImageFolder(data_dir, transform=data_transforms)

# Display accuracy
def show_accuracy():
    # Dummy dataset loader for accuracy
    # Replace this with a validation dataset loader if you have one
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    accuracy = evaluate_model(model, val_loader)
    #st.write(f'Validation Accuracy: {accuracy:.4f}')

# Predict function
def predict_image(image):
    model.eval()
    image = Image.open(image)
    image = data_transforms(image).unsqueeze(0)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, preds = torch.max(probs, 1)
    return class_names[preds[0]], confidence.item()

# Streamlit interface
st.title("Who are you madafakah!!! 🫵🏻")
st.title("Poopy or Paia? 🤷🏻‍♀️")
st.write("Upload an image to recognize poopy or paia")

show_accuracy()

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Save uploaded file temporarily
    temp_file_path = os.path.join("temp", uploaded_file.name)
    os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    
    # Predict
    prediction, confidence = predict_image(temp_file_path)
    
    # Print specific message for the class 'Prapty'
    if prediction == 'Prapty':
        st.write('Prediction: Bubblegum Babu 🐥')
    else:
        st.write(f'Prediction: {prediction} 😎')
    
    # Show prediction confidence
    st.write(f'Confidence: {confidence:.4f}')

    # Clean up temporary file
    os.remove(temp_file_path)
