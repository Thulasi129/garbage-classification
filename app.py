import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json

# --- 1. Load Model, Class Names, and Define Transformations ---
MODEL_PATH = 'garbage_classifier_cnn.pth'
CLASS_NAMES_PATH = 'class_names.json'

# Check for CUDA availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load class names
try:
    with open(CLASS_NAMES_PATH, 'r') as f:
        class_names = json.load(f)
    NUM_CLASSES = len(class_names) # Dynamically set NUM_CLASSES
except FileNotFoundError:
    print(f"Error: '{CLASS_NAMES_PATH}' not found.")
    print("Please run the 'train_cnn.py' script first to generate the class names.")
    exit()

# Re-create the model structure (ResNet-18)
model = models.resnet18(weights=None) # Using weights=None as we are loading our own
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, NUM_CLASSES)

# Load the trained model state
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
except FileNotFoundError:
    print(f"Error: Model file '{MODEL_PATH}' not found.")
    print("Please run the 'train_cnn.py' script first to train and save the model.")
    exit()

model.to(device)
model.eval()  # Set the model to evaluation mode

# Define the image transformations (must be the same as during training)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

print("PyTorch model and class names loaded successfully.")

# --- 2. Define the Prediction Function ---
def predict(image):
    """
    Predicts the class of an uploaded image.
    'image' is a NumPy array from the Gradio interface.
    """
    try:
        # Convert NumPy array to PIL Image
        img_pil = Image.fromarray(image.astype('uint8'), 'RGB')
        
        # Preprocess the image and add a batch dimension
        img_tensor = preprocess(img_pil).unsqueeze(0)
        
        # Move tensor to the correct device
        img_tensor = img_tensor.to(device)
        
        # Make a prediction
        with torch.no_grad():
            outputs = model(img_tensor)
            
            # Apply softmax to get probabilities
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # Get the top prediction
            top_prob, top_catid = torch.topk(probabilities, 1)

        # Prepare the output for the Gradio Label component, showing only the top prediction
        top_class_name = class_names[top_catid.item()]
        confidences = {top_class_name: float(top_prob.item())}
        
        return confidences

    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return {"Error": 1.0}

# --- 3. Create and Launch the Gradio Interface ---
print("Launching the Gradio interface...")

# Define the input and output components
image_input = gr.Image(label="Upload Garbage Image")
label_output = gr.Label(num_top_classes=4, label="Predictions")

# Create the interface
iface = gr.Interface(
    fn=predict,
    inputs=image_input,
    outputs=label_output,
    title="Garbage Classification",
    description="Upload an image of garbage to classify it into one of four categories: Biodegradable, Non-Biodegradable, Recyclable, or Non-Recyclable.",
    examples=[
        ["dataset/biodegradable/biological_1.jpg"],
        ["dataset/recyclable/cardboard_1.jpg"]
    ]
)

# Launch the web server
if __name__ == "__main__":
    iface.launch(inbrowser=True, share=True)