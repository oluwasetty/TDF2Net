# link of dataset: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
# to run this file run command ppip install torch torchvision matplotlib opencv-python Pillow, then python x-ai.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

# --- Configuration ---
DATA_DIR = 'data'
TRAIN_DIR = os.path.join(DATA_DIR, 'training')
TEST_DIR = os.path.join(DATA_DIR, 'testing')
MODEL_PATH = 'brain_tumor_classifier.pth'
NUM_EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# --- Part 1: Model Training ---

def train_model():
    """Trains the CNN and saves the weights."""
    print("--- Starting Model Training ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define transforms for the training and testing sets
    data_transforms = {
        'training': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'testing': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Load the datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x])
                      for x in ['training', 'testing']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
                   for x in ['training', 'testing']}
    
    class_names = image_datasets['training'].classes
    num_classes = len(class_names)
    print(f"Classes found: {class_names}")

    # Load a pre-trained ResNet18 model and modify the final layer
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print('-' * 10)

        model.train()  # Set model to training mode
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders['training']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(image_datasets['training'])
        epoch_acc = running_corrects.double() / len(image_datasets['training'])
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    print("\n--- Training Finished ---")
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    return model, class_names


# --- Part 2: XAI with Grad-CAM ---

class GradCAM:
    """Class for generating Grad-CAM heatmaps."""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.save_activation)
        # Use a hook on the tensor itself for gradients
        # This is more robust than a backward hook on the layer
        
    def save_activation(self, module, input, output):
        self.activations = output
        output.register_hook(self.save_gradient)

    def save_gradient(self, grad):
        self.gradients = grad

    def generate_heatmap(self, input_tensor, class_idx=None):
        self.model.eval()
        self.model.zero_grad()
        
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()

        # Target the specific class output for backpropagation
        one_hot_output = torch.zeros_like(output)
        one_hot_output[0][class_idx] = 1
        output.backward(gradient=one_hot_output, retain_graph=True)
        
        # Get the captured gradients and activations
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]

        # Pool the gradients across the channels
        weights = np.mean(gradients, axis=(1, 2))
        
        # Create the heatmap
        heatmap = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            heatmap += w * activations[i]

        # Apply ReLU and normalize
        heatmap = np.maximum(heatmap, 0)
        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)
        return heatmap, class_idx


def apply_and_visualize_gradcam(model, class_names, image_path):
    """Loads an image, generates a Grad-CAM heatmap, and displays it."""
    print(f"\n--- Generating XAI Explanation for {image_path} ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load and preprocess the image
    img = Image.open(image_path).convert('RGB')
    
    # Use the same transforms as the test set
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(img).unsqueeze(0).to(device)

    # Initialize Grad-CAM
    # For ResNet18, `layer4` is the last convolutional block
    grad_cam = GradCAM(model=model, target_layer=model.layer4)
    
    # Generate heatmap
    heatmap, predicted_class_idx = grad_cam.generate_heatmap(input_tensor)
    predicted_class_name = class_names[predicted_class_idx]
    print(f"Model Prediction: {predicted_class_name}")

    # Visualization
    img_cv = cv2.imread(image_path)
    img_cv = cv2.resize(img_cv, (224, 224))
    
    heatmap_resized = cv2.resize(heatmap, (img_cv.shape[1], img_cv.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    
    superimposed_img = cv2.addWeighted(img_cv, 0.6, heatmap_colored, 0.4, 0)

    # Display results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f"XAI Explanation (Grad-CAM)\nPrediction: {predicted_class_name}", fontsize=16)
    
    ax1.imshow(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    ax1.set_title("Original Image")
    ax1.axis('off')
    
    ax2.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    ax2.set_title("Model Focus Heatmap")
    ax2.axis('off')

    plt.tight_layout()
    plt.show()


# --- Main Execution ---
if __name__ == '__main__':
    # 1. Train the model (or skip if already trained)
    if not os.path.exists(MODEL_PATH):
        train_model()
    else:
        print(f"Found existing model at {MODEL_PATH}, skipping training.")

    # 2. Load the trained model and run XAI
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get class names from the directory structure
    class_names = sorted([d.name for d in os.scandir(TRAIN_DIR) if d.is_dir()])
    num_classes = len(class_names)
    
    # Re-create the model architecture
    model_for_xai = models.resnet18(weights=None) # No pre-trained weights needed here
    num_ftrs = model_for_xai.fc.in_features
    model_for_xai.fc = nn.Linear(num_ftrs, num_classes)
    
    # Load the saved state dictionary
    model_for_xai.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model_for_xai.to(device)
    model_for_xai.eval()

    # 3. Choose an image to explain and visualize the result
    # !!! IMPORTANT: Change this path to an image you want to test !!!
    test_image_path = 'data/testing/glioma/Te-gl_0011.jpg'
    
    if os.path.exists(test_image_path):
        apply_and_visualize_gradcam(model_for_xai, class_names, test_image_path)
    else:
        print(f"\nError: Test image not found at '{test_image_path}'.")
        print("Please update the 'test_image_path' variable in the script to point to a valid image.")