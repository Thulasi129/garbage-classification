import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
import os
import json
from collections import Counter
from sklearn.metrics import classification_report

def train_model():
    # Define paths
    dataset_dir = 'dataset'
    model_save_path = 'garbage_classifier_cnn.pth'
    class_names_path = 'class_names.json'

    # Check for CUDA availability
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # Define transformations for the training and validation sets
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load the dataset
    full_dataset = datasets.ImageFolder(dataset_dir, data_transforms)

    # Split the dataset into training and validation sets (80% train, 20% validation)
    train_size = int(0.20 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    # Get the true labels for the training subset
    train_labels = [full_dataset.targets[i] for i in train_dataset.indices]
    
    # Count occurrences of each class
    class_counts = Counter(train_labels)
    num_classes = len(full_dataset.classes)
    class_sample_counts = torch.tensor([class_counts[i] for i in range(num_classes)], dtype=torch.double)



    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Get class names and save them
    class_names = full_dataset.classes
    with open(class_names_path, 'w') as f:
        json.dump(class_names, f)
    print(f"Class names saved to {class_names_path}")
    print(f"Classes: {class_names}")

    # Load a pre-trained ResNet-18 model
    model = models.resnet18(pretrained=True)

    # Freeze all the parameters in the pre-trained model
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the last convolutional layer (layer4)
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Get the number of input features for the classifier
    num_ftrs = model.fc.in_features

    # Replace the final fully connected layer with a new one for our number of classes
    model.fc = nn.Linear(num_ftrs, len(class_names))

    # Move the model to the designated device
    model = model.to(device)

    # Define the loss function and optimizer
    # Define priority class and weight
    priority_class = 'non-recyclable'
    priority_weight = 3.0

    # Calculate class weights for the loss function
    base_weights = 1.0 / class_sample_counts
    
    # Get class names to find the index of the priority class
    class_names_for_weights = full_dataset.classes
    
    # Find the index of the priority class
    priority_class_index = -1
    if priority_class in class_names_for_weights:
        priority_class_index = class_names_for_weights.index(priority_class)
    
    # Create a tensor for weights
    class_weights = torch.tensor(base_weights, dtype=torch.float32)
    
    # If the priority class is found, apply the priority weight
    if priority_class_index != -1:
        class_weights[priority_class_index] *= priority_weight
        print(f"Applying priority weight of {priority_weight} to class '{priority_class}' at index {priority_class_index}")

    # Normalize weights so that they sum to num_classes
    class_weights = class_weights / class_weights.sum() * num_classes

    # Move weights to the correct device
    class_weights = class_weights.to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Update the optimizer to include the unfrozen layer's parameters
    optimizer = optim.Adam([
        {'params': model.layer4.parameters()},
        {'params': model.fc.parameters()}
    ], lr=0.001)

    # --- Training Loop ---
    num_epochs = 1
    print("Starting training...")

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        running_corrects = 0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            if (i+1) % 10 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {i+1}/{len(train_loader)} - Loss: {loss.item():.4f}')


        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)

        print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    # Save the model
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}')

    print('Training finished.')

    # --- Validation Loop ---
    model.eval()  # Set model to evaluate mode
    val_running_loss = 0.0
    val_running_corrects = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            val_running_loss += loss.item() * inputs.size(0)
            val_running_corrects += torch.sum(preds == labels.data)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss = val_running_loss / len(val_dataset)
    val_acc = val_running_corrects.double() / len(val_dataset)
    print(f'Epoch {epoch+1}/{num_epochs} - Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
    
    # Print classification report
    # print(classification_report(all_labels, all_preds, target_names=class_names))
    
    print('-' * 20)

if __name__ == '__main__':
    train_model()