import shutil
import os
import torchvision.transforms as transforms
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader 
from PIL import Image
from data_setup import licenceplate
from ultralytics import YOLO 
import requests

os.getcwd()

file_url = r"C:\Users\pault\Documents\5. Projects\5. Licence Plate Detection\data"

data_transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])])

# Create a dataset object with the file path and transformation function
train_dataset = licenceplate(root_dir=file_url, split='train', transform = None)
valid_dataset = licenceplate(root_dir=file_url, split='valid', transform = None)

# Creating data loaders
train_dataloader = DataLoader(train_dataset, batch_size=15, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=15, shuffle=True)



if not os.path.exists('models'):
    os.makedirs('models')
    
model = YOLO('models/yolov8s.pt')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define an optimizer and a loss function here (adjust according to your needs)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_function = torch.nn.CrossEntropyLoss()  # This is a placeholder, replace with the appropriate loss function for your task

# Training loop
num_epochs = 10  # Specify the number of epochs

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for images, targets in train_dataloader:  # Assuming your dataset returns images and their corresponding targets
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]  # Adjust this line if your target format differs

        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        
        # Compute loss
        loss = loss_function(outputs, targets)  # Adjust this line based on how your outputs and targets are structured
        running_loss += loss.item()
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {running_loss/len(train_dataloader)}")
    
    # Validation step
    model.eval()
    val_running_loss = 0.0
    with torch.no_grad():
        for images, targets in valid_dataloader:
            images = images.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)
            loss = loss_function(outputs, targets)
            val_running_loss += loss.item()
            
    print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_running_loss/len(valid_dataloader)}")

print("Training completed.")