#!/usr/bin/env python
# coding: utf-8

# In[9]:


# Import necessary libraries
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, roc_curve
import matplotlib.pyplot as plt
from transformers import AutoModelForImageClassification as BeitForImageClassification
from collections import Counter
from sklearn.model_selection import train_test_split

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# Load the dataset
full_data = torchvision.datasets.ImageFolder(root='Dataset', transform=transform)


# Split the dataset into train, validation, and test sets
train_data, temp_data = train_test_split(full_data, test_size=0.3)
val_data, test_data = train_test_split(temp_data, test_size=0.5)


# Create data loaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=True)


# Calculate class weights
class_counts = Counter(img_tuple[1] for img_tuple in full_data.imgs)
class_weights = {cls: len(full_data) / count for cls, count in class_counts.items()}
class_weights = [class_weights[i] for i in range(len(class_weights))]


# Convert to tensor
class_weights = torch.tensor(class_weights, dtype=torch.float)



# Move to GPU if available
# if torch.cuda.is_available():
class_weights = class_weights.to('cuda')


from torch import nn

# Load the model
model = BeitForImageClassification.from_pretrained('microsoft/beit-base-patch16-224')


num_classes = len(class_weights) 
model.classifier = nn.Linear(model.classifier.in_features, num_classes)


# Define loss function with class weights and optimizer
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# Move the model to the GPU
model = model.to('cuda')



num_epochs = 200


# Training loop
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to('cuda')
        labels = labels.to('cuda')
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs.logits, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Epochs Done Till Now:', epoch+1)


# Validation loop
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in val_loader:
        images = images.to('cuda')
        labels = labels.to('cuda')
        outputs = model(images)
        _, predicted = torch.max(outputs.logits.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Validation Accuracy: {} %'.format(100 * correct / total))


# Testing loop and metrics
model.eval()
all_labels = []
all_predictions = []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to('cuda')
        labels = labels.to('cuda')
        outputs = model(images)
        _, predicted = torch.max(outputs.logits.data, 1)
        all_labels.extend(labels)
        all_predictions.extend(predicted)


# Move tensors to CPU memory
all_labels = [label.cpu() for label in all_labels]
all_predictions = [prediction.cpu() for prediction in all_predictions]

# Confusion matrix
cm = confusion_matrix(all_labels, all_predictions)
print('Confusion Matrix:')
print(cm)

from sklearn.metrics import classification_report

# Print the classification report
print(classification_report(all_labels, all_predictions))



torch.save(model.state_dict(), 'alzheimer_model.pth')

