# BEIT Based Alzheimer's Classification

This project implements an image classification model to detect Alzheimer's disease using the BEIT (Bidirectional Encoder Representations from Image Transformers) model from the Hugging Face library. The model is trained on a dataset of images using PyTorch and various supporting libraries.

## Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Training](#training)
- [Validation](#validation)
- [Testing and Metrics](#testing-and-metrics)
- [Results](#results)
- [Model Saving](#model-saving)
- [Contributing](#contributing)
- [License](#license)

## Installation

To use this project, you'll need to have Python installed along with the following libraries:

- torch
- torchvision
- transformers
- sklearn
- matplotlib

You can install these libraries using pip:

```sh
pip install torch torchvision transformers sklearn matplotlib
```

## Dataset
The dataset should be organized in a folder structure compatible with torchvision.datasets.ImageFolder. Place your dataset in a directory and update the path in the script accordingly.

## Usage
Here is a basic outline of how to use the script:

1. **Define Transformations**: Resize images and convert them to tensors.
2. **Load the Dataset**: Load your dataset using `torchvision.datasets.ImageFolder`.
3. **Split the Dataset**: Split the dataset into training, validation, and test sets.
4. **Create Data Loaders**: Create data loaders for batching and shuffling the data.
5. **Calculate Class Weights**: Calculate class weights to handle class imbalance.
6. **Training Loop**: Train the model with forward and backward passes.
7. **Validation Loop**: Evaluate the model on the validation set.
8. **Testing and Metrics**: Test the model on the test set and calculate metrics.
9. **Save the Model**: Save the trained model to a file.

## Training
To train the model, run the script. Adjust the hyperparameters such as num_epochs, batch_size, and learning rate as needed.
```sh
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to('cpu')
        labels = labels.to('cpu')
        outputs = model(images)
        loss = criterion(outputs.logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Epoch {} completed'.format(epoch + 1))
```
## Validation
Evaluate the model on the validation set and print the accuracy:
```
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
```
## Testing and Metrics
Test the model on the test set, compute the confusion matrix and classification report:
```
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
all_labels = [label.cpu() for label in all_labels]
all_predictions = [prediction.cpu() for prediction in all_predictions]
cm = confusion_matrix(all_labels, all_predictions)
print('Confusion Matrix:')
print(cm)
print(classification_report(all_labels, all_predictions))
```
## Results
The model achieved the following results on the test set:
```
                precision    recall  f1-score   support

           0       0.92      0.89      0.91       123
           1       1.00      0.91      0.95        11
           2       0.95      0.96      0.96       480
           3       0.93      0.93      0.93       346

    accuracy                           0.94       960
   macro avg       0.95      0.92      0.94       960
weighted avg       0.94      0.94      0.94       960

```


## Model Saving
Save the trained model's state dictionary to a file for future use:
```
torch.save(model.state_dict(), 'alzheimer_model.pth')
```
## Contributing
Contributions are welcome! Please open an issue or submit a pull request.

## License
This project is licensed under the MIT License.

