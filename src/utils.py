import torch
import os
import matplotlib.pyplot as plt

# Helper function to calculate metrics for each batch
def calculate_metrics(predictions, labels):
    predictions = predictions.cpu().numpy().flatten()
    labels = labels.cpu().numpy().flatten()

    precision = precision_score(labels, predictions, average='macro', zero_division=1)
    recall = recall_score(labels, predictions, average='macro', zero_division=1)
    f1 = f1_score(labels, predictions, average='macro', zero_division=1)

    return precision, recall, f1

# Plotting function for metrics
def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, train_f1s, val_f1s):
    epochs = range(1, len(train_losses) + 1)

    # Plot Loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Plot Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_accuracies, label='Training Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Plot F1 Score
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_f1s, label='Training F1 Score')
    plt.plot(epochs, val_f1s, label='Validation F1 Score')
    plt.title('Training and Validation F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.show()


# Function to save the model
def save_model(model, model_name="model.pth", folder_name="models"):
    # Create 'models' folder if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    # Save model to the specified folder
    model_path = os.path.join(folder_name, model_name)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
