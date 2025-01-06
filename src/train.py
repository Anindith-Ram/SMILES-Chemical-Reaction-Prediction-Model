import torch
from torch.amp import autocast, GradScaler
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

# Early stopping function
def early_stopping(val_loss, best_loss, patience_counter, patience):
    if val_loss < best_loss:
        patience_counter = 0
        best_loss = val_loss
    else:
        patience_counter += 1
    return patience_counter, best_loss, patience_counter >= patience


# Function to load pretrained model (no tokenizer path needed anymore)
def load_pretrained_model(model, model_path):
    # Load model weights
    model.load_state_dict(torch.load(model_path))
    return model


# Model training function with progress bar, loss, and accuracy tracking
def train_model(model, train_loader, val_loader, optimizer, device, num_epochs=5, patience=3):
    model.to(device)
    scaler = GradScaler('cuda')

    # Lists to track metrics over epochs
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    train_f1s = []
    val_f1s = []

    with open('training_results.txt', 'w') as log_file:  # Log the results to a file
        best_loss = np.inf
        patience_counter = 0

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            total_correct = 0
            total_tokens = 0

            # Tracking precision, recall, f1
            total_precision = 0
            total_recall = 0
            total_f1 = 0
            batches = 0

            train_pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}")

            for batch in train_pbar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                products = batch['labels'].to(device)

                optimizer.zero_grad()

                # Forward pass with AMP
                with torch.amp.autocast('cuda'):
                    outputs = model(input_ids, attention_mask, products)
                    loss = nn.CrossEntropyLoss()(outputs.view(-1, outputs.size(-1)), products.view(-1))

                # Backward pass with AMP
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                total_loss += loss.item()

                # Calculate accuracy
                _, predictions = torch.max(outputs, dim=-1)
                total_correct += (predictions == products).sum().item()
                total_tokens += products.numel()

                # Calculate precision, recall, and F1 for the batch
                precision, recall, f1 = calculate_metrics(predictions, products)
                total_precision += precision
                total_recall += recall
                total_f1 += f1
                batches += 1

            avg_train_loss = total_loss / len(train_loader)
            avg_train_accuracy = total_correct / total_tokens
            avg_train_f1 = total_f1 / batches

            # Append to lists for graphing
            train_losses.append(avg_train_loss)
            train_accuracies.append(avg_train_accuracy)
            train_f1s.append(avg_train_f1)

            # Log training results
            log_file.write(f"Epoch {epoch+1}/{num_epochs}:\n")
            log_file.write(f"Training - Loss: {avg_train_loss:.4f}, Accuracy: {avg_train_accuracy:.4f}, F1: {avg_train_f1:.4f}\n")

            print(f"Training Results - Loss: {avg_train_loss:.4f}, Accuracy: {avg_train_accuracy:.4f}, F1: {avg_train_f1:.4f}")

            # Validation
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total_tokens = 0
            val_f1 = 0
            val_batches = 0

            with tqdm(total=len(val_loader), desc="Validating", unit="batch") as pbar:
                with torch.no_grad():
                    for batch in val_loader:
                        input_ids = batch['input_ids'].to(device)
                        attention_mask = batch['attention_mask'].to(device)
                        products = batch['labels'].to(device)

                        # Forward pass
                        outputs = model(input_ids, attention_mask, products)
                        val_loss += nn.CrossEntropyLoss()(outputs.view(-1, outputs.size(-1)), products.view(-1)).item()

                        # Calculate accuracy
                        _, predictions = torch.max(outputs, dim=-1)
                        val_correct += (predictions == products).sum().item()
                        val_total_tokens += products.numel()

                        # Calculate precision, recall, and F1 for validation
                        precision, recall, f1 = calculate_metrics(predictions, products)
                        val_f1 += f1
                        val_batches += 1

                        pbar.update(1)

            avg_val_loss = val_loss / len(val_loader)
            avg_val_accuracy = val_correct / val_total_tokens
            avg_val_f1 = val_f1 / val_batches

            # Append to lists for graphing
            val_losses.append(avg_val_loss)
            val_accuracies.append(avg_val_accuracy)
            val_f1s.append(avg_val_f1)

            # Log validation results
            log_file.write(f"Validation - Loss: {avg_val_loss:.4f}, Accuracy: {avg_val_accuracy:.4f}, F1: {avg_val_f1:.4f}\n")
            print(f"Validation Results - Loss: {avg_val_loss:.4f}, Accuracy: {avg_val_accuracy:.4f}, F1: {avg_val_f1:.4f}")

            # Check for early stopping
            patience_counter, best_loss, stop = early_stopping(avg_val_loss, best_loss, patience_counter, patience)
            if stop:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # After training completes, plot the graphs
    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, train_f1s, val_f1s)
