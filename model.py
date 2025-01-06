import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig
from rdkit import Chem
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # Progress bar for loops
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
from torch.amp import autocast, GradScaler
import os
import matplotlib.pyplot as plt  # For plotting

# Load and preprocess the data
def load_and_preprocess_data(file_path):
    # Process the file and create the DataFrame with reactions
    reaction_smiles = []
    with open(file_path, 'r') as file:
        for line_num, line in enumerate(file, 1):
            reaction = line.strip().rstrip(',')
            if reaction:
                reaction_smiles.append(reaction)
    
    # Create DataFrame with reactions
    df = pd.DataFrame(reaction_smiles, columns=['reaction'])

    # Split reactions into reactants and products
    def split_reactions(reaction):
        try:
            parts = reaction.split('>>', 1)  # Split only on the first occurrence of '>>'
            if len(parts) == 2:
                return pd.Series(parts)
            else:
                return pd.Series([None, None])  # If reaction is not valid
        except Exception as e:
            with open('invalid_smiles_log.txt', 'a') as f:
                f.write(f"Error splitting reaction at line {line_num}: {reaction}, Error: {e}\n")
            return pd.Series([None, None])
    
    df[['reactants', 'products']] = df['reaction'].apply(split_reactions)

    # Canonicalize the SMILES
    def preprocess_smiles(smiles):
        if pd.isna(smiles) or smiles == '':
            return None
        try:
            # Split multi-component SMILES by '.'
            components = smiles.split('.')
            canonical_components = []
            for component in components:
                mol = Chem.MolFromSmiles(component)
                if mol:
                    # Canonicalize each component
                    Chem.SanitizeMol(mol)
                    canonical_components.append(Chem.MolToSmiles(mol, canonical=True))
                else:
                    return None  # Skip invalid SMILES
            return '.'.join(canonical_components)
        except Exception as e:
            with open('invalid_smiles_log.txt', 'a') as f:
                f.write(f"SMILES Parse Error at line {line_num}: {smiles}, Error: {e}\n")
            return None

    # Show progress during preprocessing
    tqdm.pandas(desc="Preprocessing SMILES")
    df['preprocessed_reactants'] = df['reactants'].progress_apply(preprocess_smiles)
    df['preprocessed_products'] = df['products'].progress_apply(preprocess_smiles)

    # Drop rows with invalid SMILES
    df.dropna(subset=['preprocessed_reactants', 'preprocessed_products'], inplace=True)
    
    return df


# Custom dataset
class ChemicalReactionDataset(Dataset):
    def __init__(self, reactants, products, tokenizer, max_length):
        self.reactants = reactants
        self.products = products
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.reactants)

    def __getitem__(self, idx):
        reactant = self.reactants[idx]
        product = self.products[idx]

        encoding = self.tokenizer(
            reactant,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        labels = self.tokenizer(
            product,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )['input_ids']

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': labels.flatten()
        }


# Transformer decoder with attention
class TransformerDecoderWithAttention(nn.Module):
    def __init__(self, vocab_size, hidden_size=768, num_layers=6, num_heads=8, dropout=0.1):
        super(TransformerDecoderWithAttention, self).__init__()
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_size, nhead=num_heads, dropout=dropout),
            num_layers=num_layers
        )
        self.fc_out = nn.Linear(hidden_size, vocab_size)

    def forward(self, decoder_input_ids, memory):
        outputs = self.decoder(tgt=decoder_input_ids, memory=memory)
        return self.fc_out(outputs)


# Combined RoBERTa + Transformer Decoder model with Max Pooling
class RoBERTaWithMaxPoolingAndAttention(nn.Module):
    def __init__(self, roberta_model, decoder, tokenizer, hidden_size=768, max_length=256, dropout_rate=0.1):
        super(RoBERTaWithMaxPoolingAndAttention, self).__init__()
        self.roberta = roberta_model
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.hidden_size = hidden_size
        self.max_length = max_length

        # Define embedding for product tokens (decoder input) to match hidden size
        self.product_embedding = nn.Embedding(self.tokenizer.vocab_size, self.hidden_size)

        # Max pooling layer
        self.pooling = nn.AdaptiveMaxPool1d(256)  # Max pooling to reduce sequence length but keep it compatible

        self.product_embedding_dropout = nn.Dropout(p=dropout_rate)
        self.roberta_output_dropout = nn.Dropout(p=dropout_rate)


    def forward(self, input_ids, attention_mask, products):
        # Encode reactants with RoBERTa (encoder)
        reactant_encoding = self.roberta(input_ids, attention_mask=attention_mask)
        memory = reactant_encoding.last_hidden_state

        memory = self.roberta_output_dropout(memory)

        # Apply pooling layer to reduce sequence length, keep batch and embedding dims intact
        memory = memory.permute(0, 2, 1)  # Change shape to (batch_size, embedding_size, seq_len)
        memory = self.pooling(memory)     # Pooling applied along the sequence length
        memory = memory.permute(0, 2, 1)  # Revert back to (batch_size, seq_len, embedding_size)

        # Encode the products using embedding to match the hidden size
        decoder_input_ids = self.product_embedding(products)
        decoder_input_ids = self.product_embedding_dropout(decoder_input_ids)

        # Pass the encoded memory to the decoder
        outputs = self.decoder(decoder_input_ids, memory)

        return outputs

# Helper function to calculate metrics for each batch
def calculate_metrics(predictions, labels):
    predictions = predictions.cpu().numpy().flatten()
    labels = labels.cpu().numpy().flatten()

    precision = precision_score(labels, predictions, average='macro', zero_division=1)
    recall = recall_score(labels, predictions, average='macro', zero_division=1)
    f1 = f1_score(labels, predictions, average='macro', zero_division=1)

    return precision, recall, f1


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


# Main execution
if __name__ == "__main__":
    # Hyperparameters (alterable)
    num_epochs = 10  # Adjusted for early stopping
    batch_size = 8
    learning_rate = 2e-5
    weight_decay = 1e-2  # Added weight decay
    dropout_rate = 0.1  # Dropout rate
    hidden_size = 768  # Changeable hidden size variable (768 is default for RoBERTa)

    # Paths for model saving
    model_name = r'chemical_reaction_roberta_with_attention_model_3.pth'
    folder_name = r'C:\Users\anind\VS Code\SMILES\models'

    # Check if pretrained model exists
    pretrained = False  # Set to True if you want to load the model, False to train from scratch

    # Load and preprocess data
    df = load_and_preprocess_data(r'C:\Users\anind\VS Code\SMILES\USPTO_50K.txt')

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    # Load pretrained RobertaTokenizer (no custom tokenizer path needed)
    tokenizer = RobertaTokenizer.from_pretrained('seyonec/ChemBERTa-zinc-base-v1')

    # Adjust RoBERTa hidden size dynamically 
    config = RobertaConfig.from_pretrained('seyonec/ChemBERTa-zinc-base-v1', hidden_size=hidden_size)
    roberta_model = RobertaModel(config)

    # Initialize decoder
    vocab_size = len(tokenizer)
    num_layers = 6  # Experiment here
    num_heads = 12  # Experiment here

    # Initialize decoder with the dropout rate and hidden size
    decoder = TransformerDecoderWithAttention(
        vocab_size=vocab_size, 
        hidden_size=hidden_size, 
        num_layers=num_layers, 
        num_heads=num_heads, 
        dropout=dropout_rate)

    # Initialize combined model with max pooling
    model_path = r"C:\Users\anind\VS Code\SMILES\chemical_reaction_roberta_with_attention_model.pth"
    model = RoBERTaWithMaxPoolingAndAttention(
        roberta_model=roberta_model, 
        decoder=decoder, 
        tokenizer=tokenizer, 
        hidden_size=hidden_size, 
        max_length=256,
        dropout_rate=dropout_rate)

    # If pretrained is True, load the pretrained model
    if pretrained:
        print("Loading pretrained model...")
        model = load_pretrained_model(model, model_path)

    # Prepare datasets and dataloaders
    max_length = 256
    train_dataset = ChemicalReactionDataset(
        train_df['preprocessed_reactants'].tolist(), 
        train_df['preprocessed_products'].tolist(), 
        tokenizer, 
        max_length)
    val_dataset = ChemicalReactionDataset(
        val_df['preprocessed_reactants'].tolist(), 
        val_df['preprocessed_products'].tolist(), 
        tokenizer, 
        max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Set up optimizer with weight decay
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Train the model with early stopping
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # If not loading pretrained model, train from scratch
    if not pretrained:
        train_model(model, train_loader, val_loader, optimizer, device, num_epochs=num_epochs, patience=3)

        # Save the model after training
        save_model(model, model_name=model_name, folder_name=folder_name)
