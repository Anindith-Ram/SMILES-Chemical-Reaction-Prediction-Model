import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig
from torch.optim import AdamW
from model import RoBERTaWithMaxPoolingAndAttention, TransformerDecoderWithAttention, save_model
from data_processing import load_and_preprocess_data, ChemicalReactionDataset
from train_utils import train_model

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
