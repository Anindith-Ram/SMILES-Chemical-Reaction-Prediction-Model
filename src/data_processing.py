import pandas as pd
from rdkit import Chem
from tqdm import tqdm  # Progress bar for loops

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
