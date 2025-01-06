import pandas as pd
from rdkit import Chem
from tqdm import tqdm  # Progress bar for loops

def load_and_preprocess_data(file_path):
    reaction_smiles = []
    with open(file_path, 'r') as file:
        for line_num, line in enumerate(file, 1):
            reaction = line.strip().rstrip(',')
            if reaction:
                reaction_smiles.append(reaction)
    
    df = pd.DataFrame(reaction_smiles, columns=['reaction'])

    def split_reactions(reaction):
        try:
            parts = reaction.split('>>', 1)
            if len(parts) == 2:
                return pd.Series(parts)
            else:
                return pd.Series([None, None])
        except Exception as e:
            with open('invalid_smiles_log.txt', 'a') as f:
                f.write(f"Error splitting reaction at line {line_num}: {reaction}, Error: {e}\n")
            return pd.Series([None, None])

    df[['reactants', 'products']] = df['reaction'].apply(split_reactions)

    def preprocess_smiles(smiles):
        if pd.isna(smiles) or smiles == '':
            return None
        try:
            components = smiles.split('.')
            canonical_components = []
            for component in components:
                mol = Chem.MolFromSmiles(component)
                if mol:
                    Chem.SanitizeMol(mol)
                    canonical_components.append(Chem.MolToSmiles(mol, canonical=True))
                else:
                    return None
            return '.'.join(canonical_components)
        except Exception as e:
            with open('invalid_smiles_log.txt', 'a') as f:
                f.write(f"SMILES Parse Error: {smiles}, Error: {e}\n")
            return None

    tqdm.pandas(desc="Preprocessing SMILES")
    df['preprocessed_reactants'] = df['reactants'].progress_apply(preprocess_smiles)
    df['preprocessed_products'] = df['products'].progress_apply(preprocess_smiles)

    df.dropna(subset=['preprocessed_reactants', 'preprocessed_products'], inplace=True)
    return df
