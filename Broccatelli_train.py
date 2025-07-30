import re
import torch
from torch import nn
import torch.nn.functional as F
from tdc.single_pred import ADME
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tdc import Evaluator
import torch.optim as optim
import math
from rdkit import Chem
from rdkit.Chem import Descriptors
import numpy as np
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from sklearn.model_selection import StratifiedKFold
from rdkit.Chem import AllChem

#Used for augmentation
def randomize_smiles(smiles, n_aug=10):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    
    augmented = set()
    for _ in range(n_aug):
        rand_smiles = Chem.MolToSmiles(mol, doRandom=True)
        augmented.add(rand_smiles)
    return list(augmented)

#Custom Dataset
class TDCWrapperDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        smiles = self.data.iloc[idx]['Drug']
        label = self.data.iloc[idx]['Y']
        return smiles, label

#Loading and splitting data
data = ADME(name = 'pgp_broccatelli')
split = data.get_split()

train_df = split['train']
valid_df = split['valid']
test_df = split['test']

smiles_list = train_df['Drug'].to_list()

targets = train_df['Y'].to_list()

augmented_data = []

for _, row in train_df.iterrows():
    smiles = row['Drug']
    label = row['Y']
    
    augmented_data.append((smiles, label))  # original

    for new_smiles in randomize_smiles(smiles, n_aug=12):
        augmented_data.append((new_smiles, label)) 

#Adding new SMILES into the train dataset
import pandas as pd
train_df = pd.DataFrame(augmented_data, columns=['Drug', 'Y'])

#Regex tokenizer
def tokenize_smiles(smiles):
    pattern =  "(?:\[[^\[\]]{1,10}\])" + "|" + \
               "Cl|Br" + "|" + \
               "\%\d{2}|\d" + "|" + \
               "\=|\#|\-|\+|\\\\|\/|\(|\)|\.|:|~|@|\?|>|<|\*|\$|\!|\||\{|\}|" + \
               "[A-Z][a-z]?" 

    regex = re.compile(pattern)
    tokens = regex.findall(smiles)
    
    return tokens

#Creating vocabulary based on the augmented training dataset
vocab = {
    '<PAD>' : 0,
    '<UNK>' : 2,
    '<CLS>' : 1
}
k = 3
for i in range(len(smiles_list)):
    for token in tokenize_smiles(smiles_list[i]):
        if token not in vocab:
            vocab[token] = k
            k += 1

M = len(vocab)

#Loading additional molecule features to use in the classifier layer
def get_rdkit_features(smiles_list):
    features = []
    for smiles in smiles_list:
        if isinstance(smiles, tuple): 
            smiles = smiles[0]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            features.append([0.0] * 12) 
        else:
            features.append([
                Descriptors.MolWt(mol),
                Descriptors.MolLogP(mol),
                Descriptors.NumRotatableBonds(mol),
                Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol),
                Descriptors.TPSA(mol),
                Descriptors.NumAliphaticRings(mol),
                Descriptors.NumAromaticRings(mol),
                Descriptors.FractionCSP3(mol),
                Descriptors.NumHeteroatoms(mol),
                Descriptors.HeavyAtomCount(mol),
                Descriptors.NumValenceElectrons(mol)
            ])
    return torch.tensor(features, dtype=torch.float)

#Positional encoding model
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  

        pe[:, 1::2] = torch.cos(position * div_term) 

        pe = pe.unsqueeze(0)  
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].detach() 

#Tokenizing and determining the padding for gven smiles
def prep_molecule(smiles):
    result, all_indices = [], []
    max = 0
    for smi in smiles:
        tokens = tokenize_smiles(smi)
        if len(tokens) > max:
            max = len(tokens)

        indices = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
        all_indices.append(indices)
    
    paddings = []

    for indices in all_indices:
        indices += [vocab['<PAD>']] * (max - len(indices))
        padding = [False]* (max + 1)
        for i in range(len(indices),max):
            padding[i + 1] = True
        paddings.append(padding)
    first_column = [1] * len(all_indices)
    result = [first_column[i:i+1]+row for i, row in enumerate(all_indices)]
    return result, paddings

class Model(nn.Module):
    def __init__(self, num_layer, num_head, d_model):
        super().__init__()
        self.feature_mlp = nn.Sequential(
            nn.Linear(12, 48),
            nn.ReLU(48),
            nn.Linear(48,12),
            nn.ReLU(12)
        )
        self.Positional = PositionalEncoding(d_model, 256)
        self.Embedding = nn.Embedding(M, d_model)
        encoder = nn.TransformerEncoderLayer(d_model, num_head, 4 * d_model,0.2,batch_first=True, norm_first=True)
        layer_norm = nn.LayerNorm(d_model)
        self.TransformerEncoder = nn.TransformerEncoder(encoder, num_layer, layer_norm)
        self.classifier = nn.Sequential(
            nn.Linear(d_model+12, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 1)
        )
        self.norm = nn.BatchNorm1d(num_features=12)

    def forward(self, X):
        features = get_rdkit_features(X)
        X, padding = prep_molecule(X)
        X = torch.tensor(X)
        
        features = self.norm(features)
        add_features = self.feature_mlp(features)
        embedding = self.Embedding(X)
        
        input = self.Positional(embedding)
        padding = torch.tensor(padding)
        output = self.TransformerEncoder(input, src_key_padding_mask = padding)

        cls = output[:,0,:]

        result = torch.cat([cls, add_features], dim = 1)

        logits = self.classifier(result)
        return logits
    
if __name__ == "__main__":  
    num_epochs = 15
    d_model = 128
    num_head = 8
    num_layers = 6

    model = Model(num_layer=num_layers, num_head=num_head, d_model = d_model)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5   , weight_decay=5e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5, verbose=True)

    train_dataset = TDCWrapperDataset(train_df)
    test_dataset = TDCWrapperDataset(valid_df)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, drop_last=True)

    max_score = 0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for smiles, labels in train_loader:
            labels  = labels.float().view(-1, 1)

            optimizer.zero_grad()
            outputs = model(smiles)                  
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            preds = torch.round(torch.sigmoid(outputs))
            train_correct += (preds == labels).sum().item()
            train_total   += labels.size(0)

        train_loss /= len(train_loader)
        train_acc   = 100.0 * train_correct / train_total

        val_loss = 0.0
        val_correct = 0
        val_total = 0

        probs = []
        y_true = []
        model.eval()
        with torch.no_grad():
            for smiles, labels in test_loader:
                labels  = labels.float().view(-1, 1)
                y_true.extend(labels)
                outputs = model(smiles)
                probs.extend(outputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                preds = torch.round(torch.sigmoid(outputs))
                val_correct += (preds == labels).sum().item()
                val_total   += labels.size(0)

        val_loss /= len(test_loader)
        val_acc   = 100.0 * val_correct / val_total
        evaluator = Evaluator(name='ROC-AUC')
        
        score = evaluator(y_pred=probs, y_true=y_true)
        torch.save(model.state_dict(), f"_Epoch{epoch + 1}.pt")
        if max_score < score:
            torch.save(model.state_dict(), "Hehe_state.pt")

            max_score = score
        print(f"Epoch [{epoch+1}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%  "
            f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.2f}%  Val AUC-ROC: {score:.4f}")
        
        scheduler.step(val_loss)

