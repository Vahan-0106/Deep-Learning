from Molecule_Examination import Model
from torch.utils.data import DataLoader
from tdc import Evaluator
import torch
from rdkit import RDLogger
from Molecule_Examination import randomize_smiles
RDLogger.DisableLog('rdApp.*')


model = Model(num_layer=6, num_head=8, d_model = 128)
model.load_state_dict(torch.load("Epoch9(Best).pt"))
model.eval()

from Molecule_Examination import TDCWrapperDataset 
from Molecule_Examination import test_df
test_dataset = TDCWrapperDataset(test_df)
test_loader = DataLoader(test_dataset, batch_size=16)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

correct = 0
total = 0
probs = []
y_true = []
with torch.no_grad():
    for i in range(len(test_dataset)):
        smiles, target = test_dataset[i] 
        smiles_list = randomize_smiles(smiles, n_aug=15)
        smiles_list.append(smiles) 

        logits_sum = 0

        for aug_smiles in smiles_list:
            input_list = [aug_smiles] 
            logit = model(input_list) 
            logits_sum += logit.squeeze()

        avg_logit = logits_sum / len(smiles_list)

        pred = (avg_logit > 0.5).float()
        correct += (pred.item() == target)
        total += 1

        prob = torch.sigmoid(avg_logit).item()
        probs.append(prob)
        y_true.append(target)

accuracy = correct / total
print(f"Test Accuracy: {accuracy:.2%}")

evaluator = Evaluator(name='ROC-AUC')
auc_score = evaluator(y_pred=probs, y_true=y_true)
print(f"Leaderboard Metric Score (AUC-ROC): {auc_score:.4}")
