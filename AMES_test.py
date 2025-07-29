model = Model(num_layer=6, num_head=8, d_model=128).to(device)
model.load_state_dict(torch.load("State.pt", map_location=device))
model.eval()

test_dataset = TDCWrapperDataset(test_df)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Number of trainable parameters: {count_parameters(model)}")

correct = 0
total = 0
probs = []
y_true = []

with torch.no_grad():
    for i in range(len(test_dataset)):
        smiles, target = test_dataset[i]
        smiles_list = randomize_smiles(smiles, n_aug=8)
        smiles_list.append(smiles)  # include the original

        logits_sum = 0.0

        # For each SMILES, pass a singleton list into the model
        for aug_smiles in smiles_list:
            aug_input = [aug_smiles]  # make it a batch of 1
            logit = model(aug_input).squeeze().cpu()
            logits_sum += logit

        avg_logit = logits_sum / len(smiles_list)

        pred = (avg_logit > 0.5)
        correct += int(pred.item() == float(target))
        total += 1

        prob = torch.sigmoid(avg_logit).item()
        probs.append(prob)
        y_true.append(target)

accuracy = correct / total
print(f"Test Accuracy: {accuracy:.2%}")

evaluator = Evaluator(name='ROC-AUC')
auc_score = evaluator(y_pred=probs, y_true=y_true)
print(f"Leaderboard Metric Score (AUC-ROC): {auc_score:.4f}")
