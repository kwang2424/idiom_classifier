import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
import numpy as np
import datasets

# Define a dataset class
class TokenizedTextDataset(Dataset):
    def __init__(self, data_frame):
        self.data_frame = data_frame
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]
        tokens = row['tokens']
        label = row['is_pie']

        # Join tokens into a single string
        text = ' '.join(tokens)

        # Tokenize the text into numerical IDs
        inputs = self.tokenizer(
            text, 
            padding="max_length", 
            max_length=128, 
            truncation=True, 
            return_tensors='pt'
        )

        # Return a tuple with the inputs and label
        inputs['label'] = torch.tensor(label, dtype=torch.long)

        return inputs

# Load your data
data = datasets.load_dataset('Gooogr/pie_idioms', split="train[:5%]")
tokens = [x['tokens'] for x in data]
is_pie = [1 if x['is_pie'] else 0 for x in data]
data = pd.DataFrame({
    "tokens": tokens,
    "is_pie": is_pie
})

# Create the dataset and dataloader
dataset = TokenizedTextDataset(data)
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

# Define the model
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Define the optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# Training function
def train_epoch(model, dataloader, optimizer, device):
    model.train()  # Set the model to training mode
    total_loss = 0

    for batch in dataloader:
        input_ids = batch['input_ids'].squeeze(1).to(device)
        attention_mask = batch['attention_mask'].squeeze(1).to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()  # Clear previous gradients

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        loss.backward()  # Backpropagate the loss
        optimizer.step()  # Update model parameters

        total_loss += loss.item()  # Accumulate loss

    return total_loss / len(dataloader)

# Evaluation function
def evaluate(model, dataloader, device):
    model.eval()  # Set the model to evaluation mode
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].squeeze(1).to(device)
            attention_mask = batch['attention_mask'].squeeze(1).to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    return accuracy, precision, recall, f1

# Main training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 5

for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, optimizer, device)

    print(f"Epoch {epoch + 1}/{num_epochs}")
    print(f"Train Loss: {train_loss}")

    # Evaluation
    accuracy, precision, recall, f1 = evaluate(model, train_loader, device)

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
