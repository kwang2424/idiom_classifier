import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel, BertConfig
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import torchtext
import datasets
import time

torchtext.disable_torchtext_deprecation_warning()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
torch.cuda.empty_cache()
# for k, v in data.items():
#     if k == 'train':
#         print(data['train'][0]['tokens'])

class Bert(nn.Module):
    def __init__(self, device):
        super(Bert, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        mask_token = torch.tensor(self.tokenizer.mask_token_id).to(device)
        self.linear = nn.Linear(self.bert.config.hidden_size, 1)
        self.softmax = nn.Sigmoid() # Use sigmoid for binary classification
        self.to(device)

        # embedding_size = self.config.to_dict()['hidden_size']
        # print(embedding_size)
        # mask_token = torch.tensor(self.tokenizer.mask_token_id).to(device)
        # # print('...', type(vocab_size))
        # self.linear = nn.Linear(embedding_size, 1)
        # # self.linear = nn.Linear(vocab_size, 1)
        
        # # print('get past here?')
        # self.softmax = nn.Softmax(dim=1)
        # # self.double()

    def forward(self, x, attention_mask=None):
        with torch.no_grad():
            print(x.shape, x, attention_mask.shape, attention_mask)
            # Avoid gradients for BERT since we're not fine-tuning it
            encoded_layers = self.bert(x, attention_mask=attention_mask)
        hidden_states = encoded_layers[0]  # Get the last hidden state
        x = self.linear(hidden_states)
        # print('softmax', nn.Softmax(dim=1)(x), nn.Softmax(dim=1)(x).shape)
        # print(self.softmax(x).shape, self.softmax(x), torch.round(self.softmax(x)).shape, torch.round(self.softmax(x)))
        return torch.round(self.softmax(x))


if __name__ == "__main__":
    # label_pipeline = lambda x: int(x) - 1
    data = datasets.load_dataset('Gooogr/pie_idioms', split="train[:5%]")
    train_testvalid = data.train_test_split(test_size=0.1)
    test_valid = train_testvalid['test'].train_test_split(test_size=0.5)
    train_iter = train_testvalid['train']
    # print(train_testvalid, test_valid)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # tokenizer = get_tokenizer("basic_english")
    # def yield_tokens(data_iter):
    #     print(len(data_iter), type(data_iter), type(data_iter[0]))
    #     for text in data_iter:
    #         yield tokenizer(' '.join(text))
    
    # text_pipeline = lambda x: vocab(tokenizer(x))
    pie_pipeline = lambda x: 1 if x == 'pie' else 0
    # vocab = build_vocab_from_iterator(yield_tokens(train_iter['tokens']), specials=["<unk>"])
    # vocab.set_default_index(vocab["<unk>"])
    # vocab_size = len(vocab)
    # def tokenize_function(examples):
    #     return [tokenizer(e, padding="longest") for e in examples]  # Adjust truncation as needed
    
    def collate_batch(batch):
        idiom, is_pie, texts, ner_tags = zip(*[b.values() for b in batch])

        # Tokenize with BERT tokenizer and handle padding
        tokens = [model.tokenizer(' '.join(x), padding="max_length", truncation=True) for x in texts]
        input_ids = torch.tensor([item['input_ids'] for item in tokens], dtype=torch.int64).to(device)
        attention_mask = torch.tensor([item['attention_mask'] for item in tokens], dtype=torch.float32).to(device)
        # print(attention_mask.type(), input_ids.type())
        # Convert labels to tensors (modify based on your label type)
        is_pie = torch.tensor([pie_pipeline(x) for x in is_pie], dtype=torch.int64)

        return is_pie.to(device), attention_mask, input_ids
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = Bert(device).to(device)

    dataloader = DataLoader(
        train_iter,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_batch
    )
    for i, batch in enumerate(dataloader):
        print(batch)
        break
    
    def train(dataloader):
        model.train()
        total_acc, total_count = 0, 0
        log_interval = 500
        start_time = time.time()
        epoch = 1
        for idx, (is_pie, attention_mask, tokens) in enumerate(dataloader):
            optimizer.zero_grad()
            predicted_label = model(tokens, attention_mask)
            # print(predicted_label, predicted_label.type())
            # predicted_label = torch.where(predicted_label > 0.5, torch.tensor(1, dtype=torch.float32), torch.tensor(0, dtype=torch.float32))
            # print(predicted_label.type())
            loss = criterion(predicted_label, is_pie.unsqueeze(1))
            # print(loss, loss.type())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            print(predicted_label, is_pie.unsqueeze(1), predicted_label.shape, is_pie.unsqueeze(1).shape, is_pie.shape)
            total_acc += (predicted_label == is_pie).sum().item()
            total_count += 1
            if idx % log_interval == 0 and idx > 0:
                elapsed = time.time() - start_time
                print(total_acc, total_count)
                print(
                    "| epoch {:3d} | {:5d}/{:5d} batches "
                    "| accuracy {:8.3f}".format(
                        epoch, idx, len(dataloader), total_acc / total_count
                    )
                )
                total_acc, total_count = 0, 0
                start_time = time.time()

    def evaluate(dataloader):
        model.eval()
        total_acc, total_count = 0, 0

        with torch.no_grad():
            for idx, (is_pie, attention_mask, tokens) in enumerate(dataloader):
                predicted_label = model(tokens, attention_mask)
                loss = criterion(predicted_label, is_pie.unsqueeze(1))
                total_acc += (predicted_label == is_pie).sum().item()
                total_count += 1
        return total_acc / total_count

    # Hyperparameters
    EPOCHS = 1  # epoch
    LR = 5  # learning rate
    BATCH_SIZE = 4  # batch size for training

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
    total_accu = None
    test_iter = test_valid['test']
    
    valid_iter = test_valid['train']
    dataloader = DataLoader(
        train_iter,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_batch
    )
    test_loader = DataLoader(
        test_iter, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch
    )
    # print(test_iter)
    valid_loader = DataLoader(
        valid_iter, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch
    )

    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = time.time()
        train(dataloader)
        accu_val = evaluate(valid_loader)
        if total_accu is not None and total_accu > accu_val:
            scheduler.step()
        else:
            total_accu = accu_val
        print("-" * 59)
        print(
            "| end of epoch {:3d} | time: {:5.2f}s | "
            "valid accuracy {:8.3f} ".format(
                epoch, time.time() - epoch_start_time, accu_val
            )
        )
        print("-" * 59)

    print("Checking the results of test dataset.")
    accu_test = evaluate(test_loader)
    print("test accuracy {:8.3f}".format(accu_test))