import datasets
import numpy as np
from transformers import BertTokenizerFast
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification
import json
from transformers import pipeline
from transformers import TrainingArguments, Trainer

data = datasets.load_dataset('Gooogr/pie_idioms')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

def tokenize_and_align_labels(examples):
    tokenized_input = tokenizer(examples['tokens'], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples['ner_tags']):
        word_ids = tokenized_input.word_ids(batch_index=i)
        previous_word_index = None
        word_labels = []
        for index in word_ids:
            if index is None:
                word_labels.append(-100)
            elif index != previous_word_index:
                word_labels.append(label[index])
            else:
                word_labels.append(label[index])
            previous_word_index = index
        labels.append(word_labels)
    tokenized_input['labels'] = labels
    return tokenized_input

def compute_metrics(eval_preds):
    """
    Function to compute the evaluation metrics for Named Entity Recognition (NER) tasks.
    The function computes precision, recall, F1 score and accuracy.

    Parameters:
    eval_preds (tuple): A tuple containing the predicted logits and the true labels.

    Returns:
    A dictionary containing the precision, recall, F1 score and accuracy.
    """
    pred_logits, labels = eval_preds

    pred_logits = np.argmax(pred_logits, axis=2)
    # the logits and the probabilities are in the same order,
    # so we don’t need to apply the softmax

    # We remove all the values where the label is -100
    predictions = [
        [label_list[eval_preds] for (eval_preds, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(pred_logits, labels)
    ]

    true_labels = [
        [label_list[l] for (eval_preds, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(pred_logits, labels)
    ]

    print("Sample true predictions:", predictions[0])
    print("Sample true labels:", true_labels[0])

    results = metric.compute(predictions=predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


label_list = data['train'].features['ner_tags'].feature.names
tokenized_datasets = data.map(tokenize_and_align_labels, batched =True)
model = AutoModelForTokenClassification.from_pretrained('bert-base-uncased', num_labels=3)

metric = datasets.load_metric("seqeval")
data_collator = DataCollatorForTokenClassification(tokenizer)

args = TrainingArguments(
    "test-ner",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

model.save_pretrained('ner_model')
tokenizer.save_pretrained('tokenizer')

id2label = {
    str(i): label for i,label in enumerate(label_list)
}
label2id = {
    label: str(i) for i,label in enumerate(label_list)
}

config = json.load(open('ner_model/config.json'))
config['id2label'] = id2label
config['label2id'] = label2id
json.dump(config, open('ner_model/config.json','w'))






