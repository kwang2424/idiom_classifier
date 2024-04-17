import datasets
import numpy as np
from transformers import BertTokenizerFast
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification
import json
from transformers import pipeline
from transformers import TrainingArguments, Trainer

data = datasets.load_dataset('Gooogr/pie_idioms')
tokenizer = BertTokenizerFast.from_pretrained('tokenizer')


model_fine_tuned = AutoModelForTokenClassification.from_pretrained('ner_model')

nlp = pipeline('ner', model=model_fine_tuned, tokenizer=tokenizer)
example1 = "I bought a piece of cake for my mom's birthday"
result = nlp(example1)
print(result)