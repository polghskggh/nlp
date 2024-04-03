import csv

import numpy as np
from datasets import load_dataset, DatasetDict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, DataCollatorWithPadding, \
    Trainer


def load_module():
    pretrained_path = 'FacebookAI/roberta-base'
    model = AutoModelForSequenceClassification.from_pretrained(pretrained_path)
    return model


# Load data
def load_data():
    data_path = "rajpurkar/squad_v2"
    train = load_dataset(data_path, split='train')
    val = load_dataset(data_path, split='validation')
    dataset = DatasetDict()

    train_clean = {}
    val_clean = {}

    train_clean['text'] = train['context'] + train['question']
    val_clean['text'] = val['context'] + val['question']

    train_clean['label'] = train['is_impossible']
    val_clean['label'] = val['is_impossible']

    dataset['train'], dataset['validation'] = train_clean, val_clean
    return dataset


# prepare header for the data to save
def prepare_header(header: list[str], filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows([header])


# save data
def store_data(data: list[list], filename):
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)
        file.flush()


def tokenize(dataset, tokenizer):
    return tokenizer(dataset['text'], padding=True, truncation=True, return_tensors='pt')


# extract predictions and labels from model output
def prepare_labels(pred):
    logits, labels = pred
    predictions = np.argmax(logits, axis=-1)
    return compute_metrics(predictions, labels)


# compute and save accuracy, precision, recall and F1
def compute_metrics(predictions, labels):
    computations = [accuracy_score(labels, predictions),
                    precision_score(labels, predictions, average='macro'),
                    recall_score(labels, predictions, average='macro'),
                    f1_score(labels, predictions, average='macro')]

    store_data([computations], 'metrics.csv')

    metrics = {
        'accuracy': computations[0],
        'precision': computations[1],
        'recall': computations[2],
        'f1': computations[3],
    }
    return metrics


def train_classifier():
    prepare_header(['accuracy', 'precision', 'recall', 'f1'], 'metrics.csv')

    # prepare tokenizer and model
    pretrained_path = 'FacebookAI/roberta-base'
    tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
    model_classification = load_module()

    # hyperparams
    training_args = TrainingArguments(
        output_dir='model/',
        evaluation_strategy='epoch',
        learning_rate=1e-5,
        per_device_train_batch_size=9,
        per_device_eval_batch_size=9,
        num_train_epochs=10
    )

    dataset = load_data()

    # tokenize data
    dataset = dataset.map(tokenize, batched=True, fn_kwargs={'tokenizer': tokenizer})

    # train the model
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = Trainer(
        model=model_classification,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        tokenizer=tokenizer,
        compute_metrics=prepare_labels,
        data_collator=data_collator,
    )
    trainer.train()
    return model_classification
