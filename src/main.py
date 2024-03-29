from datasets import load_dataset, DatasetDict

from transformers import (Trainer, TrainingArguments, DataCollatorWithPadding,
                          AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.dummy import DummyClassifier

import numpy as np
import csv


def load_module(model_type: str = 'seq2seq'):
    if model_type == 'seq2seq':
        pretrained_path = 'google/roberta2roberta_L-24_discofuse'
        model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_path)
    else:
        pretrained_path = 'FacebookAI/roberta-base'
        model = AutoModelForSequenceClassification.from_pretrained(pretrained_path)
    return model


# load the datasets
def load_data():
    data_path = "rajpurkar/squad"
    train = load_dataset(data_path, split='train[:1000]')
    val = load_dataset(data_path, split='validation[:100]')
    dataset = DatasetDict()
    dataset['train'], dataset['validation'] = train, val
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


# make final evaluation of the model
def test_model(trainer, dataset):
    logits, labels, _ = trainer.predict(dataset)
    predictions = np.argmax(logits, axis=-1)
    metrics = compute_metrics(predictions, labels)
    conf_matrix = confusion_matrix(labels, predictions)
    return metrics, conf_matrix


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


# evaluate the baseline model on the metrics
def evaluate_baseline(dataset):
    baseline = DummyClassifier(strategy='most_frequent')
    baseline.fit(dataset['train']['text'], dataset['train']['labels'])
    predictions = baseline.predict(dataset['test']['text'])
    return compute_metrics(predictions, dataset['test']['labels'])


def main():
    # prepare data
    prepare_header(['accuracy', 'precision', 'recall', 'f1'], 'metrics.csv')
    prepare_header(['0', '1', '2', '3', '4', '5', '6'], 'confusion.csv')

    # prepare tokenizer and model
    pretrained_path = 'FacebookAI/roberta-base'
    tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
    model_classification = load_module('classification')
    model_seq2seq = load_module('seq2seq')
    # hyperparams
    training_args = TrainingArguments(
        output_dir='model/',
        evaluation_strategy='epoch',
        learning_rate=1e-5,
        per_device_train_batch_size=9,
        per_device_eval_batch_size=9,
        num_train_epochs=10
    )
    return
    dataset = load_data()

    # tokenize data
    dataset = dataset.map(tokenize, batched=True, fn_kwargs={'tokenizer': tokenizer})

    # train the model
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        tokenizer=tokenizer,
        compute_metrics=prepare_labels,
        data_collator=data_collator,
    )
    trainer.train()

    # final evaluation
    print('Baseline Metrics:', evaluate_baseline(dataset))
    scores, confusion_mat = test_model(trainer, dataset['test'])
    store_data(confusion_mat, 'confusion.csv')


if __name__ == '__main__':
    main()
