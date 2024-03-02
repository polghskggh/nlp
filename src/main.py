from datasets import load_dataset

from transformers import (Trainer, TrainingArguments, DataCollatorWithPadding,
                          AutoTokenizer, AutoModelForSequenceClassification)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.dummy import DummyClassifier

import numpy as np
import csv


def load_data():
    data_files = {
        "train": "../data/train-clean.csv",
        "test": "../data/test-clean.csv",
        "validation": "../data/val-clean.csv",
    }
    return load_dataset("csv", data_files=data_files, delimiter='\t'),


def tokenize(dataset, tokenizer):
    return tokenizer(dataset["text"], padding=True, truncation=True, return_tensors="pt")


def test_model(trainer, dataset):
    logits, labels, _ = trainer.predict(dataset)
    predictions = np.argmax(logits, axis=-1)
    metrics = compute_metrics(predictions, labels)
    conf_matrix = confusion_matrix(labels, predictions)
    return metrics, conf_matrix


def prepare_labels(pred):
    logits, labels = pred
    predictions = np.argmax(logits, axis=-1)
    return compute_metrics(predictions, labels)


def prepare_header(header: list[str], filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows([header])

    # flush data into file


def store_data(data: list[list], filename):
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)
        file.flush()


def compute_metrics(predictions, labels):
    computations = [accuracy_score(labels, predictions),
                    precision_score(labels, predictions, average='macro'),
                    recall_score(labels, predictions, average='macro'),
                    f1_score(labels, predictions, average='macro')]

    store_data([computations], "metrics.csv")

    metrics = {
        'accuracy': computations[0],
        'precision': computations[1],
        'recall': computations[2],
        'f1': computations[3],
    }
    return metrics


def evaluate_baseline(dataset):
    baseline = DummyClassifier(strategy='most_frequent')
    baseline.fit(dataset['train']['text'], dataset['train']['labels'])
    predictions = baseline.predict(dataset['test']['text'])
    return compute_metrics(predictions, dataset['test']['labels'])


def main():
    dataset = load_data()[0]

    pretrained_path = "FacebookAI/roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(pretrained_path)

    prepare_header(['accuracy', 'precision', 'recall', 'f1'], "metrics.csv")

    prepare_header(["0","1","2","3","4","5","6"], "confusion.csv")

    training_args = TrainingArguments(
        output_dir="model/",
        evaluation_strategy="epoch",
        learning_rate=1e-5,
        per_device_train_batch_size=9,
        per_device_eval_batch_size=9,
        num_train_epochs=10
    )

    dataset = dataset.map(tokenize, batched=True, fn_kwargs={"tokenizer": tokenizer})

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(pretrained_path, num_labels=7)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=prepare_labels,
        data_collator=data_collator,
    )
    trainer.train()

    # final evaluation
    print(evaluate_baseline(dataset))

    scores, confusion_mat = test_model(trainer, dataset["validation"])
    print(scores, '\n', confusion_mat)

    store_data(confusion_mat, "confusion.csv")


if __name__ == '__main__':
    main()
