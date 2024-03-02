from datasets import load_dataset
import evaluate
from transformers import (Trainer, TrainingArguments, DataCollatorWithPadding,
                          AutoTokenizer, AutoModelForSequenceClassification)
from sklearn.metrics import confusion_matrix
import numpy as np


def load_data():
    data_files = {
        "train": "../data/train-clean.csv",
        "test": "../data/val-clean.csv",
        "validation": "../data/val-clean.csv",
    }
    return load_dataset("csv", data_files=data_files, delimiter='\t'),


def tokenize(dataset, tokenizer):
    return tokenizer(dataset["text"])


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


def compute_metrics(predictions, labels):
    metrics = [evaluate.load("f1"), evaluate.load("precision"), evaluate.load("recall")]
    computations = [metric.compute(predictions=predictions, references=labels, average='micro') for metric in metrics]
    final_metric = evaluate.load("accuracy")
    computations.append(final_metric.compute(predictions=predictions, references=labels))
    return {
        'accuracy': computations[3]['accuracy'],
        'precision': computations[1]['precision'],
        'recall': computations[2]['recall'],
        'f1': computations[0]['f1'],
    }


def main():
    dataset = load_data()[0]

    pretrained_path = "google-bert/bert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(pretrained_path)

    training_args = TrainingArguments(
        output_dir="model/",
        evaluation_strategy="epoch",
        learning_rate=1e-5,
        per_device_train_batch_size=9,
        per_device_eval_batch_size=9,
        num_train_epochs=1
    )

    dataset = dataset.map(tokenize, batched=True, fn_kwargs={"tokenizer": tokenizer})

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(pretrained_path, num_labels=7)

    for param in model.bert.parameters():
        param.requires_grad = False

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

    # scores, confusion_mat = test_model(trainer, dataset["test"])
    # print(scores, '\n', confusion_mat)


if __name__ == '__main__':
    main()
