from datasets import load_dataset
import evaluate
from transformers import (Trainer, TrainingArguments, DataCollatorWithPadding,
                          AutoTokenizer, AutoModelForSequenceClassification)
from sklearn.metrics import confusion_matrix


def load_data():
    data_files = {
        "train": "../data/train-clean.csv",
        "test": "../data/test-clean.csv",
        "validation": "../data/val-clean.csv",
    }
    return load_dataset("csv", data_files=data_files, delimiter='\t'),


def tokenize(dataset, tokenizer):
    return tokenizer(dataset["text"])


def test_model(model, dataset):
    predictions = model.forward(dataset["text"]).logits.argmax(axis=1)
    metrics = compute_metrics(zip(predictions, dataset["label"]))
    confusion_matrix(dataset["label"], predictions)
    return metrics, confusion_matrix


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=1)

    metrics = [evaluate.load("accuracy"), evaluate.load("f1"), evaluate.load("precision"), evaluate.load("recall")]
    comutations = [metric.compute(predictions, labels) for metric in metrics]

    return comutations


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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )
    trainer.train()

    scores, confusion_mat = test_model(trainer.model, dataset["test"])
    print(scores, confusion_mat)


if __name__ == '__main__':
    main()
