from datasets import load_dataset
from transformers import (Trainer, AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, DataCollatorWithPadding)


def load_data():
    data_files = {
        "train": "data/train-emotion-clean.tsv",
        "test": "data/test-emotion-clean.tsv",
        "validation": "data/val-emotion-clean.tsv",
    }
    return load_dataset("csv", data_files=data_files, delimiter='\t'),


def tokenize(dataset, tokenizer):
    dataset["prompt"] = tokenizer(dataset["prompt"])


def main():
    dataset = load_data()[0]

    pretrained_path = "SamLowe/roberta-base-go_emotions"
    tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
    model = AutoModelForSequenceClassification.from_pretrained(pretrained_path)

    training_args = TrainingArguments(
        output_dir="model/",
        learning_rate=1e-5,
        per_device_train_batch_size=10,
        per_device_eval_batch_size=10,
        num_train_epochs=20,
    )

    print(dataset)
    tokenize(dataset["train"], tokenizer)
    tokenize(dataset["test"], tokenizer)
    tokenize(dataset["val"], tokenizer)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )


if __name__ == '__main__':
    main()
