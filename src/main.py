import torch.cuda
from datasets import load_dataset
from transformers import (Trainer, TrainingArguments, DataCollatorWithPadding, AutoTokenizer, AutoModelForSequenceClassification)
from torch.nn import CrossEntropyLoss


def load_data():
    data_files = {
        "train": "data/train-emotion-clean.tsv",
        "test": "data/test-emotion-clean.tsv",
        "validation": "data/val-emotion-clean.tsv",
    }
    return load_dataset("csv", data_files=data_files, delimiter='\t'),


def tokenize(dataset, tokenizer):
    return tokenizer(dataset["text"])


def main():
    dataset = load_data()[0]

    pretrained_path = "google-bert/bert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(pretrained_path)

    training_args = TrainingArguments(
        output_dir="model/",
        learning_rate=1e-5,
        per_device_train_batch_size=9,
        per_device_eval_batch_size=9,
        num_train_epochs=20
    )

    dataset = dataset.map(tokenize, batched=True, fn_kwargs={"tokenizer": tokenizer})

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(pretrained_path,
                                                               num_labels=9)

    print("Hello")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.train()


if __name__ == '__main__':
    main()
