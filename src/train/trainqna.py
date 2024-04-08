from datasets import load_dataset
from transformers import AutoTokenizer, DefaultDataCollator, AutoModelForQuestionAnswering
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback

tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")


# inspired by Hugging Face tutorial
# https://huggingface.co/docs/transformers/tasks/question_answering
def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs


def train_qna():
    # load and preprocess data
    training_data = load_dataset("squad", split="train")
    validation_data = load_dataset("squad", split="validation")
    tokenized_training_data = training_data.map(preprocess_function, batched=True)
    tokenized_validation_data = validation_data.map(preprocess_function, batched=True)

    # prepare model and train model
    data_collator = DefaultDataCollator()
    model = AutoModelForQuestionAnswering.from_pretrained("FacebookAI/roberta-base")

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=1,  # Stop after 1 evaluation without improvement
        early_stopping_threshold=0.001,  # A minimum improvement of 0.001 is required
    )

    training_args = TrainingArguments(
        output_dir="qa_model",
        evaluation_strategy="steps",
        learning_rate=2e-5,
        per_device_train_batch_size=5,
        per_device_eval_batch_size=5,
        num_train_epochs=10,
        save_steps=3000,
        weight_decay=0.01,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_training_data,
        eval_dataset=tokenized_validation_data,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[early_stopping_callback],
    )

    trainer.train()
    return model


if __name__ == '__main__':
    train_qna()
