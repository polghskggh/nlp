import torch
from datasets import load_dataset
from transformers import AutoModelForQuestionAnswering, AutoModelForSequenceClassification
from transformers import AutoTokenizer
import json

from train.trainseq2seq import train_seq2seq

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print(device)
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(device)
else:
    device = torch.device("cpu")
    print(device)

batch_size = 30


def predict_classifier(input_batch):
    model = AutoModelForSequenceClassification.from_pretrained("programs/nlp/checkpoint-13000-classifier")
    tokenizer = AutoTokenizer.from_pretrained("programs/nlp/checkpoint-13000-classifier")

    inputs = [context + question for context, question in zip(input_batch["context"], input_batch["question"])]
    inputs = tokenizer(inputs, truncation=True, padding=True, return_tensors="pt")
    outputs = None

    with torch.no_grad():
        for current in range(0, len(inputs["input_ids"]), batch_size):
            if outputs is None:
                outputs = model(inputs["input_ids"][current: current + batch_size],
                                inputs["attention_mask"][current: current + batch_size]).logits.argmax(-1)
            else:
                outputs = torch.cat((outputs,
                                     model(inputs["input_ids"][current: current + batch_size],
                                           inputs["attention_mask"][current: current + batch_size]).logits.argmax(-1)), dim=0)

    labels = outputs
    return labels


def predict_question_answer(input_batch): 
    model = AutoModelForQuestionAnswering.from_pretrained("programs/nlp/checkpoint-7500-seq2seq")
    tokenizer = AutoTokenizer.from_pretrained("programs/nlp/checkpoint-7500-seq2seq")

    inputs = tokenizer(input_batch["question"], input_batch["context"],
                       truncation=True, padding=True, return_tensors="pt")
    outputs = None
    answer_start_index = None
    answer_end_index = None
    with torch.no_grad():
        for current in range(0, len(inputs["input_ids"]), batch_size):
            outputs = model(inputs["input_ids"][current: current + batch_size],
                            inputs["attention_mask"][current: current + batch_size])
            if answer_start_index is None: 
                answer_start_index = outputs.start_logits.argmax(-1)
                answer_end_index = outputs.end_logits.argmax(-1)
            else:
                answer_start_index = torch.cat((answer_start_index, outputs.end_logits.argmax(-1)), dim=0)
                answer_end_index = torch.cat((answer_end_index, outputs.end_logits.argmax(-1)), dim=0)

    answer_start_index = outputs.start_logits.argmax(-1)
    answer_end_index = outputs.end_logits.argmax(-1)

    answers = [tokenizer.decode(inputs.input_ids[0, start: end + 1])
               for start, end in zip(answer_start_index, answer_end_index)]
    return answers


def predict(input_batch):
    answers = predict_question_answer(input_batch)
    print("answers ready")
    labels = predict_classifier(input_batch)

    ret = {key: answer if label.item() else "" for key, label, answer in zip(input_batch["id"], labels, answers)}
    return json.dumps(ret)


def main():
    validation_data = load_dataset("squad_v2", split="validation[:500]")
    print("dataset_loaded")
    with open("output", 'w', encoding='utf-8') as f:
        print(predict(validation_data), file=f)


if __name__ == '__main__':
    main()
