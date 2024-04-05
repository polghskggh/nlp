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
elif torch.roc.is_available():
    device = torch.device("roc")
    print(device)
else:
    device = torch.device("cpu")
    print(device)

batch_size = 30


def batch_pass(model, inputs):
    outputs = []
    for current in range(0, len(inputs["input_ids"]), batch_size):
        outputs.append(model(inputs["input_ids"][current: current + batch_size],
                             inputs["attention_mask"][current: current + batch_size]))
        print(f"batch {current} passed")
    return outputs


def predict_classifier(input_batch):
    model = AutoModelForSequenceClassification.from_pretrained("programs/nlp/checkpoint-13000-classifier")
    tokenizer = AutoTokenizer.from_pretrained("programs/nlp/checkpoint-13000-classifier")

    inputs = [context + question for context, question in zip(input_batch["context"], input_batch["question"])]
    inputs = tokenizer(inputs, truncation=True, padding=True, return_tensors="pt")

    with torch.no_grad():
        outputs = batch_pass(model, inputs)

    labels = torch.cat([output.logits for output in outputs]).argmax(dim=-1)

    return labels


def predict_question_answer(input_batch): 
    model = AutoModelForQuestionAnswering.from_pretrained("qa_model/checkpoint-5000")
    tokenizer = AutoTokenizer.from_pretrained("qa_model/checkpoint-5000")

    inputs = tokenizer(input_batch["question"], input_batch["context"],
                       truncation=True, padding=True, return_tensors="pt")

    with torch.no_grad():
        outputs = batch_pass(model, inputs)

    answer_start_index = torch.cat([output.start_logits for output in outputs]).argmax(dim=-1)
    answer_end_index = torch.cat([output.end_logits for output in outputs]).argmax(dim=-1)

    answers = [tokenizer.decode(inputs.input_ids[0, start: end + 1])
               for start, end in zip(answer_start_index, answer_end_index)]
    return answers


def predict(input_batch):
    labels = predict_classifier(input_batch)
    print("classifier done")
    answers = predict_question_answer(input_batch)
    print("answering done")
    print(labels.shape)
    ret = {key: answer if label.item() else "" for key, label, answer in zip(input_batch["id"], labels, answers)}
    return json.dumps(ret)


def main():
    validation_data = load_dataset("squad_v2", split="validation[:100]")
    print("dataset_loaded")
    results = predict(validation_data)

    with open("output", 'w', encoding='utf-8') as f:
        print(results, file=f)


if __name__ == '__main__':
    main()
