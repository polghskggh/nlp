from datasets import load_dataset, DatasetDict

from transformers import (Trainer, TrainingArguments, DataCollatorWithPadding,
                          AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.dummy import DummyClassifier

import numpy as np
import csv





# make final evaluation of the model
def test_model(trainer, dataset):
    logits, labels, _ = trainer.predict(dataset)
    predictions = np.argmax(logits, axis=-1)
    metrics = compute_metrics(predictions, labels)
    conf_matrix = confusion_matrix(labels, predictions)
    return metrics, conf_matrix





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




    # final evaluation
    scores, confusion_mat = test_model(trainer, dataset['test'])
    store_data(confusion_mat, 'confusion.csv')


if __name__ == '__main__':
    main()
