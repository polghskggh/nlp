# Question answering system

## Installation

Required packages are listed in `requirements.txt`
You can run `setup.py` script to install the packages.

Additionally, you may need to install accelerate:
```bash
pip install transformers[torch]
```

## How to run

Here we guide you on how to use our code.

### Training the models

In the `train` directory, there two scripts: `trainclassifier.py` and `trainqna.py`.

Each script trains the corresponding model and saves it (with potentially multiple checkpoints).

You should run each of the scripts in terminal by calling:
```bash
python3 trainclassifier.py
python3 trainqna.py
```

### Generating predictions

`main.py` contains the code for generating predictions and storing them in a json file. 

**Here you need to make sure that the correct path is specified when loading the trained models.** You can choose which model (or checkpoint) to load in the functions *predict_classifier* and *predict_question_answer* functions

### Evaluating the system

To evaluate the predictions made by the system, you can `evaluation.py`. This is the evaluation script provided by SQuAD 2.0. To run use:
```bash
python3 evaluation.py <path_to_dev-v2.0> <path_to_predictions>
```
We use the development/validation set, as the test set is not publicly available. The dataset is available in the `data` folder.

### Notes

Most of our work utilizes the Hugging Face library. The library contains not only transformers, but also datasets. Therefore, we access the SQuAD datasets from the Hugging Face API most of time, with one exclusion when running the evaluation script.
