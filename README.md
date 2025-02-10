# Naive Bayes Classifier

## Project Description
This project implements a **Naive Bayes Classifier** to perform relation extraction on a dataset containing four types of relationships: publisher, director, performer, and characters.

The objective is to classify relationships between pairs of entities within text data.
Key features of the project include:

- **Data preprocessing** with entity tagging for head and tail positions.
- **Training and evaluation** using metrics such as accuracy, precision, recall, and F1-score.
- **3-fold cross-validation** to ensure robustness and mitigate overfitting.
- Final evaluation on a held-out test dataset to measure **generalization performance**.

## Requirements
- `Ensure that Python 3.7 or later installed. Can be checked your running: python --version`
### Install the following libraries to run the project. These libraries are used for data processing, model training, and evaluation:
  - numpy
  - scikit-learn
  - pandas
These dependencies can be installed by running: `pip install numpy scikit-learn pandas`

## 3-rd Party Libraries
You do not need to list `nltk` and `pandas` here.

* `main.py L:[line]` used `[Library Name]` for [reason].
* main.py L:[18]: Used `numpy` to calculate priors during training.
* main.py L:[45]: Used `numpy` to calculate conditional probabilities and metrics like macro-averaged precision.

## Execution Instructions

## Execution
After installing the required libraries, run the program using the following command:
`python3 src/main.py --train data/train.csv --test data/test.csv --output output/test.csv`
--train specifies the path to the training data (train.csv).
--test specifies the path to the test data (test.csv).
--output specifies the path where the output (predictions) will be saved (test.csv).

## Directory Structure
```
./
│
├── src/
│   └── main.py          # Main script for training and evaluation.
│
├── data/
│   ├── train.csv        # Training data file.
│   ├── test.csv         # Test data file.
│
├── output/
│   └── test.csv         # Generated predictions from test data.
│
└── README.md            # Project documentation.
```
## Implementation Details
### File: src/main.py
Contains the complete implementation for:

1.  Data Loading and Preprocessing
    - Reads data from .csv files.
    - Tokenizes and tags **entity positions** (<HEAD> and <TAIL>).
2.  Naive Bayes Classifier
    - Implements the core Naive Bayes logic, including **Laplace smoothing**.
    - Handles **stopword removal** (via nltk).
3.  Training and Evaluation
    - Includes training, prediction, and metric calculation (confusion matrix,     precision, recall, and F1-score).
    - Performs 3-fold cross-validation on training data.
4. Output Generation
    - Saves predictions for test data to the specified output file.

## Data

### *src/main.py*
This is the main script of the project. It contains the logic for:
- Loading and preprocessing the dataset.
- Implementing the Naive Bayes classifier.
- Training the model.
- Evaluating the model using cross-validation and various metrics.
- Generating output predictions on the test data.

### *data/*
Contains the following data files:

- train.csv: Training dataset with tokenized sentences, labels, and entity positions.
- test.csv: Test dataset formatted similarly to train.csv for evaluation.

Data can be found in [data/train.txt](data/train.txt),and the in-domain test data can be found in [data/test.txt](data/test.txt).

## Metrics and Evaluation
The classifier is evaluated on:

- **Accuracy**: Proportion of correct predictions.
- **Precision**: Measure of positive predictive value per class.
- **Recall**: Measure of sensitivity per class.
- **F1-Score**: Harmonic mean of precision and recall.
- **Cross-validation** ensures consistent performance across subsets of the training data.
