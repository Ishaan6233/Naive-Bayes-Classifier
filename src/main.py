import argparse
from collections import defaultdict

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold


class NaiveBayesClassifier:
    def __init__(self, smoothing=1.0, remove_stopwords=True):
        self.smoothing = smoothing
        self.remove_stopwords = remove_stopwords
        self.vocabulary = set()  # Store all unique words from training data
        self.prior = {}  # Prior probabilities for each class
        self.cond_prob = {}  # Conditional probabilities for each word given a class
        self.classes = []  # List of unique classes

        # Download stopwords if removal is enabled
        if self.remove_stopwords:
            nltk.download("stopwords", quiet=True)
            self.stopwords = set(stopwords.words("english"))
        else:
            self.stopwords = set()

    def _filter_stopwords(self, words):
        """Remove stopwords from the list of words if enabled."""
        if self.remove_stopwords:
            return [word for word in words if word.lower() not in self.stopwords]
        return words

    def train(self, X, y):
        """
        Train the Naive Bayes classifier on training data.
        Args:
            X: List of training documents (strings).
            y: List of corresponding class labels.
        """
        self.classes = np.unique(y)  # Extract unique class labels
        n_docs = len(y)

        # Calculate prior probabilities for each class using Laplace smoothing
        self.prior = {
            cls: (np.sum(y == cls) + self.smoothing) / (n_docs + self.smoothing * len(self.classes))
            for cls in self.classes
        }

        word_counts = {cls: defaultdict(int) for cls in self.classes}

        # Count word frequencies for each class
        for doc, label in zip(X, y):
            words = doc.split()  # Tokenize document into words
            filtered_words = self._filter_stopwords(words)  # Remove stopwords if enabled
            for word in filtered_words:
                word_counts[label][word] += 1  # Increment word count for the label
                self.vocabulary.add(word)  # Add word to vocabulary

        vocab_size = len(self.vocabulary)  # Total number of unique words in the vocabulary
        self.cond_prob = {cls: {} for cls in self.classes}

        # Calculate conditional probabilities for each word given a class
        for cls in self.classes:
            total_words = sum(word_counts[cls].values())  # Total word count for the class
            for word in self.vocabulary:
                # Apply Laplace smoothing to the conditional probability calculation
                self.cond_prob[cls][word] = (
                                                    word_counts[cls][word] + self.smoothing
                                            ) / (total_words + self.smoothing * vocab_size)

    def predict(self, X):
        """
        Predict the class for each document.
        Args:
            X: List of documents to classify.
        Returns:
            List of predicted class labels.
        """
        predictions = []
        for doc in X:
            # Initialize log-probabilities with log-priors for each class
            scores = {cls: np.log(self.prior.get(cls, 1e-10)) for cls in self.classes}
            words = doc.split()  # Tokenize document into words
            filtered_words = self._filter_stopwords(words)  # Remove stopwords if enabled
            for word in filtered_words:
                if word in self.vocabulary:
                    for cls in self.classes:
                        # Add log-conditional probabilities for each word
                        scores[cls] += np.log(self.cond_prob[cls].get(word, 1e-10))
            predictions.append(max(scores, key=scores.get))  # Predict class with highest score
        return predictions


def load_data(file_path):
    """Load data from a CSV file."""
    try:
        data = pd.read_csv(file_path)
    except Exception as e:
        raise ValueError(f"Error loading file {file_path}: {e}")

    # Extract relevant columns from the dataset
    texts = data["tokens"]  # Tokenized sentences
    labels = data["relation"]  # Relationship labels
    head_positions = data["head_pos"]  # Positions of head entities
    tail_positions = data["tail_pos"]  # Positions of tail entities
    row_ids = data.get("row_id", range(len(data)))  # Optional row IDs

    return texts, labels, head_positions, tail_positions, row_ids


def preprocess_data(file_path):
    """
    Preprocess the data by tagging entity positions in sentences.
    Returns Processed data as a DataFrame with tagged sentences and relations.
    """
    df = pd.read_csv(file_path)
    processed_data = []

    for _, row in df.iterrows():
        tokens = row['tokens'].split()  # Tokenize the sentence
        relation = row['relation']  # Extract relation label
        head_pos = list(map(int, row['head_pos'].split()))  # Parse head entity positions
        tail_pos = list(map(int, row['tail_pos'].split()))  # Parse tail entity positions

        # Tag head entity in the sentence
        tokens[head_pos[0]] = '<HEAD>' + tokens[head_pos[0]]
        tokens[head_pos[-1]] = tokens[head_pos[-1]] + '</HEAD>'

        # Tag tail entity if it exists
        if tail_pos != [0]:
            tokens[tail_pos[0]] = '<TAIL>' + tokens[tail_pos[0]]
            tokens[tail_pos[-1]] = tokens[tail_pos[-1]] + '</TAIL>'

        tagged_sentence = ' '.join(tokens)  # Combine tokens back into a tagged sentence
        processed_data.append({'sentence': tagged_sentence, 'relation': relation})

    return pd.DataFrame(processed_data)


def evaluate(y_true, y_pred, classes):
    """Evaluate the classifier."""
    confusion_matrix = pd.DataFrame(0, index=classes, columns=classes)
    for true, pred in zip(y_true, y_pred):
        confusion_matrix.at[true, pred] += 1

    precision = {}
    recall = {}
    f1_scores = []
    tp_sum = 0  # Total true positives for micro-average
    fp_sum = 0  # Total false positives for micro-average
    fn_sum = 0  # Total false negatives for micro-average

    for cls in classes:
        tp = confusion_matrix.at[cls, cls]
        fp = confusion_matrix[cls].sum() - tp
        fn = confusion_matrix.loc[cls].sum() - tp
        precision[cls] = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall[cls] = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision[cls] * recall[cls]) / (precision[cls] + recall[cls]) if (precision[cls] + recall[
            cls]) > 0 else 0
        f1_scores.append(f1)

        # Accumulate totals for micro-averaging
        tp_sum += tp
        fp_sum += fp
        fn_sum += fn

    # Micro-averaged precision and recall
    micro_precision = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else 0
    micro_recall = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else 0

    # Macro-averaged precision and recall
    macro_precision = np.mean(list(precision.values()))
    macro_recall = np.mean(list(recall.values()))

    # Macro-averaged F1-score
    macro_f1 = np.mean(f1_scores)

    return confusion_matrix, precision, recall, micro_precision, micro_recall, macro_precision, macro_recall, macro_f1


def cross_val_score(model, X, y, k=3):
    """
    Perform k-fold cross-validation.
    Args:
        model: The model to train and evaluate.
        X: List of documents (texts).
        y: List of labels.
        k: Number of folds (default is 3).
    Returns:
        avg_accuracy: Average accuracy across folds.
        avg_metrics: Dictionary of averaged precision, recall, and F1-score.
    """
    # Convert X and y to lists to avoid ndarray slicing issues
    X = list(X)
    y = list(y)

    kf = KFold(n_splits=k, shuffle=True, random_state=42)  # Create k-fold split
    fold_accuracies = []
    fold_metrics = []

    for train_index, val_index in kf.split(X):
        # Split data into training and validation
        X_train = [X[i] for i in train_index]
        X_val = [X[i] for i in val_index]
        y_train = [y[i] for i in train_index]
        y_val = [y[i] for i in val_index]

        # Train the model
        model.train(X_train, y_train)

        # Predict on validation data
        predictions = model.predict(X_val)

        # Calculate accuracy and other metrics
        accuracy = accuracy_score(y_val, predictions)
        fold_accuracies.append(accuracy)

        # Calculate precision, recall, and F1-score
        confusion_matrix, precision, recall, _, _, _, _, macro_f1 = evaluate(y_val, predictions, model.classes)
        fold_metrics.append(macro_f1)  # Use macro F1-score as a summary metric

    # Average the accuracy and metrics across all folds
    avg_accuracy = np.mean(fold_accuracies)
    avg_metrics = np.mean(fold_metrics)

    return avg_accuracy, avg_metrics


def main(args):
    # Preprocess the training and testing data
    train_data = preprocess_data(args.train)
    test_data = preprocess_data(args.test)

    # Extract text and labels from the processed data
    train_texts = train_data["sentence"]
    train_labels = train_data["relation"]
    test_texts = test_data["sentence"]
    test_labels = test_data["relation"]

    # Initialize Naive Bayes classifier with smoothing factor 1.0
    nb = NaiveBayesClassifier(smoothing=1.0)

    # Perform 3-fold cross-validation on the training data
    print("\nPerforming 3-fold cross-validation on the training set...")
    avg_train_accuracy, avg_train_f1 = cross_val_score(nb, train_texts, train_labels, k=3)
    print(f"Average Training Accuracy (Cross-Validation): {avg_train_accuracy * 100:.2f}%")
    print(f"Average Training Macro F1-Score (Cross-Validation): {avg_train_f1 * 100:.2f}%")

    # Train the Naive Bayes classifier on the full training set
    print("\nTraining on the full training set...")
    nb.train(train_texts, train_labels)

    # Evaluate the classifier on the training set
    print("\nEvaluating on the training set...\n")
    train_predictions = nb.predict(train_texts)
    train_accuracy = accuracy_score(train_labels, train_predictions)

    # Save the training set evaluation metrics
    train_confusion_matrix, train_precision, train_recall, train_micro_prec, train_micro_recall, train_macro_prec, train_macro_recall, train_macro_f1 = evaluate(
        train_labels, train_predictions, nb.classes
    )

    # Display training set performance
    print("Training Set Confusion Matrix:")
    print(train_confusion_matrix)

    print("\nTraining Set Precision per class:")
    for cls, value in train_precision.items():
        print(f"  {cls}: {value * 100:.2f}%")

    print("\nTraining Set Recall per class:")
    for cls, value in train_recall.items():
        print(f"  {cls}: {value * 100:.2f}%")

    print(f"\nTraining Set Accuracy: {train_accuracy * 100:.2f}%")
    print(f"Training Set Micro-averaged Precision: {train_micro_prec * 100:.2f}%")
    print(f"Training Set Macro-averaged Precision: {train_macro_prec * 100:.2f}%")
    print(f"Training Set Macro-averaged F1-Score: {train_macro_f1 * 100:.2f}%")

    # Evaluate the classifier on the test set
    print("\nEvaluating on the test set...\n")
    test_predictions = nb.predict(test_texts)
    test_accuracy = accuracy_score(test_labels, test_predictions)

    # Save the predictions to a CSV file
    output_data = pd.DataFrame({
        "original_label": test_labels,
        "output_label": test_predictions,
        "row_id": test_data.get("row_id", range(len(test_data)))  # Use row_id or generate default range
    })
    output_data.to_csv(args.output, index=False)

    # Evaluate the model's performance on the test set
    test_confusion_matrix, test_precision, test_recall, test_micro_prec, test_micro_recall, test_macro_prec, test_macro_recall, test_macro_f1 = evaluate(
        test_labels, test_predictions, nb.classes
    )

    # Display test set performance
    print("Test Set Confusion Matrix:")
    print(test_confusion_matrix)

    print("\nTest Set Precision per class:")
    for cls, value in test_precision.items():
        print(f"  {cls}: {value * 100:.2f}%")

    print("\nTest Set Recall per class:")
    for cls, value in test_recall.items():
        print(f"  {cls}: {value * 100:.2f}%")

    print(f"\nTest Set Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Test Set Micro-averaged Precision: {test_micro_prec * 100:.2f}%")
    print(f"Test Set Macro-averaged Precision: {test_macro_prec * 100:.2f}%")
    print(f"Test Set Macro-averaged F1-Score: {test_macro_f1 * 100:.2f}%")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Naive Bayes for Relation Extraction")
    parser.add_argument("--train", type=str, required=True, help="Path to training CSV file")
    parser.add_argument("--test", type=str, required=True, help="Path to test CSV file")
    parser.add_argument("--output", type=str, required=True, help="Path to save output CSV file")
    args = parser.parse_args()
    main(args)
