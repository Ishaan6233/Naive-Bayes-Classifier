from main import NaiveBayesClassifier, preprocess_data, cross_val_score, defaultdict, accuracy_score, pd


class ImprovedNaiveBayesClassifier(NaiveBayesClassifier):
    def __init__(self, smoothing=1.0, remove_stopwords=True, use_bigrams=True, rare_word_threshold=2):
        super().__init__(smoothing, remove_stopwords)
        self.use_bigrams = use_bigrams
        self.rare_word_threshold = rare_word_threshold
        self.word_frequency = defaultdict(int)

    def _generate_ngrams(self, words, n=1):
        """Generate n-grams (default unigrams) from a list of words."""
        if n == 1:
            return words
        return [' '.join(words[i:i + n]) for i in range(len(words) - n + 1)]

    def _filter_rare_words(self):
        """Filter rare words based on the threshold."""
        self.rare_words = {word for word, count in self.word_frequency.items() if count < self.rare_word_threshold}

    def train(self, X, y):
        # Preprocess the sentences and count word frequencies
        processed_X = []
        for doc in X:
            words = self._filter_stopwords(doc.split())
            unigrams = self._generate_ngrams(words, n=1)
            bigrams = self._generate_ngrams(words, n=2) if self.use_bigrams else []
            features = unigrams + bigrams
            processed_X.append(' '.join(features))

            # Update word frequency counts
            for word in features:
                self.word_frequency[word] += 1

        # Filter rare words from the vocabulary
        self._filter_rare_words()

        # Remove rare words from training data
        filtered_X = []
        for doc in processed_X:
            words = doc.split()
            filtered_words = [word for word in words if word not in self.rare_words]
            filtered_X.append(' '.join(filtered_words))

        # Call the original train method with filtered data
        super().train(filtered_X, y)

    def predict(self, X):
        # Transform test sentences into unigram and bigram features
        processed_X = []
        for doc in X:
            words = self._filter_stopwords(doc.split())
            unigrams = self._generate_ngrams(words, n=1)
            bigrams = self._generate_ngrams(words, n=2) if self.use_bigrams else []
            features = unigrams + bigrams
            filtered_features = [word for word in features if word not in self.rare_words]
            processed_X.append(' '.join(filtered_features))

        # Call the original predict method with filtered data
        return super().predict(processed_X)


def main(args):
    # Preprocess data and initialize improved classifier
    train_data = preprocess_data(args.train)
    test_data = preprocess_data(args.test)

    train_texts = train_data["sentence"]
    train_labels = train_data["relation"]
    test_texts = test_data["sentence"]
    test_labels = test_data["relation"]

    # Initialize improved Naive Bayes classifier
    improved_nb = ImprovedNaiveBayesClassifier(smoothing=1.0, use_bigrams=True, rare_word_threshold=3)

    # Perform 3-fold cross-validation
    print("\nPerforming 3-fold cross-validation on the training set...")
    avg_train_accuracy, avg_train_f1 = cross_val_score(improved_nb, train_texts, train_labels, k=3)
    print(f"Average Training Accuracy (Cross-Validation): {avg_train_accuracy * 100:.2f}%")
    print(f"Average Training Macro F1-Score (Cross-Validation): {avg_train_f1 * 100:.2f}%")

    # Train on the full training set and evaluate
    print("\nTraining on the full training set...")
    improved_nb.train(train_texts, train_labels)

    print("\nEvaluating on the test set...\n")
    test_predictions = improved_nb.predict(test_texts)
    test_accuracy = accuracy_score(test_labels, test_predictions)

    # Save test set results
    output_data = pd.DataFrame({
        "original_label": test_labels,
        "output_label": test_predictions,
        "row_id": test_data.get("row_id", range(len(test_data)))
    })
    output_data.to_csv(args.output, index=False)

    print(f"\nTest Set Accuracy: {test_accuracy * 100:.2f}%")


# Correct the entry point condition to use "__name__ == '__main__'"
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Improved Naive Bayes for Relation Extraction")
    parser.add_argument("--train", type=str, required=True, help="Path to training CSV file")
    parser.add_argument("--test", type=str, required=True, help="Path to test CSV file")
    parser.add_argument("--output", type=str, required=True, help="Path to save output CSV file")
    args = parser.parse_args()
    main(args)
