#!/usr/bin/env python3

import pickle
import random

from ingredient_parser.en import PreProcessor

from .averaged_perceptron import AveragedPerceptron


class IngredientTagger:
    """Class to tag ingredient sentence tokens.

    Attributes
    ----------
    labels : set[str]
        Labels that each ingredient sentence token can be tagged as.
    model : AveragedPerceptron
        Averaged Perceptron model used for tagging.
    """

    def __init__(self):
        self.model = AveragedPerceptron()
        self.labels: set[str] = set()

    def tag(self, sentence: str) -> list[tuple[str, str, float]]:
        """Tag a sentence with labels using Averaged Perceptron model.

        Parameters
        ----------
        sentence : str
            Sentence to tag tokens of.

        Returns
        -------
        list[tuple[str, str, float]]
            List of (token, label, confidence) tuples.
        """
        labels = []
        p = PreProcessor(sentence)
        prev_label, prev_label2 = "-START-", "-START2-"
        for token, features in zip(p.tokenized_sentence, p.sentence_features()):
            converted_features = self._convert_features(
                features, prev_label, prev_label2
            )
            label, confidence = self.model.predict(converted_features)
            labels.append((token, label, confidence))

            prev_label2 = prev_label
            prev_label = label

        return labels

    def tag_from_features(
        self, sentence_features: list[dict[str, str | bool]]
    ) -> list[tuple[str, float]]:
        """Tag a sentence with labels using Averaged Perceptron model.

        This function accepts a list of tokens and a list of features for each token,
        rather than calculate the features in the function.

        Parameters
        ----------
        sentence_features : list[dict[str, str | bool]]
            List of feature dicts for each token.

        Returns
        -------
        list[tuple[str, float]]
            List of (label, confidence) tuples.
        """
        labels = []
        prev_label, prev_label2 = "-START-", "-START2-"
        for features in sentence_features:
            converted_features = self._convert_features(
                features, prev_label, prev_label2
            )
            label, confidence = self.model.predict(converted_features)
            labels.append((label, confidence))

            prev_label2 = prev_label
            prev_label = label

        return labels

    def _convert_features(
        self, features: dict[str, str | bool], prev_label: str, prev_label2: str
    ) -> set[str]:
        """Convert features dict to set of strings.

        The model weights use the features as keys, so they need to be a string rather
        than a key: value pair.
        For string features, the string is prepared by joining the key and value by "=".
        For boolean features, the string is prepared just using the key.

        Additional features are added based on the labels of the previous two tokens.

        Parameters
        ----------
        features : dict[str, str | bool]
            Dictionary of features for token, obtained from PreProcessor.sentence_features().
        prev_label : str
            Label of previous token.
        prev_label2 : str
            Label of token before previous token.

        Returns
        -------
        set
            Set of features as strings
        """
        converted = set()
        for key, value in features.items():
            if isinstance(value, bool) and value:
                converted.add(key)
            elif isinstance(value, str):
                converted.add(key + "=" + value)

        # Add extra features based on labels of previous tokens.
        converted.add("prev_label=" + prev_label)
        converted.add("prev_label2=" + prev_label2)
        converted.add("prev_label2+prev_label=" + prev_label2 + "+" + prev_label)
        converted.add("prev_label+pos=" + prev_label + "+" + features["pos"])  # type: ignore

        return converted

    def save(self, path: str) -> None:
        """Save trained model to given path.

        The weights and labels are saved as a tuple.

        Parameters
        ----------
        path : str
            Path to save model weights to.
        """
        with open(path, "wb") as f:
            pickle.dump((self.model.weights, self.model.labels), f)

    def load(self, path: str) -> None:
        """Load saved model at given path.

        Parameters
        ----------
        path : str
            Path to model to load.
        """
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.model.weights, self.labels = data
            self.model.labels = self.labels

    def train(
        self,
        training_features: list[list[dict]],
        truth: list[list[str]],
        n_iter: int = 10,
        min_abs_weight: float = 0.1,
        verbose: bool = True,
    ) -> None:
        """Train model using example sentences and their true labels.

        Parameters
        ----------
        training_features : list[list[dict]]
            List of sentence_features() lists for each sentence.
        truth : list[list[str]]
            List of true label lists for each sentence.
        n_iter : int, optional
            Number of training iterations.
            Default is 10.
        min_abs_weight : float, optional
            Weights below this value will be pruned after training.
        verbose : bool, optional
            If True, print performance at each iteration.
            Default is is True.
        """
        if len(self.model.labels) == 0:
            raise ValueError("Set the model labels before training.")

        # We need to convert to list before we start training so that we can shuffle the list
        # after each training epoch.
        training_data = list(zip(training_features, truth))

        for iter_ in range(n_iter):
            n = 0  # numer of total tokens this iteration
            c = 0  # number of correctly labelled tokens this iteration
            for sentence_features, truth_labels in training_data:
                prev_label, prev_label2 = "-START-", "-START2-"
                for features, true_label in zip(sentence_features, truth_labels):
                    converted_features = self._convert_features(
                        features, prev_label, prev_label2
                    )
                    guess, _ = self.model.predict(converted_features)
                    self.model.update(true_label, guess, converted_features)

                    prev_label2 = prev_label
                    # Use the guess here to avoid to model becoming over-reliant on the historical labels
                    # being correct
                    prev_label = guess

                    n += 1
                    c += guess == true_label

            if verbose:
                print(f"Iter {iter_}: {100*c/n:.1f}%")

            random.shuffle(training_data)

        self.model.average_weights()
        self.model.prune_weights(min_abs_weight)
