#!/usr/bin/env python3

import gzip
import json
import random
import mimetypes
from collections import defaultdict

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

    def __init__(self, weights_file: str | None = None):
        self.model = AveragedPerceptron()
        self.labeldict = {}
        self.labels: set[str] = set()

        if weights_file is not None:
            self.load(weights_file)

    def __repr__(self):
        return f"IngredientTagger(labels={self.labels})"

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
        prev_label, prev_label2, prev_label3 = "-START-", "-START2-", "-START3-"
        for token, features in zip(p.tokenized_sentence, p.sentence_features()):
            label, confidence = (self.labeldict.get(features["stem"]), 1.0)
            if not label:
                converted_features = self._convert_features(
                    features, prev_label, prev_label2, prev_label3
                )
                label, confidence = self.model.predict(
                    converted_features, return_score=True
                )

            labels.append((token.text, label, confidence))

            prev_label3 = prev_label2
            prev_label2 = prev_label
            prev_label = label

        return labels

    def tag_from_features(
        self, sentence_features: list[dict[str, str | bool]]
    ) -> list[tuple[str, float]]:
        """Tag a sentence with labels using Averaged Perceptron model.

        This function accepts a list of features for each token,
        rather than calculating the features from the tokens.

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
        prev_label, prev_label2, prev_label3 = "-START-", "-START2-", "-START3-"
        for features in sentence_features:
            label, confidence = (self.labeldict.get(features["stem"]), 1.0)
            if not label:
                converted_features = self._convert_features(
                    features, prev_label, prev_label2, prev_label3
                )
                label, confidence = self.model.predict(
                    converted_features, return_score=True
                )

            labels.append((label, confidence))

            prev_label3 = prev_label2
            prev_label2 = prev_label
            prev_label = label

        return labels

    def _convert_features(
        self,
        features: dict[str, str | bool],
        prev_label: str,
        prev_label2: str,
        prev_label3: str,
    ) -> set[str]:
        """Convert features dict to set of strings.

        The model weights use the features as keys, so they need to be a string rather
        than a key: value pair.
        For string features, the string is prepared by joining the key and value by "=".
        For int and float features, the string is prepared by joining the key and value
        by "=".
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
        prev_label2 : str
            Label of token before token before previous token.

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
            elif isinstance(value, (int, float)):
                converted.add(key + "=" + str(value))

        # Add extra features based on labels of previous tokens.
        converted.add("prev_label=" + prev_label)
        converted.add("prev_label2=" + prev_label2)
        converted.add("prev_label3=" + prev_label2)
        converted.add("prev_label2+prev_label=" + "+".join((prev_label2, prev_label)))
        converted.add("prev_label3+prev_label2=" + "+".join((prev_label3, prev_label2)))
        converted.add(
            "prev_label3+prev_label2+prev_label="
            + "+".join((prev_label3, prev_label2, prev_label))
        )
        converted.add("prev_label+pos=" + prev_label + "+" + features["pos"])  # type: ignore

        return converted

    def save(self, path: str, compress: bool = True) -> None:
        """Save trained model to given path.

        The weights and labels are saved as a tuple.

        Parameters
        ----------
        path : str
            Path to save model weights to.
        compress : bool, optional
            If True, compress .json file using gzip.
            Default is True.
        """
        data = {
            "labels": list(self.model.labels),
            "weights": self.model.weights,
            "labeldict": self.labeldict,
        }
        if compress:
            if not path.endswith(".gz"):
                path = path + ".gz"

            with gzip.open(path, "wt", encoding="utf-8") as f:
                json.dump(data, f)
        else:
            with open(path, "w") as f:
                json.dump(data, f, separators=(",", ":"))

    def load(self, path: str) -> None:
        """Load saved model at given path.

        Parameters
        ----------
        path : str
            Path to model to load.
            .json and .json.gz are accepted formats.
        """
        mimetype, encoding = mimetypes.guess_type(path)
        if mimetype != "application/json":
            raise ValueError("Model must be a .json or .json.gz file.")

        if encoding == "gzip":
            with open(path, "rb") as f:
                data = json.loads(gzip.decompress(f.read()))
        else:
            with open(path, "r") as f:
                data = json.load(f)

        self.model.weights = data["weights"]
        self.labels = set(data["labels"])
        self.labeldict = data["labeldict"]
        self.model.labels = self.labels

    def train(
        self,
        training_features: list[list[dict]],
        truth: list[list[str]],
        n_iter: int = 10,
        min_abs_weight: float = 0.1,
        quantize: bool = False,
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

        self._make_labeldict(training_features, truth)

        # We need to convert to list before we start training so that we can shuffle the list
        # after each training epoch.
        training_data = list(zip(training_features, truth))

        for iter_ in range(n_iter):
            n = 0  # numer of total tokens this iteration
            c = 0  # number of correctly labelled tokens this iteration
            for sentence_features, truth_labels in training_data:
                prev_label, prev_label2, prev_label3 = "-START-", "-START2-", "-START3-"
                for features, true_label in zip(sentence_features, truth_labels):
                    guess = self.labeldict.get(features["stem"])
                    if not guess:
                        converted_features = self._convert_features(
                            features, prev_label, prev_label2, prev_label3
                        )
                        # Do not calculate score for prediction during training because
                        # the weights have not been averaged, so the values could be
                        # massive and cause OverflowErrors.
                        guess, _ = self.model.predict(
                            converted_features, return_score=False
                        )
                        self.model.update(true_label, guess, converted_features)

                    prev_label3 = prev_label2
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
        if quantize:
            self.model.quantize()

    def _make_labeldict(
        self, sentence_features: list[list[dict]], sentence_labels: list[list[str]]
    ) -> None:
        """Generate dict of unambiguous token stems and their labels.

        This dict only includes token stems that occur more than FREQ_THRESHOLD times
        in the training data, and have the given label at least AMBIGUITY_THRESHOLD of
        the time.

        Token stems are used instead of tokens themselves because the stem feature is always
        present. The token feature is only present if it differs from the stem.

        Parameters
        ----------
        sentence_token_stems : list[list[str]]
            Stems for each token in each sentence
        sentence_labels : list[list[str]]
            Label for each token in each sentence
        """
        counts = defaultdict(lambda: defaultdict(int))
        for features, labels in zip(sentence_features, sentence_labels):
            for feats, label in zip(features, labels):
                stem = feats["stem"]
                counts[stem][label] += 1

        FREQ_THRESHOLD = 10
        AMBIGUITY_THRESHOLD = 1

        self.labeldict = {}
        for stem, tag_freqs in counts.items():
            label, mode = max(tag_freqs.items(), key=lambda item: item[1])
            n = sum(tag_freqs.values())
            if n >= FREQ_THRESHOLD and (float(mode) / n) >= AMBIGUITY_THRESHOLD:
                self.labeldict[stem] = label
