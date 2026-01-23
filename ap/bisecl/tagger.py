#!/usr/bin/env python3

import gzip
import json
import logging
import mimetypes
import random
from collections import defaultdict

from ingredient_parser.en import FeatureDict, PreProcessor
from tqdm import tqdm

from .perceptron import AveragedPerceptronBISECL

logger = logging.getLogger(__name__)


class IngredientTaggerBISECL:
    """Class to tag ingredient sentence tokens.

    Attributes
    ----------
    labels : set[str]
        Labels that each ingredient sentence token can be tagged as.
    model : AveragedPerceptron
        Averaged Perceptron model used for tagging.
    only_positive_bool_features : bool, optional
        If True, only use features with boolean values if the value is True.
        If False, always use features with boolean values.
        Default is False.
    """

    def __init__(
        self,
        weights_file: str | None = None,
        only_positive_bool_features: bool = False,
        apply_label_constraints: bool = True,
    ):
        self.model = AveragedPerceptronBISECL()
        self.labeldict = {}
        self.labels: set[str] = set()

        if weights_file is not None:
            self.load(weights_file)

        self.only_positive_bool_features = only_positive_bool_features
        self.apply_label_constraints = apply_label_constraints

    def __repr__(self):
        return f"IngredientTaggerBISECL(labels={self.labels})"

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
        p = PreProcessor(sentence)
        sentence_features = p.sentence_features()
        features = [self._convert_features(f) for f in sentence_features]
        stems = [f["stem"] for f in sentence_features]
        pos = [f["pos"] for f in sentence_features]
        label_scores = self.model.predict_sequence(
            features, stems, pos, constrain_transitions=self.apply_label_constraints
        )

        labels = [
            (token.text, label, score)
            for token, (label, score) in zip(p.tokenized_sentence, label_scores)
        ]
        return labels

    def tag_from_features(
        self, sentence_features: list[FeatureDict]
    ) -> list[tuple[str, float]]:
        """Tag a sentence with labels using Averaged Perceptron model.

        This function accepts a list of features for each token,
        rather than calculating the features from the tokens.

        Parameters
        ----------
        sentence_features : list[FeatureDict]
            List of feature dicts for each token.

        Returns
        -------
        list[tuple[str, float]]
            List of (label, confidence) tuples.
        """
        features = [self._convert_features(f) for f in sentence_features]
        stems = [f["stem"] for f in sentence_features]
        pos = [f["pos"] for f in sentence_features]
        return self.model.predict_sequence(
            features, stems, pos, constrain_transitions=self.apply_label_constraints
        )

    def _convert_features(self, features: FeatureDict) -> set[str]:
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
        features : FeatureDict
            Dictionary of features for token, obtained from
            PreProcessor.sentence_features().

        Returns
        -------
        set
            Set of features as strings
        """
        converted = set()
        for key, value in features.items():
            if isinstance(value, bool) and self.only_positive_bool_features:
                if value:
                    converted.add(key)
            elif isinstance(value, str):
                converted.add(key + "=" + value)
            elif isinstance(value, (int, float, bool)):
                converted.add(key + "=" + str(value))

        return converted

    def save(self, path: str, compress: bool = True) -> str:
        """Save trained model to given path.

        The weights and labels are saved as a tuple.

        Parameters
        ----------
        path : str
            Path to save model weights to.
        compress : bool, optional
            If True, compress .json file using gzip.
            Default is True.

        Returns
        -------
        str
            File path to saved model.
        """
        data = {
            "labels": list(self.model.labels),
            "weights": self.model.weights,
            "labeldict": self.labeldict,
        }

        if not path.endswith(".json"):
            path = path + ".json"

        if compress:
            if not path.endswith(".gz"):
                path = path + ".gz"

            with gzip.open(path, "wt", encoding="utf-8") as f:
                json.dump(data, f)
        else:
            with open(path, "w") as f:
                json.dump(data, f, separators=(",", ":"))

        return path

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
        training_features: list[list[FeatureDict]],
        truth: list[list[str]],
        n_iter: int = 10,
        min_abs_weight: float = 0.1,
        min_feat_updates: int = 0,
        quantize_bits: int | None = None,
        make_label_dict: bool = False,
        show_progress: bool = True,
    ) -> None:
        """Train model using example sentences and their true labels.

        Parameters
        ----------
        training_features : list[list[FeatureDict]]
            List of sentence_features() lists for each sentence.
        truth : list[list[str]]
            List of true label lists for each sentence.
        n_iter : int, optional
            Number of training iterations.
            Default is 10.
        min_abs_weight : float, optional
            Weights below this value will be pruned after training.
        min_feat_updates : int, optional
            Minimum number of feature updates required to consider feature.
        make_label_dict : bool, optional
            If True, create a dict of labels for tokens that are unambiguous in the
            training data. Default i False.
        show_progress: bool, optional
            If True, show progress bar for iterations.
            Default is True.
        """
        if len(self.model.labels) == 0:
            raise ValueError("Set the model labels before training.")

        # Set min_feature_updates for model
        self.model.min_feat_updates = min_feat_updates

        # Convert training features to set outside the training loop so we don't do it
        # at every epoch.
        converted_training_features = [
            [self._convert_features(f) for f in sentence_features]
            for sentence_features in training_features
        ]
        # Extract token stems and pos tags - we need them for generated features based
        # on the previous and next labels in a sequence.
        sentence_stems = [
            [f["stem"] for f in sentence_features]
            for sentence_features in training_features
        ]
        sentence_pos = [
            [f["pos"] for f in sentence_features]
            for sentence_features in training_features
        ]

        # We need to convert to list before we start training so that we can shuffle the
        # list after each training epoch.
        training_data = list(
            zip(converted_training_features, sentence_stems, sentence_pos, truth)
        )

        for iter_ in tqdm(range(n_iter), disable=not show_progress):
            n = 0  # numer of total tokens this iteration
            c = 0  # number of correctly labelled tokens this iteration
            for features_seq, stem_seq, pos_seq, truth_labels in training_data:
                current_labels: dict[int, str] = {}
                indices_to_label = set(range(len(features_seq)))

                while indices_to_label:
                    best_score = -float("inf")
                    best_candidate = (-1, "_UNASSIGNED_")

                    for i in indices_to_label:
                        stem = stem_seq[i]
                        pos = pos_seq[i]
                        for label in self.labels:
                            features = features_seq[i] | self.model.label_features(
                                current_labels, i, stem, pos, label, len(features_seq)
                            )
                            score = self.model._score(features, label)
                            if score > best_score:
                                best_score = score
                                best_candidate = (i, label)

                    pred_idx, predict_label = best_candidate
                    true_label = truth_labels[pred_idx]
                    stem = stem_seq[pred_idx]
                    pos = pos_seq[pred_idx]
                    self.model.update(
                        true_label,
                        predict_label,
                        features_seq[pred_idx]
                        | self.model.label_features(
                            current_labels,
                            pred_idx,
                            stem,
                            pos,
                            predict_label,
                            len(features_seq),
                        ),
                        features_seq[pred_idx]
                        | self.model.label_features(
                            current_labels,
                            pred_idx,
                            stem,
                            pos,
                            true_label,
                            len(features_seq),
                        ),
                    )

                    # Assign the true label for the current token index so the model
                    # learns the correct context from labels.
                    current_labels[pred_idx] = true_label
                    indices_to_label.remove(pred_idx)

                    c += predict_label == true_label
                    n += 1

            logger.debug(f"Iter {iter_}: {100 * c / n:.2f}% correct tokens.")

            random.shuffle(training_data)

        self.model.filter_features()
        self.model.average_weights()
        self.model.prune_weights(min_abs_weight)
        if quantize_bits:
            self.model.quantize(quantize_bits)

    def _make_labeldict(
        self,
        sentence_features: list[list[FeatureDict]],
        sentence_labels: list[list[str]],
    ) -> None:
        """Generate dict of unambiguous token stems and their labels.

        This dict only includes token stems that occur more than FREQ_THRESHOLD times
        in the training data, and have the given label at least AMBIGUITY_THRESHOLD of
        the time.

        Token stems are used instead of tokens themselves because the stem feature is
        always present. The token feature is only present if it differs from the stem.

        Parameters
        ----------
        sentence_features : list[list[str]]
            List of FeatureDicts for each sentence.
        sentence_labels : list[list[str]]
            Label for each token in each sentence.
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
