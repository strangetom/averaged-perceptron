#!/usr/bin/env python3

import io
import json
import logging
import mimetypes
import random
import tarfile
import time
from collections import defaultdict

import numpy as np
from ingredient_parser.en import FeatureDict, PreProcessor
from tqdm import tqdm

from ap._constants import ILLEGAL_TRANSITIONS

from .perceptron import (
    AveragedPerceptronNumpy,
)

logger = logging.getLogger(__name__)


class IngredientTaggerNumpy:
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
    apply_label_constraints : bool, optional
        If True, constrain the predictions of the current label based on the predicted
        sequence so far. This only applies during inference and not during training.
        If False, no constraints are applied.
        Default is True.
    """

    def __init__(
        self,
        *,
        labels: list[str] | None = None,
        weights_file: str | None = None,
        only_positive_bool_features: bool = False,
        apply_label_constraints: bool = True,
    ):
        """Initialise IngredientTaggerNumpy

        Parameters
        ----------
        labels : list[str] | None, optional
            List of labels, only required for training.
        weights_file : str | None, optional
            Path to weights file, only required for inference.
        only_positive_bool_features : bool, optional
            If True, only use features with boolean values if the value is True.
            If False, always use features with boolean values.
            Default is False.
        apply_label_constraints : bool, optional
            If True, constrain the predictions of the current label based on the
            predicted sequence so far. This only applies during inference and not during
            training.
            If False, no constraints are applied.
            Default is True.

        Raises
        ------
        ValueError
            Description
        """
        if labels is None and weights_file is None:
            raise ValueError(
                (
                    "Either a list of labels must be provided (for training) "
                    "or a weights file must be provided (for inference)."
                )
            )
        elif labels is not None and weights_file is not None:
            raise ValueError(
                (
                    "Only one of a list of labels must be provided (for training) "
                    "or a weights file must be provided (for inference)."
                )
            )

        self.labeldict = {}
        if labels is not None:
            self.labels = labels
            self.model = AveragedPerceptronNumpy(labels=labels, training_mode=True)

        if weights_file is not None:
            self.load(weights_file)

        self.only_positive_bool_features = only_positive_bool_features
        self.apply_label_constraints = apply_label_constraints

    def __repr__(self):
        return f"IngredientTaggerNumpy(labels={self.labels})"

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
        if self.model.weights.size == 0:
            raise ValueError("AveragedPerceptronNumpy model does not have any weights.")

        labels, labels_only = [], []
        p = PreProcessor(sentence)
        prev_label, prev_label2, prev_label3 = "-START-", "-START2-", "-START3-"
        for token, features in zip(p.tokenized_sentence, p.sentence_features()):
            label, confidence = (self.labeldict.get(features["stem"]), 1.0)
            if not label:
                constrained_labels = self._apply_constraints(labels_only)

                converted_features = self._convert_features(
                    features, prev_label, prev_label2, prev_label3
                )
                label, confidence = self.model.predict(
                    converted_features, constrained_labels, return_score=True
                )

            labels.append((token.text, label, confidence))
            labels_only.append(label)

            prev_label3 = prev_label2
            prev_label2 = prev_label
            prev_label = label

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
        if self.model.weights.size == 0:
            raise ValueError("AveragedPerceptronNumpy model does not have any weights.")

        labels, labels_only = [], []
        prev_label, prev_label2, prev_label3 = "-START-", "-START2-", "-START3-"
        for features in sentence_features:
            label, confidence = (self.labeldict.get(features["stem"]), 1.0)  # type:ignore
            if not label:
                constrained_labels = self._apply_constraints(labels_only)

                converted_features = self._convert_features(
                    features, prev_label, prev_label2, prev_label3
                )
                label, confidence = self.model.predict(
                    converted_features, constrained_labels, return_score=True
                )

            labels.append((label, confidence))
            labels_only.append(label)

            prev_label3 = prev_label2
            prev_label2 = prev_label
            prev_label = label

        return labels

    def _convert_features(
        self,
        features: FeatureDict,
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
        features : FeatureDict
            Dictionary of token features token, obtained from
            PreProcessor.sentence_features().
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
            if isinstance(value, bool) and self.only_positive_bool_features:
                if value:
                    converted.add(key)
            elif isinstance(value, str):
                converted.add(key + "=" + value)
            elif isinstance(value, (int, float, bool)):
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

    def _apply_constraints(self, sequence: list[str]) -> set[str]:
        """Apply constraints on labels by removing options from the set of possible
        labels for the next token in the sequence.

        Parameters
        ----------
        sequence : list[str]
            Sequence of labels.

        Returns
        -------
        set[str]
            Set of invalid labels for next item in sequence.
        """
        constrained_labels = set()
        if not sequence:
            return constrained_labels

        if not self.apply_label_constraints:
            return constrained_labels

        # B_NAME_TOK must occur before I_NAME_TOK.
        # If the sequence contains NAME_SEP, only check labels after last NAME_SEP.
        # If the sequence does not contain NAME_SEP, check all labels.
        # If B_NAME_TOK not found, remove I_NAME_TOK from set.
        if "NAME_SEP" in sequence:
            # Find index of last occurance of NAME_SEP in sequence
            name_sep_idx = max(i for i, v in enumerate(sequence) if v == "NAME_SEP")
            if "B_NAME_TOK" not in sequence[name_sep_idx:]:
                constrained_labels.add("I_NAME_TOK")
        else:
            if "B_NAME_TOK" not in sequence:
                constrained_labels.add("I_NAME_TOK")

        prev_label = sequence[-1]
        if prev_label in ILLEGAL_TRANSITIONS:
            constrained_labels |= ILLEGAL_TRANSITIONS[prev_label]

        return constrained_labels

    def save(self, path: str) -> None:
        """Save trained model to given path.

        The model comprises 3 files:
          1. weights.npy   - numpy array of weights.
          2. features.json - list of features, ordered per the weights matrix rows
          3. labels.json   - list of labels, ordered per the weights matrix columns

        These are all saved to a .tar.gz file.

        Parameters
        ----------
        path : str
            Path to save model weights to.
        """
        if path.endswith(".tar"):
            path = path + ".gz"
        elif not path.endswith(".tar.gz"):
            path = path + ".tar.gz"

        weights = self.model.weights
        features = list(self.model.feature_vocab)
        labels = self.model.labels

        # The weights matrix may be larger than the number of features due to how it's
        # resized. Discard any rows after the total number of features.
        resized_weights = weights[: len(features), :]

        with tarfile.open(path, "w:gz") as tar:
            # Add features file
            features_data = json.dumps(features, indent=2).encode("utf-8")
            features_buffer = io.BytesIO(features_data)
            features_buffer.seek(0)
            features_info = tarfile.TarInfo(name="features.json")
            features_info.size = features_buffer.getbuffer().nbytes
            features_info.mtime = time.time()
            tar.addfile(features_info, features_buffer)

            # Add labels file
            labels_data = json.dumps(labels, indent=2).encode("utf-8")
            labels_buffer = io.BytesIO(labels_data)
            labels_buffer.seek(0)
            labels_info = tarfile.TarInfo(name="labels.json")
            labels_info.size = labels_buffer.getbuffer().nbytes
            labels_info.mtime = time.time()
            tar.addfile(labels_info, labels_buffer)

            # Add weights matrix
            npy_buffer = io.BytesIO()
            np.save(npy_buffer, resized_weights)
            npy_buffer.seek(0)
            npy_info = tarfile.TarInfo(name="weights.npy")
            npy_info.size = npy_buffer.getbuffer().nbytes
            npy_info.mtime = time.time()
            tar.addfile(npy_info, npy_buffer)

    def load(self, path: str) -> None:
        """Load saved model at given path.

        The expected model is a .tar.gz file containing
        * features.json
        * labels.json
        * weights.npy

        Parameters
        ----------
        path : str
            Path to model to load.
        """
        mimetype, encoding = mimetypes.guess_type(path)
        if not (mimetype == "application/x-tar" and encoding == "gzip"):
            raise ValueError("Model must be a .tar.gz file.")

        with tarfile.open(path, "r:gz") as tar:
            # Extract and read the features file
            features_file = tar.extractfile("features.json")
            if features_file:
                features = json.load(features_file)
            else:
                raise FileNotFoundError(f"Could not find features.json in {path}.")

            # Extract and read the labels file
            labels_file = tar.extractfile("labels.json")
            if labels_file:
                labels = json.load(labels_file)
            else:
                raise FileNotFoundError(f"Could not find labels.json in {path}.")

            # Extract and read the NPY file
            weights_file = tar.extractfile("weights.npy")
            if weights_file:
                weights_buffer = io.BytesIO(weights_file.read())
            else:
                raise FileNotFoundError(f"Could not find weights.npy in {path}.")
            weights = np.load(weights_buffer)

        self.model = AveragedPerceptronNumpy(labels=labels)
        self.model.weights = weights
        self.model.feature_vocab = {feat: idx for idx, feat in enumerate(features)}
        self.labels = labels
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
        quantize_bits : int | None, optional
            Description
        make_label_dict : bool, optional
            If True, create a dict of labels for tokens that are unambiguous in the
            training data. Default i False.
        show_progress : bool, optional
            If True, show progress bar for iterations.
            Default is True.
        """
        if len(self.model.labels) == 0:
            raise ValueError("Set the model labels before training.")

        if make_label_dict:
            self._make_labeldict(training_features, truth)

        # Set min_feature_updates for model
        self.model.min_feat_updates = min_feat_updates

        # We need to convert to list before we start training so that we can shuffle the
        # list after each training epoch.
        training_data = list(zip(training_features, truth))

        for iter_ in tqdm(range(n_iter), disable=not show_progress):
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
                        # Do not set constraints during training. Because the feature
                        # sets include features based on previous labels, applying
                        # constraints now means the model never learns from errors that
                        # might occur during inference and therefore never adjusts
                        # the weights appropriately.
                        guess, _ = self.model.predict(
                            converted_features, set(), return_score=False
                        )
                        self.model.update(true_label, guess, converted_features)

                    prev_label3 = prev_label2
                    prev_label2 = prev_label
                    # Use the guess here to avoid to model becoming over-reliant on the
                    # historical labels being correct
                    prev_label = guess

                    n += 1
                    c += guess == true_label

            logger.debug(f"Iter {iter_}: {100 * c / n:.2f}% correct tokens.")

            random.shuffle(training_data)

        self.model.filter_features()
        self.model.average_weights()
        self.model.prune_weights(min_abs_weight)
        if quantize_bits:
            self.model.quantize(quantize_bits)
        self.model.simplify_weights()
        # Set training_mode False now so that we don't try to resize the model matrices
        # when evaluating the model and there are features not already in the vocab.
        self.model.training_mode = False

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
        sentence_features : list[list[FeatureDict]]
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

        self.labeldict: dict[str, str] = {}
        for stem, tag_freqs in counts.items():
            label, mode = max(tag_freqs.items(), key=lambda item: item[1])
            n = sum(tag_freqs.values())
            if n >= FREQ_THRESHOLD and (float(mode) / n) >= AMBIGUITY_THRESHOLD:
                self.labeldict[stem] = label
