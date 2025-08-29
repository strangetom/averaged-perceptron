#!/usr/bin/env python3

import concurrent.futures as cf
import gzip
import json
import logging
import mimetypes
import random
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from itertools import islice, product
from typing import Iterable

from ingredient_parser.en import FeatureDict, PreProcessor
from tqdm import tqdm

from ap._constants import ILLEGAL_TRANSITIONS
from ap.greedy.perceptron import (
    AveragedPerceptron,
)

logger = logging.getLogger(__name__)


@dataclass
class EpochResults:
    weights: dict[str, dict[str, float]]
    feature_updates: dict[str, int]  # key = feature
    accum_weights: dict[str, float]  # key = (feature, label)
    iterations: int


def _convert_features(
    features: FeatureDict,
    prev_label: str,
    prev_label2: str,
    prev_label3: str,
    only_positive_bool_features: bool = False,
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
        Dictionary of token features token, obtained from
        PreProcessor.sentence_features().
    prev_label : str
        Label of previous token.
    prev_label2 : str
        Label of token before token before previous token.
    prev_label3 : str
        Description
    only_positive_bool_features : bool, optional
        If True, only include boolean features that have positive values.
        Default is False, where the positive and negative version of boolean features
        are generated.

    Returns
    -------
    set
        Set of features as strings
    """
    converted = set()
    for key, value in features.items():
        if isinstance(value, bool) and only_positive_bool_features:
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


def train_ap_single_epoch(
    training_data: list[tuple[list[FeatureDict], list[str]]],
    labels: set[str],
    initial_weights: dict[str, dict[str, float]],
) -> EpochResults:
    """Train a single iteration of the AveragedPerceptron model.

    Parameters
    ----------
    training_data : list[tuple[list[FeatureDict], list[str]]]
        List of (FeatureDicts, true label) tuples for each training sentence.
    initial_weights : dict[str, dict[str, float]]
        Initial model weights.
    """
    model = AveragedPerceptron()
    model.labels = labels
    model.weights = initial_weights

    random.shuffle(training_data)
    for sentence_features, truth_labels in training_data:
        prev_label, prev_label2, prev_label3 = "-START-", "-START2-", "-START3-"
        for features, true_label in zip(sentence_features, truth_labels):
            converted_features = _convert_features(
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
            guess, _ = model.predict(converted_features, set(), return_score=False)
            model.update(true_label, guess, converted_features)

            prev_label3 = prev_label2
            prev_label2 = prev_label
            # Use the guess here to avoid to model becoming over-reliant on the
            # historical labels being correct
            prev_label = guess

    # Update total to account for iterations since last update
    for feature, weights in model.weights.items():
        for label, weight in weights.items():
            key = (feature, label)
            model._totals[key] += (model._iteration - model._tstamps[key]) * weight

    return EpochResults(
        weights=model.weights,
        feature_updates=model._feature_updates,
        accum_weights=model._totals,
        iterations=model._iteration,
    )


def chunked(iterable: Iterable, n: int) -> Iterable:
    """Break *iterable* into lists of length *n*.

    By the default, the last yielded list will have fewer than *n* elements
    if the length of *iterable* is not divisible by *n*:

    Parameters
    ----------
    iterable : Iterable
        Iterable to chunk.
    n : int
        Size of each chunk.

    Returns
    -------
    Iterable
        Chunks of iterable with size n (or less for the last chunk).

    Examples
    --------
    >>> list(chunked([1, 2, 3, 4, 5, 6], 3))
    [[1, 2, 3], [4, 5, 6]]

    >>> list(chunked([1, 2, 3, 4, 5, 6, 7, 8], 3))
    [[1, 2, 3], [4, 5, 6], [7, 8]]
    """

    def take(n: int, iterable: Iterable) -> list:
        """Return first n items of the iterable as a list.

        Parameters
        ----------
        n : int
            Number of items to return in list.
        iterable : Iterable
            Iterable to return items from.

        Returns
        -------
        list
            List of first n items of iterable.
        """
        return list(islice(iterable, n))

    return iter(partial(take, n, iter(iterable)), [])


def merge_weights(
    weight_dicts: list[dict[str, dict[str, float]]], n: int
) -> dict[str, dict[str, float]]:
    """Merge weights from each model.

    The merging is done by the weighted sum of each weight, where the sum weight is 1/n.
    (n is the number of shards.)

    Each element of the weight_dicts input and the output is a dict of dicts, in the
    format: d[feature][label] = weight.
    Each dict only contains non-zero weights.

    Parameters
    ----------
    weight_dicts : list[dict[str, dict[str, float]]]
        List of weight dicts.
    n : int
        Sum weight

    Returns
    -------
    dict[str, dict[str, float]]
        Dict of merged weights.
    """
    merged_weights: dict[str, dict[str, float]] = {}
    for epoch_weights in weight_dicts:
        for feature, feat_weights in epoch_weights.items():
            merged_weights.setdefault(feature, {})
            for label, weight in feat_weights.items():
                merged_weights[feature].setdefault(label, 0.0)
                merged_weights[feature][label] += weight / n

    return merged_weights


def merge_dicts(dicts: list[dict[str, int | float]]) -> dict[str, int | float]:
    """Merge dicts by summing values for features across all dicts.

    Parameters
    ----------
    dicts : list[dict[str, int | float]]
        List of dicts to merge.

    Returns
    -------
    dict[str, int | float]
        Merged dict.
    """
    merged_dict = {}
    for d in dicts:
        for feature, value in d.items():
            merged_dict.setdefault(feature, 0)
            merged_dict[feature] += value

    return merged_dict


class IngredientTagger:
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
        weights_file: str | None = None,
        only_positive_bool_features: bool = False,
        apply_label_constraints: bool = True,
    ):
        self.model = AveragedPerceptron()
        self.labeldict = {}
        self.labels: set[str] = set()

        if weights_file is not None:
            self.load(weights_file)

        self.only_positive_bool_features = only_positive_bool_features
        self.apply_label_constraints = apply_label_constraints

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
        labels, labels_only = [], []
        p = PreProcessor(sentence)
        prev_label, prev_label2, prev_label3 = "-START-", "-START2-", "-START3-"
        for token, features in zip(p.tokenized_sentence, p.sentence_features()):
            label, confidence = (self.labeldict.get(features["stem"]), 1.0)
            if not label:
                constrained_labels = self._apply_constraints(labels_only)

                converted_features = _convert_features(
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
        labels, labels_only = [], []
        prev_label, prev_label2, prev_label3 = "-START-", "-START2-", "-START3-"
        for features in sentence_features:
            label, confidence = (self.labeldict.get(features["stem"]), 1.0)  # type:ignore
            if not label:
                constrained_labels = self._apply_constraints(labels_only)

                converted_features = _convert_features(
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
                # The seperator argument removes spaces from the normal defaults
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
        if len(self.labels) == 0:
            raise ValueError("Set the tagger labels before training.")

        if make_label_dict:
            self._make_labeldict(training_features, truth)

        # Set min_feature_updates for model
        self.model.min_feat_updates = min_feat_updates

        # We need to convert to list before we start training so that we can shuffle the
        # list after each training epoch.
        training_data = list(zip(training_features, truth))

        n_shards = 8
        shard_size = int((len(training_data) + n_shards) / n_shards)
        sharded_training_data = list(chunked(training_data, shard_size))

        initial_weights = {}
        feature_updates = {}
        iterations = 0
        accum_weights = {}
        with cf.ProcessPoolExecutor(max_workers=n_shards) as executor:
            for _ in tqdm(range(n_iter), disable=not show_progress):
                futures = [
                    executor.submit(
                        train_ap_single_epoch, shard, self.labels, initial_weights
                    )
                    for shard in sharded_training_data
                ]
                epoch_results = [f.result() for f in cf.as_completed(futures)]

                for r in epoch_results:
                    iterations += r.iterations
                initial_weights = merge_weights(
                    [r.weights for r in epoch_results], n_shards
                )
                feature_updates = merge_dicts(
                    [r.feature_updates for r in epoch_results]
                )
                accum_weights = merge_dicts([r.accum_weights for r in epoch_results])

        self.model.weights = initial_weights
        self.model.labels = self.labels
        self.model._feature_updates = feature_updates
        self.model._iteration = iterations
        self.model._tstamps = {
            (feature, label): iterations
            for feature, label in product(self.model.weights.keys(), self.labels)
        }
        self.model._totals = accum_weights

        self.model.filter_features()
        self.model.average_weights()
        self.model.prune_weights(min_abs_weight)
        if quantize_bits:
            self.model.quantize(quantize_bits)

    def _make_labeldict(
        self, sentence_features: list[list[dict]], sentence_labels: list[list[str]]
    ) -> None:
        """Generate dict of unambiguous token stems and their labels.

        This dict only includes token stems that occur more than FREQ_THRESHOLD times
        in the training data, and have the given label at least AMBIGUITY_THRESHOLD of
        the time.

        Token stems are used instead of tokens themselves because the stem feature is
        always present. The token feature is only present if it differs from the stem.

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

        self.labeldict: dict[str, str] = {}
        for stem, tag_freqs in counts.items():
            label, mode = max(tag_freqs.items(), key=lambda item: item[1])
            n = sum(tag_freqs.values())
            if n >= FREQ_THRESHOLD and (float(mode) / n) >= AMBIGUITY_THRESHOLD:
                self.labeldict[stem] = label
