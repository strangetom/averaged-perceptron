#!/usr/bin/env python3

import io
import json
import logging
import mimetypes
import random
import tarfile
import time
from collections import defaultdict
from dataclasses import asdict
from itertools import product

from ingredient_parser.en import FeatureDict, PreProcessor
from tqdm import tqdm

from ap._dataclasses import ModelHyperParameters

from .perceptron import AveragedPerceptronEasiestFirst

logger = logging.getLogger(__name__)


class IngredientTaggerEasiestFirst:
    """Class to tag ingredient sentence tokens usng easiest first algorithm [1].

    References
    ----------
    [1] L. Shen, G. Satta, and A. Joshi, ‘Guided Learning for Bidirectional Sequence
        Classification’, in Proceedings of the 45th Annual Meeting of the Association of
        Computational Linguistics, A. Zaenen and A. van den Bosch, Eds, Prague, Czech
        Republic: Association for Computational Linguistics, Jun. 2007, pp. 760–767.
        Accessed: Jan. 19, 2026. [Online]. Available: https://aclanthology.org/P07-1096/


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
    ):
        self.model = AveragedPerceptronEasiestFirst()
        self.labeldict = {}
        self.labels: set[str] = set()

        if weights_file is not None:
            self.load(weights_file)

        self.only_positive_bool_features = only_positive_bool_features

    def __repr__(self):
        return f"IngredientTaggerEasiestFirst(labels={self.labels})"

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
        p = PreProcessor(sentence, custom_units={})
        sentence_features = p.sentence_features()
        features = [self._convert_features(f) for f in sentence_features]
        stems = [f["stem"] for f in sentence_features]
        pos = [f["pos"] for f in sentence_features]
        label_scores = self.model.predict_sequence(features, stems, pos)

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
        return self.model.predict_sequence(features, stems, pos)

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

    def save(self, path: str, hyperparameters: ModelHyperParameters) -> str:
        """Save trained model to given path.

        The model comprises 3 files:
          1. weights.json   - JSON object of weights.
          2. labels.json    - list of labels
          3. labeldict.json - labeldict of labels for common tokens.

        These are all saved to a .tar.gz file.

        Parameters
        ----------
        path : str
            Path to save model weights to.
        hyper_parameters : ModelHyperParameters
            Hyper parameters used to train model.

        Returns
        -------
        str
            File path to saved model.
        """
        if path.endswith(".tar"):
            path = path + ".gz"
        elif not path.endswith(".tar.gz"):
            path = path + ".tar.gz"

        labels = list(self.model.labels)
        weights = self.model.weights
        labeldict = self.labeldict

        with tarfile.open(path, "w:gz") as tar:
            # Add labels file
            labels_data = json.dumps(labels, indent=2).encode("utf-8")
            labels_buffer = io.BytesIO(labels_data)
            labels_buffer.seek(0)
            labels_info = tarfile.TarInfo(name="labels.json")
            labels_info.size = labels_buffer.getbuffer().nbytes
            labels_info.mtime = time.time()
            tar.addfile(labels_info, labels_buffer)

            # Add weights file
            weights_data = json.dumps(weights, indent=2).encode("utf-8")
            weights_buffer = io.BytesIO(weights_data)
            weights_buffer.seek(0)
            weights_info = tarfile.TarInfo(name="weights.json")
            weights_info.size = weights_buffer.getbuffer().nbytes
            weights_info.mtime = time.time()
            tar.addfile(weights_info, weights_buffer)

            # Add labeldict file
            labeldict_data = json.dumps(labeldict, indent=2).encode("utf-8")
            labeldict_buffer = io.BytesIO(labeldict_data)
            labeldict_buffer.seek(0)
            labeldict_info = tarfile.TarInfo(name="labeldict.json")
            labeldict_info.size = labeldict_buffer.getbuffer().nbytes
            labeldict_info.mtime = time.time()
            tar.addfile(labeldict_info, labeldict_buffer)

            # Add hyper parameters file
            hp_data = json.dumps(asdict(hyperparameters), indent=2).encode("utf-8")
            hp_buffer = io.BytesIO(hp_data)
            hp_buffer.seek(0)
            hp_info = tarfile.TarInfo(name="hyperparameters.json")
            hp_info.size = hp_buffer.getbuffer().nbytes
            hp_info.mtime = time.time()
            tar.addfile(hp_info, hp_buffer)

        return path

    def load(self, path: str) -> None:
        """Load saved model at given path.

        The expected model is a .tar.gz file containing
        * weights.json
        * labels.json
        * labeldict.json

        Parameters
        ----------
        path : str
            Path to model to load.
        """
        mimetype, encoding = mimetypes.guess_type(path)
        if not (mimetype == "application/x-tar" and encoding == "gzip"):
            raise ValueError("Model must be a .tar.gz file.")

        with tarfile.open(path, "r:gz") as tar:
            # Extract and read the hyper parameters file
            hyperparameters_file = tar.extractfile("hyperparameters.json")
            if hyperparameters_file:
                hyperparameters = json.load(hyperparameters_file)
            else:
                raise FileNotFoundError(
                    f"Could not find hyperparameters.json in {path}."
                )

            # Abort if saved model is not compatible with this class.
            if hyperparameters["model_type"] not in ["ap_easiest_first"]:
                raise ValueError(
                    (
                        f"Loaded model is '{hyperparameters['model_type']}' which "
                        "is not compatible with 'ap_easiest_first'."
                    )
                )

            # Extract and read the weights file
            weights_file = tar.extractfile("weights.json")
            if weights_file:
                weights = json.load(weights_file)
            else:
                raise FileNotFoundError(f"Could not find weights.json in {path}.")

            # Extract and read the labels file
            labels_file = tar.extractfile("labels.json")
            if labels_file:
                labels = json.load(labels_file)
            else:
                raise FileNotFoundError(f"Could not find labels.json in {path}.")

            # Extract and read the labeldict file
            labeldict_file = tar.extractfile("labeldict.json")
            if labeldict_file:
                labeldict = json.load(labeldict_file)
            else:
                raise FileNotFoundError(f"Could not find labeldict.json in {path}.")

        self.model.weights = weights
        self.labels = set(labels)
        self.labeldict = labeldict
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
            Number of bits to quantize weights to.
        make_label_dict : bool, optional
            If True, create a dict of labels for tokens that are unambiguous in the
            training data. Default i False.
        show_progress : bool, optional
            If True, show progress bar for iterations.
            Default is True.
        """
        if len(self.model.labels) == 0:
            raise ValueError("Set the model labels before training.")

        # Set min_feature_updates for model
        self.model.min_feat_updates = min_feat_updates

        # Sort labels to ensure that ties are always resolved in the same way.
        sorted_labels = sorted(self.labels, reverse=True)

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

                    for i, label in product(indices_to_label, sorted_labels):
                        stem = stem_seq[i]
                        pos = pos_seq[i]
                        features = features_seq[i] | self.model.label_features(
                            current_labels, i, stem, pos, label, len(features_seq)
                        )
                        score = self.model.score(features, label)
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
