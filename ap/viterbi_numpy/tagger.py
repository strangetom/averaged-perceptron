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

import numpy as np
from ingredient_parser.en import FeatureDict, PreProcessor
from tqdm import tqdm

from ap._dataclasses import ModelHyperParameters

from .perceptron import (
    AveragedPerceptronViterbiNumpy,
    label_features,
)

logger = logging.getLogger(__name__)


class IngredientTaggerViterbiNumpy:
    """Class to tag ingredient sentence tokens.

    Attributes
    ----------
    labels : set[str]
        Labels that each ingredient sentence token can be tagged as.
    model : AveragedPerceptronViterbiNumpy
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
            self.model = AveragedPerceptronViterbiNumpy(
                labels=labels, training_mode=True
            )

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
        if (
            self.model.emission_weights.size == 0
            or self.model.transition_weights.size == 0
        ):
            raise ValueError(
                "AveragedPerceptronViterbiNumpy model does not have any weights."
            )

        p = PreProcessor(sentence, custom_units={})
        features = [self._convert_features(f) for f in p.sentence_features()]
        label_scores = self.model.predict_sequence(
            features, constrain_transitions=self.apply_label_constraints
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
        if (
            self.model.emission_weights.size == 0
            or self.model.transition_weights.size == 0
        ):
            raise ValueError(
                "AveragedPerceptronViterbiNumpy model does not have any weights."
            )

        features = [self._convert_features(f) for f in sentence_features]
        return self.model.predict_sequence(
            features, constrain_transitions=self.apply_label_constraints
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
            Dictionary of token features token, obtained from
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
          1. weights.npy   - numpy array of weights.
          2. features.json - list of features, ordered per the weights matrix rows
          3. labels.json   - list of labels, ordered per the weights matrix columns

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

        emission_weights = self.model.emission_weights
        transition_weights = self.model.transition_weights
        features = list(self.model.feature_vocab)
        labels = self.model.labels

        # The weights matrix may be larger than the number of features due to how it's
        # resized. Discard any rows after the total number of features.
        resized_emission_weights = emission_weights[: len(features), :]

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

            # Add weights matrices
            npy_buffer = io.BytesIO()
            np.save(npy_buffer, resized_emission_weights)
            npy_buffer.seek(0)
            npy_info = tarfile.TarInfo(name="emission_weights.npy")
            npy_info.size = npy_buffer.getbuffer().nbytes
            npy_info.mtime = time.time()
            tar.addfile(npy_info, npy_buffer)

            npy_buffer = io.BytesIO()
            np.save(npy_buffer, transition_weights)
            npy_buffer.seek(0)
            npy_info = tarfile.TarInfo(name="transition_weights.npy")
            npy_info.size = npy_buffer.getbuffer().nbytes
            npy_info.mtime = time.time()
            tar.addfile(npy_info, npy_buffer)

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
            # Extract and read the hyper parameters file
            hyperparameters_file = tar.extractfile("hyperparameters.json")
            if hyperparameters_file:
                hyperparameters = ModelHyperParameters(
                    **json.load(hyperparameters_file)
                )
            else:
                raise FileNotFoundError(
                    f"Could not find hyperparameters.json in {path}."
                )

            # Abort if saved model is not compatible with this class.
            if hyperparameters.model_type not in ["ap_numpy"]:
                raise ValueError(
                    (
                        f"Loaded model is '{hyperparameters.model_type}' which "
                        "is not compatible with 'ap_numpy'."
                    )
                )

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
            emission_weights_file = tar.extractfile("emission_weights.npy")
            if emission_weights_file:
                emission_weights_buffer = io.BytesIO(emission_weights_file.read())
            else:
                raise FileNotFoundError(
                    f"Could not find emission_weights.npy in {path}."
                )
            emission_weights = np.load(emission_weights_buffer)

            # Extract and read the NPY file
            transition_weights_file = tar.extractfile("transition_weights.npy")
            if transition_weights_file:
                transition_weights_buffer = io.BytesIO(transition_weights_file.read())
            else:
                raise FileNotFoundError(
                    f"Could not find transition_weights.npy in {path}."
                )
            transition_weights = np.load(transition_weights_buffer)

        self.model = AveragedPerceptronViterbiNumpy(labels=labels)
        self.model.emission_weights = emission_weights
        self.model.transition_weights = transition_weights
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

        if make_label_dict:
            logger.warning("IngredientTaggerViterbi does not use label_dict.")

        # Set min_feature_updates for model
        self.model.min_feat_updates = min_feat_updates

        # Convert training features to set outside the training loop so we don't do it
        # at every epoch.
        converted_training_features = [
            [self._convert_features(f) for f in sentence_features]
            for sentence_features in training_features
        ]
        # Extract part of speech tags - we need them for generated features based on the
        # previous label in a sequence
        sentence_pos_tags = [
            [f["pos"] for f in sentence_features]
            for sentence_features in training_features
        ]

        # We need to convert to list before we start training so that we can shuffle the
        # list after each training epoch.
        training_data = list(zip(converted_training_features, sentence_pos_tags, truth))

        for iter_ in tqdm(range(n_iter), disable=not show_progress):
            n = 0  # number of total tokens this iteration
            c = 0  # number of correctly labelled tokens this iteration
            for features, pos_tags, truth_labels in training_data:
                predicted_sequence = self.model.predict_sequence(
                    features,
                    return_score=False,
                    constrain_transitions=False,
                )
                predicted_labels, _ = zip(*predicted_sequence)

                if list(predicted_labels) == truth_labels:
                    # All correct, no need to modify weights
                    n += len(predicted_labels)
                    c += len(predicted_labels)
                    self.model._iteration += len(features)
                    continue

                # Calculate features based on prev_label for each element in sequence.
                # Do this for both the predicted and true sequences.
                prev_label_features_predicted = [
                    label_features(prev_label, pos)
                    for prev_label, pos in zip(
                        ["-START-", *list(predicted_labels)], pos_tags
                    )
                ]

                # We could pre-compute this?
                prev_label_features_true = [
                    label_features(prev_label, pos)
                    for prev_label, pos in zip(["-START-", *truth_labels], pos_tags)
                ]

                # Iterate through each element in sequence and update weights.
                # Note that model.update checks if true_label == predict_label and
                # doesn't make any changes to weights.
                for (
                    predict_label,
                    true_label,
                    base_feats,
                    predicted_transition_feats,
                    truth_transition_feats,
                ) in zip(
                    predicted_labels,
                    truth_labels,
                    features,
                    prev_label_features_predicted,
                    prev_label_features_true,
                ):
                    self.model.update(
                        true_label,
                        predict_label,
                        base_feats,
                        predicted_transition_feats,
                        truth_transition_feats,
                    )

                    c += predict_label == true_label
                    n += 1

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
