#!/usr/bin/env python3

import logging
from collections import defaultdict
from itertools import product

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


class AveragedPerceptron:
    def __init__(self, features: list[str], labels: list[str]) -> None:
        # 2D array of weights for all features.
        # Rows are features indexed by the features list. Columns are labels, indexed by
        # labels list.
        self.features = features
        self.labels = labels
        self.weights = np.zeros((len(features), len(labels)))

        # Accumulated weight for each feature/label combination
        self._totals = defaultdict(float)
        # The last iteration each feature/label combination was changed.
        # This is to avoid having to update every key in self._totals every iteration,
        # we only update each feature/label key when it changes, accounting for the
        # iterations since it last changed.
        self._tstamps = defaultdict(int)

        # Count the number of times each feature has been updated (independent of the
        # label) during training.
        self._feature_updates = defaultdict(int)
        # The minimum number of feature updates required to consider the feature in
        # the prediction step.
        self.min_feat_updates = 0

        self._iteration: int = 0

    def __repr__(self):
        return f"AveragedPerceptron(labels={self.labels})"

    def _feat_to_idx(self, feat: str) -> int:
        """Convert feature name to index in features list.

        Parameters
        ----------
        feat : str
            Feature to return indices for.

        Returns
        -------
        int
            Index of feature.
        """
        try:
            return self.features.index(feat)
        except ValueError as e:
            logger.debug(f"{feat} not found in features list.")
            raise e

    def _label_to_idx(self, label: str) -> int:
        """Convert label name to index in labels list.

        Parameters
        ----------
        label : str
            Label to return index for.

        Returns
        -------
        int
            Index of label.
        """
        try:
            return self.labels.index(label)
        except ValueError as e:
            logger.debug(f"{label} not found in labels list.")
            raise e

    def _confidence(self, scores: npt.NDArray) -> npt.NDArray:
        """Calculate softmax confidence for each labels.

        To avoid OverflowError exceptions, this is implemented as log softmax.
        The non-log form of softmax, for the ith element is:
            p_i = exp(s_i) / sum_i[exp(s_i)]

        Taking logs
            log(p_i) = log( exp(s_i) / sum_i[exp(s_i)] )
            log(p_i) = log(exp(s_i)) - log(sum_i[exp(s_i)])
            log(p_i) = s_i - log(sum_i[exp(s_i)])

        The second term is dominated by the largest value, so can be approximated as
        max(s), giving

            log(p_i) = s_i - max(s)

        Then we can take the exp to get back the probability.

        See also https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/

        Parameters
        ----------
        scores : npt.NDArray
            Array of scores for labels

        Returns
        -------
        npt.NDArray
            List of confidences for labels
        """
        max_score = np.max(scores)
        return np.exp(scores - max_score)

    def predict(
        self,
        features: set[str],
        constrained_labels: set[str],
        return_score: bool = False,
    ) -> tuple[str, float]:
        """Predict the label for a token described by features set.

        Parameters
        ----------
        features : set[str]
            Set of features for token,
        constrained_labels : set[str]
            Set of labels that may not be predicted for current feature set due to
            constraints from prior predictions.
        return_score : bool, optional
            If True, return score for predicted label.
            If False, returned score is always 1.0.

        Returns
        -------
        tuple[str, float]
            Class label given features and confidence value.
        """
        active_features = [
            feat
            for feat in features
            if self._feature_updates.get(feat, 0) >= self.min_feat_updates
        ]
        feature_idx = [self._feat_to_idx(f) for f in active_features]
        scores = np.sum(self.weights[feature_idx], axis=0)

        # Mask scores array to remove constrainted labels
        constrained_label_idx = [self._label_to_idx(lbl) for lbl in constrained_labels]
        mask = np.zeros(scores.size, dtype=bool)
        mask[constrained_label_idx] = True
        masked_scores = np.ma.array(scores, mask=mask)

        best_score_idx = np.argmax(masked_scores)
        best_label = self.labels[best_score_idx]

        if return_score:
            best_confidence = scores[best_score_idx]
        else:
            best_confidence = 1.0

        return best_label, best_confidence

    def _update_feature(self, label: str, feature: str, weight: float, change: float):
        """Update weights for feature by given change

        Parameters
        ----------
        label : str
            Label to update weight for.
        feature : str
            Feature to update weights of.
        weight : float
            Weight of label for feature at time of last update.
        change : float
            Change to be applied to label weight for feature.
        """
        key = (feature, label)
        # Update total for feature/label combo to account for number of iterations
        # since last update
        self._totals[key] += (self._iteration - self._tstamps[key]) * weight
        self._tstamps[key] = self._iteration

        feature_idx = self._feat_to_idx(feature)
        label_idx = self._label_to_idx(label)
        self.weights[feature_idx, label_idx] = weight + change

        # Increment feature update count
        self._feature_updates[feature] += 1

    def update(self, truth: str, guess: str, features: set[str]) -> None:
        """Update weights for given features.

        This only makes changes if the true and predicted labels are different.

        Parameters
        ----------
        truth : str
            True label for given features.
        guess : str
            Predicted label for given features.
        features : set[str]
            Features.

        Returns
        -------
        None
        """

        self._iteration += 1

        if truth == guess:
            return None

        for feat in features:
            # Get indices for weights array
            feat_idx = self._feat_to_idx(feat)
            truth_idx = self._label_to_idx(truth)
            guess_idx = self._label_to_idx(guess)

            # Update weights for feature:
            # Increment weight for correct label by +1, decrement weights for
            # incorrect labels by -1.
            self._update_feature(truth, feat, self.weights[feat_idx, truth_idx], 1.0)
            self._update_feature(guess, feat, self.weights[feat_idx, guess_idx], -1.0)

        return None

    def average_weights(self) -> None:
        """Average the value for each weight over all updates.

        Returns
        -------
        None
        """
        new_weights = np.zeros(shape=self.weights.size)

        for feat, label in product(self.features, self.labels):
            feat_idx = self._feat_to_idx(feat)
            label_idx = self._label_to_idx(label)

            key = (feat, label)
            total = (
                self._totals[key]
                + (self._iteration - self._tstamps[key])
                * self.weights[feat_idx, label_idx]
            )
            averaged = total / float(self._iteration)
            if averaged:
                new_weights[feat_idx, label_idx] = averaged

        self.weights = new_weights
        return None

    def filter_features(self) -> None:
        """Filter features from weights dict if they were updated less than than
        min_feat_updates.

        If min_feat_updates is 0, do nothing.

        Returns
        -------
        None
        """
        if self.min_feat_updates == 0:
            # Nothing to filter
            return None

        filtered_count, initial_weight_count = 0, len(self.weights)
        for feature in list(self.weights.keys()):
            if self._feature_updates.get(feature, 0) < self.min_feat_updates:
                del self.weights[feature]
                filtered_count += 1

        filtered_pc = 100 * filtered_count / initial_weight_count
        logger.debug(
            (
                f"Removed {filtered_pc:.2f}% of features for updating "
                f"less than {self.min_feat_updates} times."
            )
        )

    def prune_weights(self, min_abs_weight: float) -> None:
        """Prune weights by removing weights smaller than min_abs_weight.

        Parameters
        ----------
        min_abs_weight : float
            Minimum absolute value of weight to keep.
        """
        initial_weight_count = np.count_nonzero(self.weights)
        self.weights[np.abs(self.weights) < min_abs_weight] = 0
        pruned_count = np.count_nonzero(self.weights)

        pruned_pc = 100 * pruned_count / initial_weight_count
        logger.debug(
            (
                f"Pruned {pruned_pc:.2f}% of weights for having absolute "
                f"values small than {min_abs_weight}."
            )
        )

    def quantize(self, nbits: int | None = None) -> None:
        """Quantize weights to nbit signed integer using linear scaling.

        Because the model weights are only used additively during inference, and we only
        consider the relative magnitudes of the weights, there is no need for keep the
        scaling factor because it would just be a multiplier of all of the weights.

        Parameters
        ----------
        nbits : int, optional
            Number of bits for integer scaling.
            If None, no quantisation is performed.
            Default is None.
        """
        if nbits is None:
            return

        max_weight = np.max(np.abs(self.weights))
        scale = (2 ** (nbits - 1) - 1) / max_weight
        self.weights = np.round(self.weights * scale)

        logger.debug(f"Quantized model weights using {nbits} of precision.")
