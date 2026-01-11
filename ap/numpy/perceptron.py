#!/usr/bin/env python3

import logging

import numpy as np

logger = logging.getLogger(__name__)


class AveragedPerceptronNumpy:
    def __init__(self, labels: list[str], n_features: int = 2**20) -> None:
        """
        Parameters
        ----------
        labels : list[str]
            List of possible labels.
        n_features : int, optional
            Size of the feature hash space.
            Larger reduces collisions but increases memory usage.
        """
        self.n_features = n_features

        # reverse=True is used here to ensure consistency with greedy AP.
        # In self.predcit, if there's a tie for the highest score the greedy AP resolves
        # using max(labels) i.e. reverse alphabetically.
        # In this version, we peform the same operation using np.argmax which resolves
        # ties using the index of the first occurance, therefore we need the label
        # indices to be in reverse alphabetical order.
        self.labels = sorted(list(labels), reverse=True) if labels else []
        self.label_to_idx = {label: i for i, label in enumerate(self.labels)}
        self.n_labels = len(self.labels)

        # Matrix of weights: [n_features, n_labels]
        self.weights = np.zeros((self.n_features, self.n_labels), dtype=np.float32)

        # Accumulated weight for each feature/label combination
        self._totals = np.zeros((self.n_features, self.n_labels), dtype=np.float64)
        # The last iteration each feature/label combination was changed.
        # This is to avoid having to update every key in self._totals every iteration,
        # we only update each feature/label key when it changes, accounting for the
        # iterations since it last changed.
        self._tstamps = np.zeros((self.n_features, self.n_labels), dtype=np.int32)

        # Count the number of times each feature has been updated (independent of the
        # label) during training.
        self._feat_updates = np.zeros((self.n_features, 1), dtype=np.int32)
        # The minimum number of feature updates required to consider the feature in
        # the prediction step.
        self.min_feat_updates = 0

        self._iteration: int = 0

    def __repr__(self):
        return f"AveragedPerceptronNumpy(labels={self.labels})"

    def _confidence(self, scores: np.ndarray) -> np.ndarray:
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
        scores : np.ndarray
            Numpy array of scores for each label.

        Returns
        -------
        np.ndarray
            Numpy array of confidence
        """
        max_score = np.max(scores)
        exp_scores = np.exp(scores - max_score)
        probability = exp_scores / exp_scores.sum()
        return np.round(probability, 3)

    def predict(
        self,
        feature_indices: list[int],
        constrained_labels: set[str],
        return_score: bool = False,
    ) -> tuple[str, float]:
        """Predict the label for a token described by features set.

        Parameters
        ----------
        feature_indices : list[int]
            List of indices correponding to features.
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
        # Don't consider until they have been updated more than min_feat_updates times.
        # feature_indices = [
        #     idx for idx in feature_indices
        #     if self._feat_updates[idx] >= self.min_feat_updates
        # ]

        # Sum weights for active features across all labels at once
        # Shape: (n_labels,) i.e. the summed score for each label.
        scores = self.weights[feature_indices].sum(axis=0)

        # Apply label constraints by setting scores for constrained labels to the less
        # than the minimum of the scores. We have do this instead of just setting the
        # value to -inf or similar because the scores may be integers if we're using a
        # quantized model.
        if constrained_labels:
            mask_indices = [
                self.label_to_idx[label]
                for label in constrained_labels
                if label in self.label_to_idx
            ]
            scores[mask_indices] = np.min(scores) - 1

        # Find the best label
        best_idx = np.argmax(scores)
        best_label = self.labels[best_idx]

        best_confidence = 1.0
        if return_score:
            best_confidence = self._confidence(scores)[best_idx]

        return best_label, best_confidence

    def _update_totals(self, label_index: int, feature_indices: list[int]):
        """Update weights for feature by given change

        Parameters
        ----------
        label_index : int
            Index of label to update total for.
        feature_indices : list[int]
            List of indices corresponding to features to update weights for.
        """
        # Calculate how many iterations passed since this weight was last touched
        iters = self._iteration - self._tstamps[feature_indices, label_index]

        # Add the accumulated weight to totals
        # totals += iterations_passed * current_weight
        self._totals[feature_indices, label_index] += (
            iters * self.weights[feature_indices, label_index]
        )

        # Update timestamp to current iteration
        self._tstamps[feature_indices, label_index] = self._iteration

        # Increment feature update count
        self._feat_updates[feature_indices] += 1

    def update(self, truth: str, guess: str, feature_indices: list[int]) -> None:
        """Update weights for given features.

        This only makes changes if the true and predicted labels are different.

        Parameters
        ----------
        truth : str
            True label for given features.
        guess : str
            Predicted label for given features.
        feature_indices : list[int]
            List of indices correponding to features.

        Returns
        -------
        None
        """

        self._iteration += 1

        if truth == guess:
            return None

        # Update the weights and accumulated totals for the features and labels are
        # changing.
        # For true label
        truth_idx = self.label_to_idx[truth]
        self._update_totals(truth_idx, feature_indices)
        self.weights[feature_indices, truth_idx] += 1.0

        # For guess label
        guess_idx = self.label_to_idx[guess]
        self._update_totals(guess_idx, feature_indices)
        self.weights[feature_indices, guess_idx] -= 1.0

        return None

    def average_weights(self) -> None:
        """Average the value for each weight over all updates.

        Returns
        -------
        None
        """
        # Final pass to bring all totals up to date
        feature_indices, label_indices = self.weights.nonzero()
        iters = self._iteration - self._tstamps[feature_indices, label_indices]
        self._totals[feature_indices, label_indices] += (
            iters * self.weights[feature_indices, label_indices]
        )

        # Average weights
        self.weights[feature_indices, label_indices] = (
            self._totals[feature_indices, label_indices] / self._iteration
        )

        return None

    def filter_features(self) -> None:
        """Filter features from weights matrix if they were updated less than than
        min_feat_updates.

        If min_feat_updates is 0, do nothing.

        Returns
        -------
        None
        """
        if self.min_feat_updates == 0:
            # Nothing to filter
            return None

        initial_weight_count = np.count_nonzero(self.weights)
        # Set row to 0 where feature has not been updated at least min_feature_updates
        # times.
        idx_to_zero = np.argwhere(
            self._feat_updates[self._feat_updates < self.min_feat_updates]
        )
        self.weights[idx_to_zero, :] = 0
        filtered_count = np.count_nonzero(self.weights)

        filtered_pc = 100 - 100 * filtered_count / initial_weight_count
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

        pruned_pc = 100 - 100 * pruned_count / initial_weight_count
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

        max_weight = np.max(self.weights)
        scale = (2 ** (nbits - 1) - 1) / max_weight
        self.weights = np.round(self.weights * scale).astype(np.int32)
        logger.debug(f"Quantized model weights using {nbits} of precision.")
