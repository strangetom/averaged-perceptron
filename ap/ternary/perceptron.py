#!/usr/bin/env python3

import logging

import numpy as np

logger = logging.getLogger(__name__)


class AveragedPerceptronTernary:
    def __init__(self, labels: list[str], training_mode: bool = False) -> None:
        """
        Parameters
        ----------
        labels : list[str]
            List of possible labels.
        training_mode : bool, optional
            If True, initialise weights matrix and other matrices needed for training.
            If False (default), do not initialise these matrices. In this case, the
            weights matrix must be set externally.
        """
        # reverse=True is used here to ensure consistency with greedy AP.
        # In self.predict, if there's a tie for the highest score the greedy AP resolves
        # using max(labels) i.e. reverse alphabetically.
        # In this version, we perform the same operation using np.argmax which resolves
        # ties using the index of the first occurrence, therefore we need the label
        # indices to be in reverse alphabetical order.
        self.labels = sorted(labels, reverse=True) if labels else []
        self.label_to_idx = {label: i for i, label in enumerate(self.labels)}
        self.n_labels = len(self.labels)

        self.training_mode = training_mode
        if self.training_mode:
            self.feature_vocab: dict[str, int] = {}
            self.next_feature_index = 0

            initial_features = 10_000
            # Matrix of weights: [n_features, n_labels]
            self.weights = np.zeros((initial_features, self.n_labels), dtype=np.float32)

            # Accumulated weight for each feature/label combination
            self._totals = np.zeros((initial_features, self.n_labels), dtype=np.float64)
            # The last iteration each feature/label combination was changed.
            # This is to avoid having to update every key in self._totals every
            # iteration, we only update each feature/label key when it changes,
            # accounting for the iterations since it last changed.
            self._tstamps = np.zeros((initial_features, self.n_labels), dtype=np.int32)

            # Count the number of times each feature has been updated (independent of
            # the label) during training.
            self._feat_updates = np.zeros((initial_features,), dtype=np.int32)
            # The minimum number of feature updates required to consider the feature in
            # the prediction step.
            self.min_feat_updates: int = 0

            # Threshold used to ternarization. This is only used during training.
            self.ternary_threshold: float = 0.0

            self._iteration: int = 0

        else:
            # Initialise to empty objects for inference.
            # These will need to be set to real values prior to inference.
            self.weights = np.array([])
            self.feature_vocab: dict[str, int] = {}

    def __repr__(self):
        return f"AveragedPerceptronTernary(labels={self.labels})"

    def update_ternary_threshold(self) -> None:
        """Update the cached ternary threshold based on active latent weights.

        This implementation follows the 0.75 * mean heuristic [1], ignoring unused
        pre-allocated rows and zeros to prevent dilution.

        References
        ----------
        [1] B. Liu, F. Li, X. Wang, B. Zhang, and J. Yan, ‘Ternary Weight Networks’, in
            ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and
            Signal Processing (ICASSP), Rhodes Island, Greece: IEEE, Jun. 2023, pp. 1–5.
            doi: 10.1109/ICASSP49357.2023.10094626.
        """
        if self.training_mode:
            # Slice to only include the current vocabulary
            active_weights = self.weights[: self.next_feature_index]
        else:
            # During inference, simplify_weights() has already removed any rows that are
            # all zeroes.
            active_weights = self.weights

        if active_weights.size == 0:
            self.ternary_threshold = 0.0
            return

        # Filter for non-zero weights.
        # Since a Perceptron is only interested in non-zero weights we need to remove
        # the zeroes from this calculation that arise from the choice to store the
        # weights as a dense matrix.
        abs_weights = np.abs(active_weights)
        non_zero_weights = abs_weights[abs_weights > 0]

        if non_zero_weights.size == 0:
            self.ternary_threshold = 0.0
        else:
            # Apply the 0.75 heuristic to the mean of non-zero active weights
            self.ternary_threshold = float(0.75 * np.mean(non_zero_weights))

    def _ternarize_weights(self, weights: np.ndarray) -> np.ndarray:
        """Convert weights to ternary using ternary threshold.

        The given threshold is used symmetrically in the conversion:
        weight > threshold: 1
        weight < -threshold: -1
        otherwise: 0

        Parameters
        ----------
        weights : np.ndarray
            2D NumPy array of weights.

        Returns
        -------
        np.ndarray
            Ternary weights.
        """
        return np.where(
            weights < -self.ternary_threshold,
            -1,
            np.where(weights > self.ternary_threshold, 1, 0),
        )

    def _resize_matrices(self) -> None:
        """Resize matrices by doubling the number of rows.

        This is called by IngredientTagger during training once the vocabulary reaches
        the current number of rows of the matrices.
        """
        current_capacity = self.weights.shape[0]
        new_capacity = current_capacity * 2

        # Helper to resize a single matrix and copy data from old matrix into new one.
        def resize_mat(old_mat: np.ndarray) -> np.ndarray:
            ndim = old_mat.ndim
            dtype = old_mat.dtype
            if ndim == 1:
                new_mat = np.zeros((new_capacity,), dtype=dtype)
                new_mat[:current_capacity] = old_mat
            else:  # assume ndim=2, since we don't use anything with higher dimensions.
                cols = old_mat.shape[1]
                new_mat = np.zeros((new_capacity, cols), dtype=dtype)
                new_mat[:current_capacity, :] = old_mat

            return new_mat

        self.weights = resize_mat(self.weights)
        self._totals = resize_mat(self._totals)
        self._tstamps = resize_mat(self._tstamps)
        self._feat_updates = resize_mat(self._feat_updates)

    def _features_to_idx(self, features: set[str]) -> np.ndarray:
        """Map feature string to indices by lookup in vocab dict.

        Whilst order is not important (hence this function accepting a set), we return
        a list because we can use that directly when indexing NumPy arrays.

        Parameters
        ----------
        features : set[str]
            Set of feature strings to hash.

        Returns
        -------
        np.ndarray
            NumPy array of integer indices for string features.
        """
        if self.training_mode:
            for feat in features:
                if feat not in self.feature_vocab:
                    self.feature_vocab[feat] = self.next_feature_index
                    self.next_feature_index += 1

            # Resize matrices if the vocab now exceeds the current number of rows.
            if self.next_feature_index >= self.weights.shape[0]:
                self._resize_matrices()

        return np.array(
            [
                self.feature_vocab[feat]
                for feat in features
                if feat in self.feature_vocab
            ]
        )

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
            Set of features for token.
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
        feature_indices = self._features_to_idx(features)

        if self.training_mode:
            # Don't consider features until they have been updated more than
            # min_feat_updates times. We only do this during training.
            feature_indices = feature_indices[
                self._feat_updates[feature_indices] >= self.min_feat_updates
            ]

            # Sum weights for active features across all labels at once
            # Shape: (n_labels,) i.e. the summed score for each label.
            if len(feature_indices) > 0:
                # When training we need to convert the active latent weights to ternary
                # to use to calculate the scores.
                active_latent = self.weights[feature_indices]
                ternary_active = self._ternarize_weights(active_latent)
                scores = ternary_active.sum(axis=0)
            else:
                scores = np.zeros(self.n_labels)

        else:
            # Inference only path.
            if len(feature_indices) > 0:
                scores = self.weights[feature_indices].sum(axis=0)
            else:
                scores = np.zeros(self.n_labels, dtype=np.int8)

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
            # Set score for constrained labels to less than lowest calculated score, but
            # constrain to -128 so we don't overflow when using int8 numbers.
            scores[mask_indices] = max([np.min(scores) - 1, np.int8(-128)])

        # Find the best label
        best_idx = np.argmax(scores)
        best_label = self.labels[best_idx]

        # Because the weights are ternary, a score doesn't make much sense.
        best_confidence = 1.0

        return best_label, best_confidence

    def _update_totals(self, label_index: int, feature_indices: np.ndarray):
        """Update weights for feature by given change

        Parameters
        ----------
        label_index : int
            Index of label to update total for.
        feature_indices : np.ndarray
            NumPy array of indices corresponding to features to update weights for.
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

        # Update the weights and accumulated totals for the features and labels are
        # changing.
        feature_indices = self._features_to_idx(features)
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
        iters = self._iteration - self._tstamps
        self._totals += iters * self.weights

        # Average weights
        self.weights = self._totals / self._iteration

        return None

    def filter_features(self) -> None:
        """Filter features from weights matrix if they were updated less than than
        min_feat_updates.

        If min_feat_updates is 0, do nothing.

        Note: We have to also set the rows in the _totals matrix to 0 here so that the
        features remain at zero weight for all labels when averaging. If we don't, then
        the features would retain whatever accumulated weight total they had at the last
        iteration the feature was updated.

        Returns
        -------
        None
        """
        if self.min_feat_updates == 0:
            # Nothing to filter
            return None

        # Initial feature count is number of features that have ever been updated.
        initial_feature_count = np.count_nonzero(self._feat_updates)

        # Set row indices where feature has not been updated at least
        # min_feature_updates times.
        # We're only considering rows with at least one feature update here to make the
        # calculation of the logging message easier. We could just do
        # np.argwhere(self._feat_updates < self.min_feat_updates), but that would also
        # include features that have never been updated.
        idx_to_zero = np.argwhere(
            np.logical_and(
                0 < self._feat_updates, self._feat_updates < self.min_feat_updates
            )
        )

        # Set weights for these rows to 0.
        # We also need to set the rows in the _totals matrix to zero for these too so
        # they remain 0 after averaging.
        self.weights[idx_to_zero, :] = 0
        self._totals[idx_to_zero, :] = 0

        filtered_count = idx_to_zero.shape[0]

        filtered_pc = 100 * filtered_count / initial_feature_count
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
        if min_abs_weight == 0:
            # Nothing to prune
            return None

        initial_weight_count = np.count_nonzero(self.weights)
        self.weights[np.abs(self.weights) < min_abs_weight] = 0
        remaining_count = np.count_nonzero(self.weights)
        pruned_pc = 100 * (1 - remaining_count / initial_weight_count)

        logger.debug(
            (
                f"Pruned {pruned_pc:.2f}% of weights for having absolute "
                f"values small than {min_abs_weight}."
            )
        )

    def simplify_weights(self) -> None:
        """Simplify weights matrix by discarding any rows that are all zeros.

        We also simplify the feature vocab to remove the entries corresponding to those
        rows too.
        """
        # Find row indices where absolute sum of weights is non zero. We keep these and
        # discard the rest.
        mask = np.abs(self.weights).sum(axis=1) > 0
        nonzero_idx = np.argwhere(mask)

        new_feature_vocab = {}
        next_feature_index = 0
        for feature, idx in sorted(self.feature_vocab.items(), key=lambda x: x[1]):
            if idx in nonzero_idx:
                new_feature_vocab[feature] = next_feature_index
                next_feature_index += 1

        self.feature_vocab = new_feature_vocab
        # Can't use nonzero_idx to index here because it has the wrong dimensions.
        self.weights = self.weights[mask]

    def ternarize(self) -> None:
        """Final transformation of averaged weights into strict int8 ternary values.

        This happens after weight averaging.
        """
        self.update_ternary_threshold()
        self.weights = self._ternarize_weights(self.weights).astype(np.int8)
