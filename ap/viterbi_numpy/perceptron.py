#!/usr/bin/env python3

import logging
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from ap._constants import ILLEGAL_TRANSITIONS

logger = logging.getLogger(__name__)


@dataclass
class LatticeElement:
    """Dataclass for holding the score and backpointer for an element in the Viterbi
    lattice.

    Attributes
    ----------
    score : float
        The best score for the current label calculated from current label combined with
        all possible previous labels.
    backpointer : str
        The previous label that, when combined with the current label, yielded the best
        score.
    """

    score: float
    backpointer: str


@lru_cache
def label_features(prev_label: str, current_pos_tag: str) -> set[str]:
    """Generate set of features based on previous label and current part of speech
    tag.

    Parameters
    ----------
    prev_label : str
        Label of previous token.
    current_pos_tag : str
        Part of speech tag for current token.

    Returns
    -------
    set[str]
        Set of feature strings.
    """
    return {
        "prev_label=" + prev_label,
        # "prev_label+pos=" + prev_label + "+" + current_pos_tag,
    }


class AveragedPerceptronViterbiNumpy:
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
        # We need to include -START- for indexing into the transition weights matrix
        self.label_to_idx = {
            label: i for i, label in enumerate([*self.labels, "-START-"])
        }
        self.n_labels = len(self.labels)

        # Dict mapping transition feature to index
        self.transition_feature_to_idx = {
            "prev_label=" + label: i for label, i in self.label_to_idx.items()
        }

        self.constraint_mask = self._precompute_constraint_mask()

        self.training_mode = training_mode
        if self.training_mode:
            self.feature_vocab: dict[str, int] = {}
            self.next_feature_index = 0

            initial_features = 10_000
            # Matrix of emission weights: [n_features, n_labels]
            # These are weights for features that are properties of an element of the
            # sequence, independent of previous labels.
            self.emission_weights = np.zeros(
                (initial_features, self.n_labels), dtype=np.float32
            )
            # Matrix of transition weights: [n_labels + 1, n_labels]
            # These are weights for features based on the previous label (rows) and
            # current label (columns).
            # Since the previous label can also be -START-, we include an additional row
            # for that label.
            self.transition_weights = np.zeros(
                (self.n_labels + 1, self.n_labels), dtype=np.float32
            )

            # Accumulated weight for each feature/label combination
            self._emission_totals = np.zeros(
                (initial_features, self.n_labels), dtype=np.float64
            )
            self._transition_totals = np.zeros(
                (self.n_labels + 1, self.n_labels), dtype=np.float64
            )
            # The last iteration each feature/label combination was changed.
            # This is to avoid having to update every key in self._totals every
            # iteration, we only update each feature/label key when it changes,
            # accounting for the iterations since it last changed.
            self._emission_tstamps = np.zeros(
                (initial_features, self.n_labels), dtype=np.int32
            )
            self._transition_tstamps = np.zeros(
                (self.n_labels + 1, self.n_labels), dtype=np.int32
            )

            # Count the number of times each feature has been updated (independent of
            # the label) during training.
            self._emission_feat_updates = np.zeros((initial_features,), dtype=np.int32)
            self._transition_feat_updates = np.zeros(
                (self.n_labels + 1,), dtype=np.int32
            )
            # The minimum number of feature updates required to consider the feature in
            # the prediction step.
            self.min_feat_updates: int = 0

            self._iteration: int = 0

        else:
            # Initialise to empty objects for inference.
            # These will need to be set to real values prior to inference.
            self.emission_weights = np.array([])
            self.transition_weights = np.array([])
            self.feature_vocab: dict[str, int] = {}

    def __repr__(self):
        return f"AveragedPerceptronViterbi(labels={self.labels})"

    def _resize_emission_matrices(self) -> None:
        """Resize emission matrices by doubling the number of rows.

        This is called by IngredientTagger during training once the vocabulary reaches
        the current number of rows of the matrices.

        Note that we don't need to resize the transition matrices because we know the
        full size of them,
        """
        current_capacity = self.emission_weights.shape[0]
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

        self.emission_weights = resize_mat(self.emission_weights)
        self._emission_totals = resize_mat(self._emission_totals)
        self._emission_tstamps = resize_mat(self._emission_tstamps)
        self._emission_feat_updates = resize_mat(self._emission_feat_updates)

    def _precompute_constraint_mask(self) -> np.ndarray:
        """Compute constraint mask.

        This is a boolean matrix of shape (n_labels, n_labels) where a value of 1 means
        that the transition from previous label (row) to current label (column) is
        forbidden.

        Returns
        -------
        np.ndarray
            Boolean matrix indicating forbidden transitions.

        """
        mask = np.zeros((self.n_labels, self.n_labels), dtype=np.bool_)

        for prev_label, constrained_labels in ILLEGAL_TRANSITIONS.items():
            prev_idx = self.label_to_idx[prev_label]
            for idx in [self.label_to_idx[label] for label in constrained_labels]:
                mask[prev_idx, idx] = 1

        return mask

    def _features_to_idx(self, features: set[str]) -> np.ndarray:
        """Map feature string to indices by lookup in vocab dict.

        If insert_missing is True, add feature missing from vocab dict.

        Whilst order is not important (hence this function accepting a set), we return
        a list because we can use that directly when indexing NumPy arrays.

        Parameters
        ----------
        features : set[str]
            Set of feature strings to return indices of.

        Returns
        -------
        np.ndarray
            NumPy array of integer indices for string features.
        """
        if self.training_mode:
            for feat in features:
                if (
                    feat not in self.feature_vocab
                    and feat not in self.transition_feature_to_idx
                ):
                    self.feature_vocab[feat] = self.next_feature_index
                    self.next_feature_index += 1

            # Resize matrices if the vocab now exceeds the current number of rows.
            if self.next_feature_index >= self.emission_weights.shape[0]:
                self._resize_emission_matrices()

        return np.array(
            [
                self.feature_vocab[feat]
                for feat in features
                if feat in self.feature_vocab
            ]
        )

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
            NumPy array of scores for each label.

        Returns
        -------
        np.ndarray
            NumPy array of confidence
        """
        max_score = np.max(scores)
        exp_scores = np.exp(scores - max_score)
        probability = exp_scores / exp_scores.sum()
        return np.round(probability, 3)

    def predict_sequence(
        self,
        features_seq: list[set[str]],
        return_score: bool = False,
        constrain_transitions: bool = True,
    ) -> list[tuple[str, float]]:
        """Predict the label sequence for a sequence of tokens described by sequence of
        features sets using Viterbi algorithm.

        Parameters
        ----------
        features_seq : list[set[str]]
            List of sets of features for tokens in sequence.
        return_score : bool, optional
            If True, return score for each predicted label.
            If False, returned score is always 1.0.
        constrain_transitions : bool, optional
            If True, constrain label transitions to only allowed transitions.
            Default is True

        Returns
        -------
        list[tuple[str, float]]
            List of (label, score) tuples for sequence.
        """
        seq_len = len(features_seq)

        # Pre-compute emission scores for all elements of sequence.
        # Rows: sequence elements
        # Columns: labels
        emission_scores = np.zeros((seq_len, self.n_labels), dtype=np.float64)
        for t, features in enumerate(features_seq):
            indices = self._features_to_idx(features)
            # Don't consider features until they have been updated more than
            # min_feat_updates times. We only do this during training.
            if self.training_mode:
                indices = indices[
                    self._emission_feat_updates[indices] >= self.min_feat_updates
                ]

            if len(indices) > 0:
                # Sum the weights for the selected features by column (label) and assign
                # to the correct row of the emission_scores matrix.
                emission_scores[t] = self.emission_weights[indices].sum(axis=0)

        # Get indices for constraint-specific labels
        b_name_idx = self.label_to_idx.get("B_NAME_TOK")
        i_name_idx = self.label_to_idx.get("I_NAME_TOK")
        name_sep_idx = self.label_to_idx.get("NAME_SEP")

        # Auxiliary matrix to track if B_NAME_TOK has occurred in the best path
        # for each label at each time step since the beginning or last NAME_SEP.
        # Rows: sequence elements
        # Columns: labels
        has_b_name = np.zeros((seq_len, self.n_labels), dtype=bool)

        # Initialize the lattice as NumPy arrays.
        # One array for the scores, initialized to -inf. This is the best score for that
        # label given the previous label specified by the backpointers array.
        # One array for the backpointers, which hold the index of the previous label
        # that resulted in the score in the lattice_scores array.
        lattice_scores = np.full((seq_len, self.n_labels), -np.inf)
        backpointers = np.zeros((seq_len, self.n_labels), dtype=np.int32)

        # Deal with the first element of the sequence separately because the previous
        # label is always "-START-".
        transition_weights = self.transition_weights[self.label_to_idx["-START-"]]
        if (
            self.training_mode
            and self._transition_feat_updates[self.label_to_idx["-START-"]]
            < self.min_feat_updates
        ):
            # Ignore the transition weights if they haven't been updated at least
            # min_feat_updates times.
            transition_weights[:] = 0
        lattice_scores[0] = transition_weights + emission_scores[0]
        backpointers[0] = self.label_to_idx["-START-"]  # Not strictly necessary

        # Apply initial constraints (e.g., I_NAME_TOK cannot be first)
        if constrain_transitions:
            lattice_scores[0, i_name_idx] = -np.inf

            # Update has_b_name matrix for first sequence element
            has_b_name[0, b_name_idx] = True

        # Forward pass, starting at t=1 because we've already initialised t=0
        for t in range(1, seq_len):
            # Get the scores for each label from the previous lattice row.
            # [:, np.newaxis] rotates this into a column vector because this is the
            # previous label to the current label, so we need to broadcast across the
            # rows of the transition matrix.
            prev_el_scores = lattice_scores[t - 1][:, np.newaxis]

            # When training, we need to ignore transition features that haven't been
            # updated at least min_feature_updates times (we've already handled this
            # for emission features when creating the emission_score matrix).
            # Create a copy of the transition matrix. If we're in training mode, zero
            # out rows corresponding to labels that haven't updated enough times (these
            # are the labels that form the prev_label=... feature).
            transition_weights = self.transition_weights.copy()
            if self.training_mode:
                transition_weights[
                    self._transition_feat_updates < self.min_feat_updates
                ] = 0

            # Candidates is a (n_label, n_label) shaped matrix containing the total
            # scores for transition from each previous label to the current label.
            # We broadcast the prev_el_scores across all rows in the transition
            # matrix and broadcast the emission_scores across all columns to end up
            # with the sum of relevant weights for each label -> label transition.
            #
            # We have to slice the transition weights to remove the last row which
            # corresponds to the "-START-" label. We have already handled this for the
            # first element of the sequence and the dimensions of prev_el_scores would
            # not match if we didn't remove it here.
            candidates = (
                prev_el_scores
                + transition_weights[: self.n_labels]
                + emission_scores[t]
            )

            # Force the scores from constrained transitions to -inf
            if constrain_transitions:
                candidates[self.constraint_mask] = -np.inf

                # Mask transitions to I_NAME_TOK from paths that lack a B_NAME_TOK
                invalid_prev_paths = ~has_b_name[t - 1]
                candidates[invalid_prev_paths, i_name_idx] = -np.inf

            # Find the best score in each column and the index of the best score in each
            # column and save to the lattice_scores and backpointers matrices
            # respectively.
            lattice_scores[t] = np.max(candidates, axis=0)
            backpointers[t] = np.argmax(candidates, axis=0)

            if constrain_transitions:
                # Update has_b_name matrix
                # Inherit state from the best predecessor for each current label.
                # We are setting the value of for each column to the value from the
                # previous row (i.e. t-1) at the index given by backpointers[t] so that
                # we inherit whether the best sequence has a B_NAME_TOk.
                has_b_name[t] = has_b_name[t - 1, backpointers[t]]
                # If current label is B_NAME_TOK, the path now has a B_NAME_TOK
                has_b_name[t, b_name_idx] = True
                # If current label is NAME_SEP, the B_NAME_TOK requirement resets
                has_b_name[t, name_sep_idx] = False

        # Back tracking
        label_seq = []
        scores = []

        # Find the best label for the last element of the lattice, since there isn't a
        # backpointer for this.
        backpointer = np.argmax(lattice_scores[-1])
        # Iterate backwards through the lattice.
        # At each step, append the backpointer that yielded the best score to the label
        # sequence. Note the the resultant label sequence will be in reverse.
        for t in range(seq_len - 1, -1, -1):
            label_seq.append(self.labels[backpointer])
            if return_score:
                # Calculate score for each label and append score for backpointer.
                confidences = self._confidence(lattice_scores[t])
                scores.append(confidences[backpointer])
            else:
                scores.append(1.0)
            backpointer = backpointers[t, backpointer]

        return list(zip(reversed(label_seq), reversed(scores)))

    def _get_pos_from_features(self, features: set[str]) -> str:
        """Get the part of speech tag from the current set of features.

        A feature set always contains a "pos=xxx" feature, from which can extract the
        part of speech tag.

        Parameters
        ----------
        features : set[str]
            Set of features.

        Returns
        -------
        str
            Part of speech tag.

        Raises
        ------
        ValueError
            Raised if no part of speech feature found.
        """
        pos = ""
        for feat in features:
            if feat.startswith("pos="):
                pos = feat.split("pos=")[-1]
                return pos

        raise ValueError("Unable to extract POS tag from features.")

    def _update_emission_totals(
        self,
        label_index: int,
        feature_indices: np.ndarray,
    ) -> None:
        """Update weights for emission feature by given change

        Parameters
        ----------
        label_index : int
            Index of label to update total for.
        feature_indices : np.ndarray
            NumPy array of indices corresponding to features to update weights for.
        feature_type : Literal["emission", "transition"]
            Type of feature, to ensure the correct matrices are updated.
        """
        # Calculate how many iterations passed since this weight was last touched
        iters = self._iteration - self._emission_tstamps[feature_indices, label_index]

        # Add the accumulated weight to totals
        # totals += iterations_passed * current_weight
        self._emission_totals[feature_indices, label_index] += (
            iters * self.emission_weights[feature_indices, label_index]
        )

        # Update timestamp to current iteration
        self._emission_tstamps[feature_indices, label_index] = self._iteration

        return None

    def _update_transition_totals(
        self,
        label_index: int,
        feature_indices: np.ndarray,
    ) -> None:
        """Update weights for transition feature by given change

        Parameters
        ----------
        label_index : int
            Index of label to update total for.
        feature_indices : np.ndarray
            NumPy array of indices corresponding to features to update weights for.
        feature_type : Literal["emission", "transition"]
            Type of feature, to ensure the correct matrices are updated.
        """
        # Calculate how many iterations passed since this weight was last touched
        iters = self._iteration - self._transition_tstamps[feature_indices, label_index]

        # Add the accumulated weight to totals
        # totals += iterations_passed * current_weight
        self._transition_totals[feature_indices, label_index] += (
            iters * self.transition_weights[feature_indices, label_index]
        )

        # Update timestamp to current iteration
        self._transition_tstamps[feature_indices, label_index] = self._iteration

        return None

    def update(
        self,
        truth: str,
        guess: str,
        predicted_features: set[str],
        truth_features: set[str],
    ) -> None:
        """Update weights for given features.

        This only makes changes if the true and predicted labels are different.

        Parameters
        ----------
        truth : str
            True label for given features.
        guess : str
            Predicted label for given features.
        predicted_features : set[str]
            Features for predicted sequence.
        truth_features : set[str]
            Features for true (correct) sequence.

        Returns
        -------
        None
        """
        self._iteration += 1

        if truth == guess:
            return None

        truth_idx = self.label_to_idx[truth]
        guess_idx = self.label_to_idx[guess]

        # Update feature weights because truth != guess.
        # We decrement the features for the predicted label by -1 and increment the
        # features for the true label by one.
        # We do the weights updates this way (unlike the greedy Averaged Perceptron)
        # because the features for the true sequence are different to the weights for
        # the (incorrect) predicted sequence - although only for the features related to
        # the previous label.
        #
        # Decrement the weights for the predicted features first.
        predicted_emission_feat_idx = self._features_to_idx(predicted_features)
        predicted_transition_feat_idx = np.array(
            [
                self.transition_feature_to_idx[feat]
                for feat in predicted_features
                if feat in self.transition_feature_to_idx
            ]
        )
        self._update_emission_totals(guess_idx, predicted_emission_feat_idx)
        self.emission_weights[predicted_emission_feat_idx, guess_idx] -= 1.0
        self._update_transition_totals(guess_idx, predicted_transition_feat_idx)
        self.transition_weights[predicted_transition_feat_idx, guess_idx] -= 1.0

        # Now increment weights for truth features
        truth_emission_feat_idx = self._features_to_idx(truth_features)
        truth_transition_feat_idx = np.array(
            [
                self.transition_feature_to_idx[feat]
                for feat in truth_features
                if feat in self.transition_feature_to_idx
            ]
        )
        self._update_emission_totals(truth_idx, truth_emission_feat_idx)
        self.emission_weights[truth_emission_feat_idx, truth_idx] += 1.0
        self._update_transition_totals(truth_idx, truth_transition_feat_idx)
        self.transition_weights[truth_transition_feat_idx, truth_idx] += 1.0

        # Increment feature update counts for all features updated.
        emission_feature_idx = np.union1d(
            predicted_emission_feat_idx, truth_emission_feat_idx
        )
        self._emission_feat_updates[emission_feature_idx] += 1
        transition_feature_idx = np.union1d(
            predicted_transition_feat_idx, truth_transition_feat_idx
        )
        self._transition_feat_updates[transition_feature_idx] += 1

        return None

    def average_weights(self) -> None:
        """Average the value for each weight over all updates.

        Returns
        -------
        None
        """
        # Final pass to bring all totals up to date
        emission_iters = self._iteration - self._emission_tstamps
        self._emission_totals += emission_iters * self.emission_weights

        transition_iters = self._iteration - self._transition_tstamps
        self._transition_totals += transition_iters * self.transition_weights

        # Average weights
        self.emission_weights = self._emission_totals / self._iteration
        self.transition_weights = self._transition_totals / self._iteration

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

        # Initial feature count is number of features that have ever been updated.
        initial_feature_count = np.count_nonzero(
            self._emission_feat_updates
        ) + np.count_nonzero(self._transition_feat_updates)

        # Set row indices where feature has not been updated at least
        # min_feature_updates times.
        # We're only considering rows with at least one feature update here to make the
        # calculation of the logging message easier. We could just do
        # np.argwhere(self._feat_updates < self.min_feat_updates), but that would also
        # include features that have never been updated.
        emission_idx_to_zero = np.argwhere(
            np.logical_and(
                0 < self._emission_feat_updates,
                self._emission_feat_updates < self.min_feat_updates,
            )
        )
        transition_idx_to_zero = np.argwhere(
            np.logical_and(
                0 < self._transition_feat_updates,
                self._transition_feat_updates < self.min_feat_updates,
            )
        )

        # Set weights for these rows to 0.
        # We also need to set the rows in the _totals matrix to zero for these too so
        # they remain 0 after averaging.
        self.emission_weights[emission_idx_to_zero, :] = 0
        self._emission_totals[emission_idx_to_zero, :] = 0
        self.transition_weights[transition_idx_to_zero, :] = 0
        self._transition_totals[transition_idx_to_zero, :] = 0

        filtered_count = emission_idx_to_zero.shape[0] + transition_idx_to_zero.shape[0]

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

        initial_weight_count = np.count_nonzero(
            self.emission_weights
        ) + np.count_nonzero(self.transition_weights)
        self.emission_weights[np.abs(self.emission_weights) < min_abs_weight] = 0
        self.transition_weights[np.abs(self.transition_weights) < min_abs_weight] = 0
        remaining_count = np.count_nonzero(self.emission_weights) + np.count_nonzero(
            self.transition_weights
        )
        pruned_pc = 100 * (1 - remaining_count / initial_weight_count)

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

        # Choose an appropriate type to minimise model size.
        if nbits <= 8:
            type_ = np.int8
        elif nbits <= 16:
            type_ = np.int16
        else:
            type_ = np.int32

        max_weight = max(np.max(self.emission_weights), np.max(self.transition_weights))
        scale = (2 ** (nbits - 1) - 1) / max_weight
        self.emission_weights = np.round(self.emission_weights * scale).astype(type_)
        self.transition_weights = np.round(self.transition_weights * scale).astype(
            type_
        )
        logger.debug(f"Quantized model weights using {nbits} bits of precision.")

    def simplify_weights(self) -> None:
        """Simplify weights matrix by discarding any rows that are all zeros.

        We also simplify the feature vocab to remove the entries corresponding to those
        rows too.

        We do not need to do this for the transition weights because the size of that
        matrix is known ahead of time.
        """
        # Find row indices where absolute sum of weights is non zero. We keep these and
        # discard the rest.
        mask = np.abs(self.emission_weights).sum(axis=1) > 0
        nonzero_idx = np.argwhere(mask)

        new_feature_vocab = {}
        next_feature_index = 0
        for feature, idx in sorted(self.feature_vocab.items(), key=lambda x: x[1]):
            if idx in nonzero_idx:
                new_feature_vocab[feature] = next_feature_index
                next_feature_index += 1

        self.feature_vocab = new_feature_vocab
        # Can't use nonzero_idx to index here because it has the wrong dimensions.
        self.emission_weights = self.emission_weights[mask]
