#!/usr/bin/env python3

import logging
from collections import defaultdict
from math import exp

logger = logging.getLogger(__name__)


class AveragedPerceptronBISECL:
    def __init__(self) -> None:
        # Dict of weights for each feature.
        # Keys are features, the value for each key is another dict where the keys are
        # the labels where the weight is non-zero and the value is the weight.
        self.weights = {}

        # Set of possible token labels.
        self.labels = set()

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
        return f"AveragedPerceptronBISECL(labels={self.labels})"

    def label_features(
        self,
        current_labels: dict[int, str],
        index: int,
        token_stem: str,
        token_pos: str,
        label: str,
        sequence_length: int,
    ) -> set[str]:
        """Generate set of features based on labels around the current index, the
        candidate label and the token stem for the current index.

        Parameters
        ----------
        current_labels : dict[int, str]
            Dict mapping index to label. Only contains labels that have been assigned so
            far.
        index : int
            Index of token under consideration.
        token_stem : str
            Token stem for token under consideration.
        token_pos : str
            Token part of speech tag for token under consideration.
        label : str
            Candidate label for current token.
        sequence_length : int
            Number of tokens in sequence, used to determine when at the start or end.

        Returns
        -------
        set[str]
            Set of feature strings.
        """

        def get_context_label(index):
            if index < 0:
                return "_START_"
            elif index >= sequence_length:
                return "_END_"
            else:
                return current_labels.get(index, "_UNASSIGNED_")

        prev_label = get_context_label(index - 1)
        prev_label2 = get_context_label(index - 2)
        prev_label3 = get_context_label(index - 3)
        next_label = get_context_label(index + 1)
        next_label2 = get_context_label(index + 2)
        next_label3 = get_context_label(index + 3)

        return {
            "prev_label=" + prev_label,
            "prev_label2=" + prev_label2,
            "prev_label3=" + prev_label3,
            "prev_label2+prev_label=" + "+".join((prev_label2, prev_label)),
            "prev_label3+prev_label2=" + "+".join((prev_label3, prev_label2)),
            "prev_label3+prev_label2+prev_label="
            + "+".join((prev_label3, prev_label2, prev_label)),
            "prev_label+pos=" + prev_label + "+" + token_pos,
            "prev_label+stem=" + prev_label + "+" + token_stem,
            "prev_label+current_label=" + prev_label + "+" + label,
            "next_label=" + next_label,
            "next_label2=" + next_label2,
            "next_label3=" + next_label3,
            "next_label+pos=" + next_label + "+" + token_pos,
            "next_label+stem=" + next_label + "+" + token_stem,
            "next_label2+next_label=" + "+".join((next_label2, next_label)),
            "next_label3+next_label2=" + "+".join((next_label3, next_label2)),
            "next_label3+next_label2+next_label="
            + "+".join((next_label3, next_label2, next_label)),
            "next_label+current_label=" + next_label + "+" + label,
            "prev_label+pos+next_label="
            + "+".join((prev_label, token_pos, next_label)),
            "prev_label+stem+next_label="
            + "+".join((prev_label, token_stem, next_label)),
            "prev_label+current_label+next_label="
            + "+".join((prev_label, label, next_label)),
        }

    def _confidence(self, scores: dict[str, float]) -> dict[str, float]:
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
        scores : dict[str, float]
            Dict of scores for labels

        Returns
        -------
        dict[str, float]
            Dict of confidences for labels
        """
        max_score = max(scores.values())
        return {label: round(exp(s - max_score), 3) for label, s in scores.items()}

    def predict_sequence(
        self,
        features_seq: list[set[str]],
        token_stems: list[str],
        token_pos: list[str],
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
        current_labels: dict[int, str] = {}
        label_confidence: dict[int, float] = {}
        indices_to_label = set(range(len(features_seq)))

        while indices_to_label:
            best_score = -float("inf")
            best_candidate = (-1, "_UNASSIGNED_")
            scores: dict[int, dict[str, float]] = defaultdict(dict)

            for i in indices_to_label:
                stem = token_stems[i]
                pos = token_pos[i]
                for label in self.labels:
                    features = features_seq[i] | self.label_features(
                        current_labels, i, stem, pos, label, len(features_seq)
                    )
                    score = self._score(features, label)
                    scores[i][label] = score
                    if score > best_score:
                        best_score = score
                        best_candidate = (i, label)

            idx, label = best_candidate
            current_labels[idx] = label
            if return_score:
                label_confidence[idx] = self._confidence(scores[idx])[label]
            else:
                label_confidence[idx] = 1.0

            indices_to_label.remove(idx)

        return [
            (current_labels[i], label_confidence[i]) for i in range(len(features_seq))
        ]

    def _score(self, features: set[str], current_label: str) -> int:
        """Calculate score for current label given the features for the token at the
        current position and the given previous label.

        Parameters
        ----------
        features : set[str]
            Set of features for token at current position.
        current_label : str
            Label to calculate score for.

        Returns
        -------
        int
            Score
        """
        score = 0
        for feat in features:
            if feat not in self.weights:
                continue

            if self._feature_updates.get(feat, 0) < self.min_feat_updates:
                continue

            # Increment score by weight for current feature for current label
            score += self.weights[feat].get(current_label, 0)

        return score

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

        self.weights[feature][label] = weight + change

        # Increment feature update count
        self._feature_updates[feature] += 1

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

        # Update feature weights because truth != guess.
        # We decrement the features for the predicted label by -1 and increment the
        # features for the true label by one. If the same feature appears in both
        # feature sets, the net change is 0.
        # We do the weights updates this way because (unlike the greedy Averaged
        # Perceptron) the weights for the true sequence and different to the weights for
        # the (incorrect) predicted sequence - although only for the features related to
        # the previous label.
        for feat in predicted_features:
            # Get weights dict for current feature, or empty dict if new feature
            weights = self.weights.setdefault(feat, {})
            # Update weights for feature:
            # Decrement weight for predicted label by -1.
            self._update_feature(guess, feat, weights.get(guess, 0.0), -1.0)

        for feat in truth_features:
            # Get weights dict for current feature, or empty dict if new feature
            weights = self.weights.setdefault(feat, {})
            # Update weights for feature:
            # Increment weight for correct label by +1
            self._update_feature(truth, feat, weights.get(truth, 0.0), 1.0)

        return None

    def average_weights(self) -> None:
        """Average the value for each weight over all updates.

        Returns
        -------
        None
        """
        for feat, weights in self.weights.items():
            new_feat_weights = {}

            for label, weight in weights.items():
                key = (feat, label)
                # Update total to account for iterations since last update
                total = (
                    self._totals[key] + (self._iteration - self._tstamps[key]) * weight
                )
                averaged = total / float(self._iteration)

                if averaged:
                    new_feat_weights[label] = averaged

            self.weights[feat] = new_feat_weights

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

        # Count initial number of features that have at least one update.
        initial_feature_count = sum(
            1 for count in self._feature_updates.values() if count > 0
        )

        filtered_count = 0
        for feature in list(self.weights.keys()):
            if self._feature_updates.get(feature, 0) < self.min_feat_updates:
                del self.weights[feature]
                filtered_count += 1

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

        new_weights = {}
        remaining_count, initial_weight_count = 0, 0
        for feature, weights in self.weights.items():
            new_feature_weights = {
                label: weight
                for label, weight in weights.items()
                if abs(weight) >= min_abs_weight and weight != 0
            }

            initial_weight_count += sum(1 for w in weights.values() if w != 0)
            if new_feature_weights != {}:
                new_weights[feature] = new_feature_weights
                remaining_count += len(new_feature_weights)

        pruned_pc = 100 * (1 - remaining_count / initial_weight_count)
        logger.debug(
            (
                f"Pruned {pruned_pc:.2f}% of weights for having absolute "
                f"values small than {min_abs_weight}."
            )
        )
        self.weights = new_weights

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

        max_weight = 0
        for scores in self.weights.values():
            max_weight = max(max_weight, max(abs(w) for w in scores.values()))

        scale = (2 ** (nbits - 1) - 1) / max_weight

        new_weights = {}
        for feature, weights in self.weights.items():
            new_feature_weights = {}

            for label, weight in weights.items():
                quantized_weight = round(weight * scale)
                if quantized_weight != 0:
                    new_feature_weights[label] = quantized_weight

            new_weights[feature] = new_feature_weights

        self.weights = new_weights
        logger.debug(f"Quantized model weights using {nbits} bits of precision.")
