#!/usr/bin/env python3

import logging
from collections import defaultdict
from math import exp

logger = logging.getLogger(__name__)


class AveragedPerceptron:
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

        self._iteration: int = 0

    def __repr__(self):
        return f"AveragedPerceptron(labels={self.labels})"

    def _confidence(self, scores: dict[str, float]) -> list[float]:
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
            Dict of non-zero scores for labels

        Returns
        -------
        list[float]
            List of confidences for labels
        """
        max_score = max(scores.values())
        return [round(exp(s - max_score), 3) for s in scores.values()]

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
            If False, return scores is always 1.0.

        Returns
        -------
        tuple[str, float]
            Class label given features and confidence value.
        """
        scores = defaultdict(float)
        for feat in features:
            if feat not in self.weights:
                continue

            weights = self.weights[feat]
            for label, weight in weights.items():
                scores[label] += weight

        # Sort by score, then alphabetically sort for stability
        best_label = max(
            self.labels - constrained_labels, key=lambda label: (scores[label], label)
        )

        if return_score:
            best_confidence = max(self._confidence(scores))
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

        self.weights[feature][label] = weight + change

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
            # Get weights dict for current feature, or empty dict if new feature
            weights = self.weights.setdefault(feat, {})
            # Update weights for feature:
            # Increment weight for correct label by +1, decrement weights for
            # incorrect labels by -1.
            self._update_feature(truth, feat, weights.get(truth, 0.0), 1.0)
            self._update_feature(guess, feat, weights.get(guess, 0.0), -1.0)

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

    def prune_weights(self, min_abs_weight: float) -> None:
        """Prune weights by removing weights smaller than min_abs_weight.

        Parameters
        ----------
        min_abs_weight : float
            Minimum absolute value of weight to keep.
        """
        new_weights = {}
        pruned_count, initial_weight_count = 0, 0
        for feature, weights in self.weights.items():
            new_feature_weights = {
                label: weight
                for label, weight in weights.items()
                if abs(weight) >= min_abs_weight and weight != 0
            }

            if new_feature_weights != {}:
                new_weights[feature] = new_feature_weights

            initial_weight_count += len(weights)
            pruned_count += len(weights) - len(new_feature_weights)

        pruned_pc = 100 * pruned_count / initial_weight_count
        logger.debug(f"Pruned {pruned_pc:.2f}% of weights.")
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
                # If the weight quantizes to zero, we can discard it as there is no need
                # to save features that have weight of 0.
                if quantized_weight != 0:
                    new_feature_weights[label] = quantized_weight

            new_weights[feature] = new_feature_weights

        self.weights = new_weights
        logger.debug(f"Quantized model weights using {nbits} of precision.")
