#!/usr/bin/env python3

from collections import defaultdict
from math import exp, log


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
        """Calculate (log) confidence for each labels.

        Can be converted to the 0-1 range by exp() each element

        Parameters
        ----------
        scores : dict[str, float]
            Dict of non-zero scores for labels

        Returns
        -------
        list[float]
            List of (log) confidences for labels
        """
        exps = [exp(s) for s in scores.values()]
        exp_sum = sum(exps)
        return [round(log(ex) - log(exp_sum), 3) for ex in exps]

    def predict(self, features: set[str]) -> tuple[str, float]:
        """Predict the label for a token described by features set.

        Parameters
        ----------
        features : set[str]
            Set of features for token

        Returns
        -------
        tuple[str, float]
            Class label for token and confidence value
        """
        scores = defaultdict(float)
        for feat in features:
            if feat not in self.weights:
                continue

            weights = self.weights[feat]
            for label, weight in weights.items():
                scores[label] += weight

        # Sort by score, then alphabetically sort for stability
        best_label = max(self.labels, key=lambda label: (scores[label], label))
        best_confidence = max(self._confidence(scores))

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
                averaged = round(total / float(self._iteration), 3)

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
        for feature, weights in self.weights.items():
            new_feature_weights = {}

            for label, weight in weights.items():
                if abs(weight) >= min_abs_weight:
                    new_feature_weights[label] = weight

            if new_feature_weights != {}:
                new_weights[feature] = new_feature_weights

        self.weights = new_weights

    def quantize(self) -> None:
        """Quantize weights to int8."""
        max_weight = 0
        for _, scores in self.weights.items():
            max_weight = max(max_weight, max(abs(w) for w in scores.values()))

        scale = 127 / max_weight

        new_weights = {}
        for feature, weights in self.weights.items():
            new_feature_weights = {}

            for label, weight in weights.items():
                new_feature_weights[label] = round(weight * scale)

            new_weights[feature] = new_feature_weights

        self.weights = new_weights
