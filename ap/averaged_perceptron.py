#!/usr/bin/env python3

from collections import defaultdict


class AveragedPerceptron:
    def __init__(self) -> None:
        # Dict of weights for each feature.
        # Keys are features, the values is a dict with keys of labels where the weight
        # is non-zero.
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

    def predict(self, features: set[str]) -> str:
        """Predict the label for a token described by features set.

        Parameters
        ----------
        features : set[str]
            Set of features for token

        Returns
        -------
        str
            Class label for token
        """
        scores = defaultdict(float)
        for feat in features:
            if feat not in self.weights:
                continue

            weights = self.weights[feat]
            for label, weight in weights.items():
                scores[label] += weight

        # Sort by score, then alphabetically sort for stability
        return max(self.labels, key=lambda label: (scores[label], label))

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

        def upd_feat(label: str, feature: str, weight: float, change: float):
            """Update weight for feature by change

            Parameters
            ----------
            label : str
                Label to update weight for.
            feature : dict[str, str | bool]
                Feature to update
            weight : int
                Weight of label at time of last update.
            change : int
                Change to label weight.
            """
            key = (feature, label)
            # Update total for feature/label combo to account for number of iterations
            # since last update
            self._totals[key] += (self._iteration - self._tstamps[key]) * weight
            self._tstamps[key] = self._iteration

            self.weights[feature][label] = weight + change

        self._iteration += 1

        if truth == guess:
            return None

        for feat in features:
            # Get weights dict for current feature, or empty dict if new feature
            weights = self.weights.setdefault(feat, {})
            # Update weights for feature:
            # Increment weight for correct label by +1, decrement weights for
            # incorrect labels by -1.
            upd_feat(truth, feat, weights.get(truth, 0.0), 1.0)
            upd_feat(guess, feat, weights.get(guess, 0.0), -1.0)

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
