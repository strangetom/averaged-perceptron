#!/usr/bin/env python3

import logging
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from math import exp

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
        "prev_label+pos=" + prev_label + "+" + current_pos_tag,
    }


class AveragedPerceptronViterbi:
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
        return f"AveragedPerceptronViterbi(labels={self.labels})"

    def _confidence(self, scores: dict[str, LatticeElement]) -> list[float]:
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
        scores : dict[str, LatticeElement]
            Dict of non-zero scores for labels

        Returns
        -------
        list[float]
            List of confidences for labels
        """
        max_score = max(s.score for s in scores.values())
        return [round(exp(s.score - max_score), 3) for s in scores.values()]

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
            If False, return score is always 1.
        constrain_transitions : bool, optional
            If True, constrain label transitions to only allowed transitions.
            Default is True

        Returns
        -------
        list[tuple[str, float]]
            List of (label, score) tuples for sequence.
        """
        seq_len = len(features_seq)
        labels = list(sorted(self.labels, reverse=True))

        # Define the lattice.
        # For each element in the sequence of features, we define a dictionary. The dict
        # keys are the possible labels for that item. The values are the LatticeElement
        # dataclass which stores the best score for that label and a backpointer to the
        # previous label that resulted in that score.
        lattice: list[defaultdict[str, LatticeElement]] = [
            defaultdict(lambda: LatticeElement(-float("inf"), ""))
            for _ in range(seq_len)
        ]

        # Initialise for first feature set of features_seq
        pos = self._get_pos_from_features(features_seq[0])
        for current_label in labels:
            if constrain_transitions and current_label == "I_NAME_TOK":
                # I_NAME_TOK cannot be the first label because it must follow B_NAME_TOK
                continue

            score = self._score(
                features_seq[0] | label_features("-START-", pos), current_label
            )
            # Select the best score, store it and set the backpointer to the
            # previous label that resulted in this score.
            if score > lattice[0][current_label].score:
                lattice[0][current_label].score = score
                lattice[0][current_label].backpointer = "-START-"

        # Forward pass, starting at t=1 because we've already initialised t=0
        for t, features in enumerate(features_seq[1:], 1):
            # Extract POS tag for current feature set.
            pos = self._get_pos_from_features(features)

            # Iterate over all combinations of previous and current labels for each
            # set of features.
            # Calculate the score for each label combination and store the best score
            # plus the backpointer to the previous label that yielded that score.
            for current_label, prev_label in product(labels, labels):
                if constrain_transitions and current_label in ILLEGAL_TRANSITIONS.get(
                    prev_label, set()
                ):
                    # If transition from prev_label to current_label is forbidden, do
                    # not calculate the score.
                    continue

                if constrain_transitions and current_label == "I_NAME_TOK":
                    # If current_label is I_NAME_TOK, check if B_NAME_TOK has occurred
                    # in the best path up to prev_label, either since the start
                    # of the sequence or since the last NAME_SEP.
                    if not self._b_name_tok_has_occurred(lattice[:t], prev_label):
                        continue

                score = lattice[t - 1][prev_label].score + self._score(
                    features | label_features(prev_label, pos), current_label
                )
                # Select the best score, store it and set the backpointer to the
                # previous label that resulted in this score.
                if score > lattice[t][current_label].score:
                    lattice[t][current_label].score = score
                    lattice[t][current_label].backpointer = prev_label

        # Back tracking
        label_seq = []
        scores = []

        # Find the best label for the last element of the lattice, since there isn't a
        # backpointer for this.
        backpointer = self._argmax(labels, lattice[-1])
        # Iterate backwards through the lattice.
        # At each step, append the backpointer that yielded the best score to the label
        # sequence. Note the the resultant label sequence will be in reverse.
        for score_dict in reversed(lattice):
            label_seq.append(backpointer)
            if return_score:
                # Calculate score for each label and append score for backpointer.
                confidences = {
                    label: conf
                    for label, conf in zip(
                        score_dict.keys(), self._confidence(score_dict)
                    )
                }
                scores.append(confidences[backpointer])
            else:
                scores.append(1.0)
            backpointer = score_dict[backpointer].backpointer

        return list(zip(reversed(label_seq), reversed(scores)))

    def _b_name_tok_has_occurred(
        self, lattice: list[defaultdict[str, LatticeElement]], end_label: str
    ) -> bool:
        """Check if B_NAME_TOK has occurred in best sequence ending with prev_label.

        If NAME_SEP occurs in the best sequence, only check if B_NAME_TOK has occurred
        since the last NAME_SEP.

        Parameters
        ----------
        lattice : list[defaultdict[str, LatticeElement]]
            Viterbi lattice
        end_label : str
            Label at end of sequence.

        Returns
        -------
        bool
            True if B_NAME_TOK has occurred in sequence since beginning or
            last NAME_SEP.
        """
        label_seq = []
        backpointer = end_label
        for score_dict in reversed(lattice):
            label_seq.append(backpointer)
            backpointer = score_dict[backpointer].backpointer

        # label_seq is generated from the end, moving towards the start, so reverse.
        label_seq = list(reversed(label_seq))

        if "NAME_SEP" in label_seq:
            # Find index of last occurrence of NAME_SEP in sequence
            name_sep_idx = max(i for i, v in enumerate(label_seq) if v == "NAME_SEP")
            if "B_NAME_TOK" in label_seq[name_sep_idx:]:
                return True
        else:
            if "B_NAME_TOK" in label_seq:
                return True

        return False

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

            # Increment score by weight for current feature for current label
            score += self.weights[feat].get(current_label, 0)

        return score

    def _argmax(
        self, labels: list[str], label_scores: dict[str, LatticeElement]
    ) -> str:
        """Return best scoring label from dict of label scores.

        If the best score if the same for multiple labels, return the label that is
        alphabetically first. This is done for stability.

        Parameters
        ----------
        labels : list[str]
            List of labels, sorted alphabetically
        label_scores : dict[str, LatticeElement]
            Dict of {label: LatticeElement}.

        Returns
        -------
        str
            Label with best score.
        """
        return max(labels, key=lambda label: (label_scores[label].score, label))

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
                if abs(weight) >= min_abs_weight or weight != 0
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
                if quantized_weight != 0:
                    new_feature_weights[label] = quantized_weight

            new_weights[feature] = new_feature_weights

        self.weights = new_weights
        logger.debug(f"Quantized model weights using {nbits} of precision.")
