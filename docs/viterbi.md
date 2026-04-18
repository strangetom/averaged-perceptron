# Viterbi Algorithm

## Introduction

A limitation of the basic implementation of the Averaged Perceptron model is that it is greedy when it's predicting the label for a token. This is because it predicts the label for each token in turn and selects the best label for that token. This is a limitation because once the model has predicted a label, it commits to that label. This might be optimal *locally* but might not be optimal *globally*, for the whole sentence.

The Viterbi algorithm is a way to address this limitation by choosing the sequence of labels that maximise the likelihood for the whole sentence, rather than each individual token.

This comes at a cost though. The Viterbi algorithm is a more complex algorithm to implement and the run time performance significantly slower.

## How It Works with the Averaged Perceptron

> [!NOTE]
>
> This is not a general explanation of the Viterbi algorithm. Instead, we focus on the specific implementation for this application.

### Algorithm Outline

Rather than commit to a label for a given token as we process each token, we will instead consider all possible label transitions for each token, score them and keep track of those scores. Once we have done this across the whole sequence, we can backtrack through the scores for each token and select the path that maximises the overall score.

Therefore, the algorithm has three parts:

1. Initialise the *lattice*.
2. Iterate *forward* through the sequence to calculate scores and store them in the lattice.
3. Back track through the lattice and use the scores the select the best path.

The lattice is the data structure we will use to keep track of the scores during the forward path. In general terms, the lattice is a $n\times m$ matrix, where $n$ is the number of labels and $m$ is the number of tokens.

$$
\begin{bmatrix}
 l_{QTY,0} & l_{QTY,1} & l_{QTY,2} & l_{QTY,3} & \cdots\\
 l_{UNIT,0} & l_{UNIT,1} & l_{UNIT,2} & l_{UNIT,3} & \cdots\\
 l_{SIZE,0} & l_{SIZE,1} & l_{SIZE,2} & l_{SIZE,3} & \cdots\\
 l_{PREP,0} & l_{PREP,1} & l_{PREP,2} & l_{PREP,3} & \cdots\\
 \vdots & \vdots & \vdots & \vdots & \ddots
\end{bmatrix}
$$


The lattice is populated one column at a time. Each element of a column represents a possible current label for the given token. We want to calculate the score for each transition from the previous label to the current label, where the previous label can be any of the possible labels. The information we store at that element is the best score and the previous label that resulted in that score.

![viterbi-lattice-forward](viterbi-lattice-forward.svg)

> **Why do we only keep the best score?**
>
> Once we've calculated all these scores and kept the best score of each (previous label, current label) pair, we then backtrack through the lattice to pick the path with the maximum overall score.
>
> If a given lattice element is in that path, then the only way to maximise the global score is the use the best score for that element. Using any other score will necessarily result in a lower overall score. Therefore, we don't need to keep track of the lower scores because they can never result in a higher global score.

Once the lattice has been populated, we can back track through it to find the best label sequence. We start with the label that has the highest score for the last element. We know what the previous label has to be to get that score, so that becomes the label before. We also know that to maximise the score for that label, we have to use the previous label we stored that resulted in that maximum score. And so on, and so on.

The result then the sequence of labels (in reverse) for the sentence that maximises the overall likelihood for that sentence.

### Implementation

#### 1. Initialise the Lattice

We will define a `LatticeElement` dataclass to store the best score and backpointer to the label for the previous token that resulted in that score.

```python
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
```

We then initialise the lattice.

```python
# Define the lattice.
# For each element in the sequence of features, we define a dictionary. The dict
# keys are the possible labels for that item. The values are the LatticeElement
# dataclass which stores the best score for that label and a backpointer to the
# previous label that resulted in that score.
lattice = [
    defaultdict(lambda: LatticeElement(-float("inf"), ""))
    for _ in range(seq_len)
]
```

#### 2. Iterate Forward Through the Sequence

The first column of the lattice has to handled specially. For all columns after this one, we can transition from any possible to label to any current label. However for the first column, we can only transition from the **-START-** label any current label.

```python
# Initialise for first feature set of features_seq
pos = self._get_pos_from_features(features_seq[0])
for current_label in labels:
    score = self._score(
        features_seq[0] | label_features("-START-", pos), current_label
    )
    # Select the best score, store it and set the backpointer to the
    # previous label that resulted in this score.
    if score > lattice[0][current_label].score:
        lattice[0][current_label].score = score
        lattice[0][current_label].backpointer = "-START-"
```

Here, the `label_features` function calculates the features that are based on the previous label, returning them as a set. We join this set with the set of features for the first token to calculate the score for each possible current label using the `_score` function.

```python
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
```

This scoring function is effectively the same functionality from the `predict` function of the greedy Averaged Perceptron. 

We update the score and backpointer for the lattice at element 0 for the current label.

We then move on to the main forward iteration, which does the same thing except we now calculate the score for every (previous label, current label) pair.

```python
# Forward pass, starting at t=1 because we've already initialised t=0
for t, features in enumerate(features_seq[1:], 1):
    # Extract POS tag for current feature set.
    pos = self._get_pos_from_features(features)

    # Iterate over all combinations of previous and current labels for each
    # set of features.
    # Calculate the score for each label combination and store the best score
    # plus the backpointer to the previous label that yielded that score.
    for current_label, prev_label in product(labels, labels):
        score = lattice[t - 1][prev_label].score + self._score(
            features | label_features(prev_label, pos), current_label
        )
        # Select the best score, store it and set the backpointer to the
        # previous label that resulted in this score.
        if score > lattice[t][current_label].score:
            lattice[t][current_label].score = score
            lattice[t][current_label].backpointer = prev_label
```

> [!NOTE]
>
> Many examples and tutorials of the Viterbi algorithms are given in the context of Hidden Markov Models. For these contexts, there are explicit transition probabilities (the probability of transition from one label to another) and emission probabilities (the probability of the label given the properties of the element in the sequence).
>
> The same idea still applies here, we just don't separate the two. The feature set we use to calculate the score contains both transition features (based on the previous label) and emission features (based on the properties of the element in the sequence). The inclusion of the score of the lattice element for the previous element means we capture the same probabilities just in a different way.

Note that the score calculation here also include the score of previous label currently being considered. This is because we want to maximise the score for the *transition* from previous label to current label.

#### 3. Backtrack to Find the Label Sequence

With a fully populated lattice, we can back track through to find the optimal sequence of labels.

The first step is to find the best label for the last element of the sequence.

```python
# Find the best label for the last element of the lattice, since there isn't a
# backpointer for this.
backpointer = self._argmax(labels, lattice[-1])
```

Where `_argmax` just returns the label with the highest score.

Now we can iterate backwards through lattice following the backpointers.

```python
# Iterate backwards through the lattice.
# At each step, append the backpointer that yielded the best score to the label
# sequence. Note the the resultant label sequence will be in reverse.
for score_dict in reversed(lattice):
    label_seq.append(backpointer)
    # Update backpointer to point to the label that resulted in the best score for last backpointer
    backpointer = score_dict[backpointer].backpointer
```

This gives us the optimal label sequence (but in reverse).

## Model Training

The way we train the model has to be adjusted slightly to be compatible with the Viterbi algorithm. In the greedy Averaged Perceptron, we predicted each token in turn and could update the weights immediately after each prediction as necessary. When using the Viterbi algorithm, we have to predict a whole sequence in one go, then update the weights for each incorrect token in the sequence.

The weight update stage is also slightly different. We want to do two things:

1. Decrement the weights for the features for the incorrect predicted label.
2. Increment the weights for the features for the correct label.

Importantly, unlike the greedy Averaged Perceptron weight updates, the sets of features for these two case might not be the exactly same because the features for a given token include features based on the previous label.

The weight update looks like this

```python
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
        """
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
```

`predicted_features` are the features that were used by the Viterbi algorithm and resulted in the incorrect predicted label.

`truth_features` are the features from the true sequence for the token that was incorrectly labelled. Importantly, `truth_features` contains the features based on the *true* previous label.

## Performance Comparison

Comparison of the Greedy and Viterbi Averaged Perceptron models, using the same hyperparameters.

| Model   | Word accuracy | Sentence accuracy | Model size | Time    |
| ------- | ------------- | ----------------- | ---------- | ------- |
| Greedy  | 97.57%        | 93.18%            | 0.96 MB    | 0:08.20 |
| Viterbi | 97.67%        | 94.08%            | 1.04 MB    | 1:35:57 |

Using the Viterbi algorithm does yield a notable improvement to sentence accuracy, but it comes at a significant increase in training time (12x longer).

## Limitations

The Viterbi implementation described here is a 1st-order implementation. This means that only the transition between the previous label and the current label is considered.

From the Greedy Averaged Perceptron, we know that using up to 3 previous labels provides useful information, which is not being considered by the Viterbi algorithm. Higher order implementations of the Viterbi algorithm are possible but they come at a considerable performance penalty. This 1st-order implementation is 12x slower to train - a 2nd-order implementation would likely be yet another order of magnitude slower.

## Performance Optimisation

Since the Viterbi implementation is so slow, it is worth investigating if we can apply similar optimisations to those used in the NumPy version of the Greedy Averaged Perceptron and see if they improve performance.

The docs for the [NumPy implementation](numpy.md) cover much of the detail, so we will only concern ourselves with the adaptions needed for the Viterbi algorithm.

> [!NOTE]
>
> To keep the implementation reasonable straightforward, only one feature based on the label of the previous element is used, which is the `prev_label=` feature.
>
> Other features based on previous label, such as `prev_label+pos=` are not implement because they increase the complexity.

### Weights matrices

For a NumPy implementation, we want to store the weights as NumPy arrays to maximise the benefits from using NumPy's vectorised operations. In this case, it is convenient to separate the features into two matrices: one for emission features and and one for transition features.

```python
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
```

The matrix for transition weights contains an extra row for the `-START-` label. We will never predict this label, but it can be the value of the previous label.

Using separate matrices for the emission and transition features means that we also need separate matrices for associated data during training: totals, time stamps and feature updates.

### Viterbi algorithm

The implementation of the Viterbi algorithm is fundamentally the same, but is adjusted to maximise the benefits from the use of NumPy.

For example, for each token of the sequence we can precalculate the emission score at the start of the `predict_sequence` function. This is because none of the emission features depend on the any of the labels.

```python
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
```

The lattice is implemented as two matrices: one for the scores for each label for each token, and one for the backpointers for each label for each token.

```python
# Initialize the lattice as NumPy arrays.
# One array for the scores, initialized to -inf. This is the best score for that
# label given the previous label specified by the backpointers array.
# One array for the backpointers, which hold the index of the previous label
# that resulted in the score in the lattice_scores array.
lattice_scores = np.full((seq_len, self.n_labels), -np.inf)
backpointers = np.zeros((seq_len, self.n_labels), dtype=np.int32)
```

The forward pass through the sequence of tokens is similar to above, but we have to consider the emission features and transition features separately.

```python
# Forward pass, starting at t=1 because we've already initialised t=0
for t in range(1, seq_len):
    # Get the scores for each label from the previous lattice row.
    # [:, np.newaxis] rotates this into a column vector because this is the
    # previous label to the current label, so we need to broadcast across the
    # rows of the transition matrix.
    prev_el_scores = lattice_scores[t - 1][:, np.newaxis]

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
        + self.transition_weights[: self.n_labels]
        + emission_scores[t]
    )

    # Find the best score in each column and the index of the best score in each
    # column and save to the lattice_scores and backpointers matrices
    # respectively.
    lattice_scores[t] = np.max(candidates, axis=0)
    backpointers[t] = np.argmax(candidates, axis=0)
```

Backtracking through the backpointers matrix is basically the same as above.

```python
# Find the best label for the last element of the lattice, since there isn't a
# backpointer for this.
backpointer = np.argmax(lattice_scores[-1])
# Iterate backwards through the lattice.
# At each step, append the backpointer that yielded the best score to the label
# sequence. Note the the resultant label sequence will be in reverse.
for t in range(seq_len - 1, -1, -1):
    label_seq.append(self.labels[backpointer])

    backpointer = backpointers[t, backpointer]
```

### Weight updates

Since we store the emission and transition weights separately, we need to make sure that we update both matrices when the model makes a mistake during training.

```python
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
```

The weights for the features for the true features are incremented in the same way.

### Execution performance comparison

Comparison of the Greedy and Viterbi (original and NumPy) Averaged Perceptron models, using the same hyperparameters and features.

| Model                 | Word accuracy | Sentence accuracy | Model size | Time    |
| --------------------- | ------------- | ----------------- | ---------- | ------- |
| Greedy (Pure Python)  | 97.94%        | 94.26%            | 2.09 MB    | 0:17:10 |
| Greedy (NumPy)        | 97.94%        | 94.26%            | 1.61 MB    | 0:06:48 |
| Viterbi (Pure python) | 98.01%        | 94.70%            | 2.38 MB    | 3:46:48 |
| Viterbi (Numpy)       | 98.01%        | 94.70%            | 1.83 MB    | 0:05:48 |

> Interestingly, the NumPy implementation of Viterbi Averaged Perceptron seems to be faster than the NumPy implementation of the Greedy Averaged Perceptron. I think this might be because the Viterbi version predicts an entire sentence in one go and therefore spends more time executing NumPy operations than the Greedy Averaged Perceptron, which predicts a token at a time and therefore has to context switch between Python and Numpy much more often.
