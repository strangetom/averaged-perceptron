# Easiest First Algorithm

## Introduction

A limitation of the greedy implementation of the Averaged Perceptron model is that always assigns label in a left to right direction. The assigned label for tokens to the left of a token under consideration provide useful information in helping determine the likeliest label for the current token. What if we could also use the labels of tokens to the right of the current token to help assign a label to the current token?

The "easiest-first" algorithm allows this by changing the order in which labels are assigned. Instead of assigning label in a left to right direction, labels are assigned based on the easiest label to assign to an unlabelled token. In this algorithm, easiest refers to the highest score across all labels across all unlabelled tokens. This means that when considering each token there may be tokens in the surrounding context window that already have labels assigned, which can help in determining the label for the current token.

## How It Works with the Averaged Perceptron

### Algorithm Outline

The basic algorithm in fairly straightforward. Starting with a completely unlabelled sequence of tokens, iteration over each token. For each token, iterate over all possible labels and calculate the score for each label based on the features of that token and the labels of surrounding tokens (which may be unassigned).

Determine the token-label pair that has the highest overall score and assign that label to that token. Repeat this process for the remaining unlabelled tokens, assigning the highest scoring label to the token for each iteration. Continue iterating this process until all tokens have been assigned labels.

=== **INSERT IMAGE HERE** ===

The key insight here is that sometimes the labels of the surrounding contextual token do not matter that much when assigning a label. The inherent features of that token alone are enough to make a confident assignment. As more tokens are labelled in the sequence, then future label assignments benefit from those easily assign labels.

### Implementation

Prediction of labels for a sequence is done for the sequence as a whole. This in contrast with the greedy Averaged Perceptron which predicts the label for each token one at a time.

First we create the structures for storing the results

```python
# The assigned labels as a dict where the token index is the key
# and the assigned label is the value.
current_labels: dict[int, str] = {}
# The confidence of the assigned label where the token index is the key
# and the confidence is the value.
label_confidence: dict[int, float] = {}
```

We also define a `set` containing the indices of tokens that have not had labels assigned.

```python
# The set of token indices that have not had labels assigned.
indices_to_label = set(range(len(features_seq)))
```

Whilst there are still indices to labels, we iterate over all combinations of unassigned index and labels to determine the highest scoring index-label pair.

```python
while indices_to_label:
    # Default values for best score and best candidate label.
    best_score = -float("inf")
    best_candidate = (-1, "_UNASSIGNED_")  # (index, label)
    # Scores is a dict for scores for each label for each index.
    # {
    #	0: {
    #		"QTY": 15.1354,
    #		"UNIT": -2.486,
    #		...
    #		},
    #	1: {...},
    #	...
    # }
    scores: dict[int, dict[str, float]] = defaultdict(dict)
    
    # Iterate over all index-pair combinations to determine best score.
	...  # (see below)

    idx, label = best_candidate
    # Assign the label for the best candidate.
    current_labels[idx] = label
	# Determine confidence for the assigned label
    label_confidence[idx] = self._confidence(scores[idx])[label]
    # Now a label has been assigned to this index, remove from set of indices to label.
    indices_to_label.remove(idx)
```

The main iteration loop is as follows.

```python
for i, label in product(indices_to_label, sorted_labels):
    # Determine the complete set of features for the current token at position i.
    stem = token_stems[i]
    pos = token_pos[i]
    features = features_seq[i] | self.label_features(
        current_labels, i, stem, pos, label, len(features_seq)
    )
    # Score the features for the label under consideration and save the score
    # so that the confidence of the assigned can be determined.
    score = self.score(features, label)
    scores[i][label] = score
    # If this score exceeds the current best score, set this index-label pair as the best.
    if score > best_score:
        best_score = score
        best_candidate = (i, label)
```

> [!NOTE]
>
> Note that the labels are sorted in alphabetical order. This is ensure that ties between scores of labels for the same token are resolved in a consistent manner because the labels are always iterated over in the same order.

The scoring calculation is the standard scoring calculation for the Averaged Perceptron: sum the weights of each feature for the given label.

```python
def score(self, features: set[str], current_label: str) -> int:
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

The calculation of the complete feature set for each token starts with the emission features for the token under consideration. These are the features based solely on properties of the token itself and it's surround context. To those emission features, we then add features based on the labels of the surrounding context tokens.

````python
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
	
    # Determine the label for the previous and next tokens.
    # If a label hasn't been assigned, use "_UNASSIGNED_".
    # If the token is at the start or end of the sequence, 
    # use "_START_" or "_END_" to indicate this.
    prev_label = get_context_label(index - 1)
    next_label = get_context_label(index + 1)
	
    # Create the features.
    return {
		"prev_label=" + prev_label,
        "prev_label+pos=" + prev_label + "+" + token_pos,
        "prev_label+stem=" + prev_label + "+" + token_stem,
        "next_label=" + next_label,
        "next_label+pos=" + next_label + "+" + token_pos,
        "next_label+stem=" + next_label + "+" + token_stem,
    }
````

We can include label features based a context window of arbitrary size. The code above assumed we are only looking at one token either side of the token under consideration, but this can be easily expanded to include more tokens.

## Model Training

The model training process looks very similar to the greedy Averaged Perceptron implementation, with some additional complexity due to the easiest-first approach. The prediction of the label for each token is done identically to the `predict()` function described above. However once a label has been assigned to a token we need to check it against the true label and update the weights if it was wrong.

```python
while indices_to_label:
    best_score = -float("inf")
    best_candidate = (-1, "_UNASSIGNED_")

    for i, label in product(indices_to_label, sorted_labels):
        stem = stem_seq[i]
        pos = pos_seq[i]
        features = features_seq[i] | self.model.label_features(
            current_labels, i, stem, pos, label, len(features_seq)
        )
        score = self.model.score(features, label)
        if score > best_score:
            best_score = score
            best_candidate = (i, label)

    pred_idx, predict_label = best_candidate
    true_label = truth_labels[pred_idx]
    stem = stem_seq[pred_idx]
    pos = pos_seq[pred_idx]
    # Update weights.
    self.model.update(
        true_label,
        predict_label,
        # The features for the predicted label.
        features_seq[pred_idx]
        | self.model.label_features(
            current_labels,
            pred_idx,
            stem,
            pos,
            predict_label,
            len(features_seq),
        ),
        # The features for the true label.
        features_seq[pred_idx]
        | self.model.label_features(
            current_labels,
            pred_idx,
            stem,
            pos,
            true_label,
            len(features_seq),
        ),
    )
```

The `model.update()` function looks like the following:

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

    Returns
    -------
    None
    """
	# Do nothing is the prediction was correct.
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

## Performance Comparison

Comparison of the Greedy and Easiest-first Averaged Perceptron models, using the same hyperparameters.

| Model         | Word accuracy | Sentence accuracy | Model size | Time    |
| ------------- | ------------- | ----------------- | ---------- | ------- |
| Greedy        | 98.15%        | 94.74%            | 0.354 MB   | 0:16:22 |
| Easiest-first | 97.80%        | 94.60%            | 0.299 MB   | 2:34:50 |

> [!IMPORTANT]
>
> It is difficult to directly compare the word and sentence accuracies because the two models have different features: the greedy model can only use label-based features for tokens to the left in the sequence.
>
> In the comparison here, the label-based features were generated following the same principals, however the "easiest-first" model also generated those features for tokens to the right of the token under consideration.

The "easiest-first" Averaged Perceptron takes significantly longer (~9.5x) to train because it has to perform $\mathcal{O}(N^2*L)$ calculations to assign the labels for a complete sequence (where N is the number of tokens and L is the number of labels). This is in contrast to the Greedy Averaged Perceptron, which only requires $\mathcal{O}(N*L)$.

The differences in word and sentence accuracy may be explained by a lack of optimisation. Both models were trained with similar features and the same hyperparameters, which may not be optimal for the "easiest-first" Averaged Perceptron given the differences in how the label-based features are generated and used. Note that the "easiest-first" Averaged Perceptron does not implement label transition constraints, so these were also disabled for the Greedy Averaged Perceptron.
