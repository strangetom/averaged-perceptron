# Averaged Perceptron

## Introduction

...[^1]

## Implementation

### Features

In general, the best information we have can have for predicted the label for a token is the token itself and the surrounding tokens (i.e. it's context). Unfortunately it is on practical is create an exhaustive database of all possible tokens and their context. Instead we extract various bits of information from each token and it's context and use these to help is predict the correct label. Each of bits of information (*"features"*) will naturally provide less information than the token itself, but the combination of the features will help us predict the label, especially for tokens we haven't seen before.

The most obvious feature is the token itself. Other useful features can be things such as 

* The token's stem
* The token's part of speech
* The token's prefix and suffix
* Whether the token is capitalised
* Whether the token is inside parentheses
* Whether the token occurs after a comma

and so on. Additionally, all these features can be calculated for the context tokens either side of a token and we can make use of them as features for the token. For example

```pytho
{
	"pos": "NN":
	"next_pos": "NN"
	"prev_pos": "CD"
}
```

are all features of the current token.

The important things about these features is that we are able to calculate them for any token in an arbitrary sentence.

Additional helpful features, which should also be quite obvious, are the labels of the context tokens. In practice we can only use the labels of tokens earlier in the sequence of tokens that make the ingredient sentence because we haven't predicted the labels for tokens further ahead.

The features of a token will be stored as a `set` of `str`. As an example, for the token "cups" in the sentence "3 cups milk", some of the feature would be 

```python
{
    "bias="
    "stem=cup",
    "pos=NN",
    "prev_label=QTY",
    "is_after_comma=False",
    "is_in_parens=False",
    ...
}
```

> [!NOTE]
>
> The data used to train this model has features in a `dict` of `{feature_name: value}`. We need to convert these into strings of the form `features_name=value` so we can the strings as the keys for the weights `dict`.

### Prediction

Given a set of features for a token, we can predict it's label by summing the weight for each label across each of the features for the token.

```py
def predict(self, features: set[str]) -> str:
    """Predict label from features.
    
    Parameters
    ----------
    features : set[str]
        Set of features for token.

    Returns
    -------
    str
    	Predcited label for token.
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
```

> [!NOTE]
>
> After we call `predict` for each token, we need to update the features for the next token to include features using the label we just predicted.

The `weights` data structure is a `dict` of `dict`s. The outer `dict` maps features to a `dict` of weights for each label for that feature:

```python
self.weights = {
    "feature1=value": {
        "QTY": 123
        "UNIT": 135, 
        ...
    },
    "feature2=value": {
        "COMMENT": -25,
        "PREP": 134, 
        ...
    },
    ...
}
```

We only include non-zero weights in the `weights` data structure. If the feature is present in the `dict`, it means all label weights for that feature are zero. If the feature is present, only labels with non-zero weights are stored.

The `weights` structure is organised like because the weights are sparse. For any given token, most of the features are not relevant.

### Training

Lets train the model to learn the weights so we can predict the label for a sequence of tokens.

The Averaged Perceptron model is error driven. This means that all weights start of as 0 and we only modify the weights whenever the model makes a mistake. When we first start training, we can naturally expect a lots of mistakes and a lot of updates. As the training progress, the rate of weight updates will decrease.

The Averaged Perceptron is an example of supervised learning. To train this model we need lots of example sentences where we know what the correct label for each token in each sequence is. This is how we will know when the model makes a mistake.

The basic steps to train the model are:

1. Iterate over each set of features and true label.
2. Predict the label given the features.
3. If the prediction is wrong, increment the weights for the true label for each of the features and decrement the weight for the incorrectly predicted label for each of the features.

There are three nested loops we iterate over when training.

```python
for epoch in range(n_epochs):
    for sentence_features, true_sentence_labels in training_sentences:
        # Set initial prev_label for start of sentence
        prev_label = "-START-"
        for features, true_label in zip(sentence_features, true_sentence_labels):
            # Update features to include previous label features
            updated_features = self.update_features(features, prev_label)
            prediction = self.predict(updated_features)
            if prediction != true_label:
                for f in features:
                    self.weights[f][true_label] += 1
                    self.weights[f][prediction] -= 1
                    
	random.shuffle(training_sentences)
```

Since we update the weights after each prediction, the model will quickly tend towards the optimal weights.

> [!NOTE]
>
> The Averaged Perceptron model only has a single hyper-parameter that can be tuned during training: `n_epochs`. This is the number of times we iterate over all the training data, shuffling the order between each training epoch. 

### Averaging

If we use the training as described so far, we end up with a model that is sensitive to the order in which the training examples are presented to the model. Even the shuffling between each epoch doesn't eliminate this problem. The result is that if we were to train the model twice with different shuffling, we would get two very different models.

We need a way to capture how the weights change over the entire training process. That way, the training sentences towards the end of the training process don't end up affecting the final weights more than the sentences at the beginning of the process. The solution to this is to average the weights over every update.

> [!IMPORTANT]
>
> The weights for each label for each feature are averaged over their values across all inner loop iterations, not just epochs. 
>
> If we train for 10 epochs, with 20 sentences, each sentence having 5 tokens, then we need to average over 10 x 20 x 5 = 1000 updates.

The most simplistic way to do the averaging would be to keep track of the accumulated the weights for each feature at each inner loop iteration, then divide by the total number of iterations. This would extremely inefficient as we would have to update the accumulated values for *every* feature even if we don't modify it's weights.

A more efficient approach is to only update the accumulated values whenever the weight changes. To do this we keep track of when the weights for a feature last changed and when it changes again, we can calculate the accumulated weight for all that time it didn't change.

```python
def update(self, truth: str, guess: str, features: set[str]):
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
```

The `_update_feature` function is where the interesting bit happens.

```python
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
```

`self._totals` keeps track of the accumulated value for each (feature, label) pair that gets updated (we don't bother for features who's weights never change). 

When the weight of a label for a feature changes, we calculate the accumulated weight since the last change and add that to the value in `self._totals`, then we update the timestamp (`self._tstamps`) to the current iteration and modify the weight; incrementing or decrementing as appropriate.

Once the training process has finished, we can update all the totals with the accumulated weights for the total number of iterations, then divide by the number of iterations to get the averaged weight.

```python
def average_weights(self):
    """Average the value for each weight over all updates.
    """
    for feat, weights in self.weights.items():
        avg_feat_weights = {}

        for label, weight in weights.items():
            key = (feat, label)
            # Update total to account for iterations since last update
            total = (
                self._totals[key] + (self._iteration - self._tstamps[key]) * weight
            )
            averaged = round(total / float(self._iteration), 3)

            if averaged:
                avg_feat_weights[label] = averaged

        self.weights[feat] = avg_feat_weights
```

### Calculating probabilities

When predicted the label for a token we can calculate the confidence in the label from the scores we calculate from the weight. We can treat each score as a log probability and normalise [^2].

In non-log terms
$$
p_i = \frac{\exp{(s_i)}}{\sum^N_{i=1}\exp{(s_i)}}
$$
Where $p_i$ is the probability of label $i$, $s_i$ is the score for label $i$ and $N$ is the number of labels.

Moving into log-space
$$
\log{(p_i)} = \log{(\exp{(s_i)})} - \log{\sum^N_{i=1}\exp{(s_i)}} \\
\log{(p_i)} = s_i - \log{\sum^N_{i=1}\exp{(s_i)}}
$$
If the model is very confident in the chosen label, meaning the score for the selected label is much higher than the scores for the other label, then the sum in second term becomes dominated by the largest values. Therefore we can simplify this further
$$
\log{(p_i)} = s_i - \max{(s)}
$$

### Pruning

## References

[^1]: https://explosion.ai/blog/part-of-speech-pos-tagger-in-python
[^2]: https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/



