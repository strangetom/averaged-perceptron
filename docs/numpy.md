# Execution Performance Optimisation

## Introduction

Baseline implementation to was written to be easily readable and without having any dependencies.

But it is basically a lot of maths.

Therefore, using [Numpy](https://numpy.org/) should allow the execution performance to be optimised.

## Changes to the data structures

In the greedy implementation of the Averaged Perceptron, the weights were stored as a `dict` of `dict`s, where both levels of `dict` only had the elements need to represent non-zero weights.

```python
weights = {
    "feature_1": {
        "label_1": ...,
        "label_n": ...,
        ...
    },
    "feature_n": {...},
    ...
}
```

This is efficient from a memory perspective because it only stores non-zero values. However it is less efficient from a calculation perspective because it requires a lot of look ups and we have to be sure to handle the cases where a key is not present.

In the Numpy implementation, the weights are stored as a 2D numpy array, initialised to zeros, where the rows correspond to features and the columns correspond to labels.

$$
weights = \begin{bmatrix}
 W^{f1}_{QTY} & W^{f1}_{UNIT} & W^{f1}_{SIZE} &  \cdots\\
 W^{f2}_{QTY} & W^{f2}_{UNIT} & W^{f2}_{SIZE} &  \cdots\\
 W^{f3}_{QTY} & W^{f3}_{UNIT} & W^{f3}_{SIZE} &  \cdots\\
 \vdots & \vdots & \vdots & \ddots
\end{bmatrix}
$$
In addition, we also need `dict` that map from feature to row index and from label to column index.

The other data structures used during training are similarly converted to Numpy arrays using the same row and column indexing.

### Resizing arrays

Due to the weights matrix being a dense matrix of zeros, we need to know the number of features and labels to be able initialise the matrices to the correct sizes. The labels are known ahead of time, but the features are not.

To solve this problem in a way that does not involve resizing the matrices every time we encounter a new feature during training, the matrices are initialised assuming some number of features. Once we reach that number of features, we resize the matrices by doubling the number of rows. If the number of features exceeds this new size, we resize again by doubling the rows. This geometric progression of resizing limits the number of resize operations that are needed.

> [!IMPORTANT]
>
> This approach will result in a weights matrix with more rows than the number of features.
>
> Once the training is complete, a matrix simplification is performed to remove all rows that only contains zeroes, and their associated features.

Note that this matrix resizing is only required during training. If we encounter a new feature during inference with a pre-trained model then it is ignored. This is the equivalent of treating it as if the weights for each label for that feature were all zeroes.

## Vectorised calculations

The main advantage of using Numpy is the ability to perform vectorised calculations. Many of the function in the greedy implementation become simpler to implement because we avoid having to write many of the loops.

### `.predict(...)`

```python
def predict(features: set[str], ...):
    # Map the features to indices
    # This returns a Numpy array of integers
    feature_indices = self._features_to_idx(features)
    
    # Sum the scores for each label across the features
    # This returns an array of floats
    scores = self.weights[feature_indices].sum(axis=0)
    
    # Find the label with the maximum score
    best_idx = np.argmax(scores)
    best_label = self.labels[best_idx]
```

### `.average_weights(...)`

```python
def average_weights(...):
    # Final pass to bring all totals up to date
    iters = self._iteration - self._tstamps
    self._totals += iters * self.weights

    # Average weights
    self.weights = self._totals / self._iteration
```

### `.filter_features(...)`

```python
def filter_features(...):
    # Find features that have been updated fewer than `min_feat_updates` times
    idx_to_zero = np.argwhere(self._feat_updates < self.min_feat_updates)
    # Set all the weights for those features to 0
    self.weights[idx_to_zero, :] = 0
```

## Saving the model

The use of a Numpy array to store the weights means that it's no longer practical to save the weights as JSON. There are 3 things we need to save to be able reload the model and use it again:

1. The list of labels, in the same order as the weights matrix columns
2. The list of features, in the same order as the weights matrix rows
3. The weights matrix

To keep all these files together, all these data are saved to a `.tar.gz` archive. The labels and features are saved a JSON files. The weights are saved as an .`npy` file.

## Performance comparison

Comparison of the Greedy and Numpy Averaged Perceptron models, using the same hyperparameters.

#### Training

| Model  | Model size | Time              |
| ------ | ---------- | ----------------- |
| Greedy | 0.388 MB   | 0:17:33 (+0%)     |
| Numpy  | 0.324 MB   | 0:06:38 (-62.20%) |
