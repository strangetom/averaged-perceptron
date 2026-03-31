# Ternary Weights

## Introduction

Ternary weights are an extreme form of quantization where the weights only have 3 possible values: -1, 0, 1. The docs [here](model-optimisation.md#Quantization) describe the process of quantizing weights as a post training step, which shows that the averaged perceptron model weights can be quanitzed quite aggressively before the model accuracy starts to degrade significantly, however there is a limit.

To end up with ternary weighs, the model needs to be trained in a way that accounts for the quantization during the training - too much information would be lost if quantization was applied post training step.

## Quantization Aware Training

Quantization Aware Training (QAT) is a technique that emulates inference-time quantization during the training process. To do this, a set of full precision "latent" weights are updated whenever the model makes an error but whenever the model is being used to predict the labels for a sequence of tokens, the relevant latent weights are quantized for the scoring calculation.

The advantage of this approach is that the weights are updated to account for the limited values the weights can take and therefore (at least in theory), the resulting model should perform better than a model quantized post training to a similar number of bits.

> [!NOTE]
>
> The implementation of QAT for training a ternary model are based on the the NumPy Averaged Perceptron implementation. This is because the ternary conversion is needed at every call to the `predict()` function which slows the training process down.
>
> Since the NumPy implementation is faster than the pure Python implementation, we use it to try to mitigate some of the expected slow down.

### Ternary conversion

Ternary conversion is the process of turning the latent weights (which are at full precision) into ternary weights which can have values -1, 0 or 1.

A symmetric threshold, $\alpha$,  is used to determine whether each weight is converted to -1, 0 or 1,  calculated according to[^1]
$$
\alpha = 0.75 E(|W|)
$$
where $E(|W|)$ is the mean of absolute weight values.

This threshold is recalculated before every training sentence according to

```python
def update_ternary_threshold(self) -> None:
    """Update the cached ternary threshold based on active latent weights.
	"""
	active_weights = self.weights[: self.next_feature_index]
    if active_weights.size == 0:
        self.ternary_threshold = 0.0
        return

    # Filter for non-zero weights.
    # Since a Perceptron is only interested in non-zero weights we need to remove
    # the zeroes from this calculation that arise from the choice to store the
    # weights as a dense matrix.
    abs_weights = np.abs(active_weights)
    non_zero_weights = abs_weights[abs_weights > 0]

    if non_zero_weights.size == 0:
        self.ternary_threshold = 0.0
    else:
        # Apply the 0.75 heuristic to the mean of non-zero active weights
        self.ternary_threshold = float(0.75 * np.mean(non_zero_weights))
```

The implementation here has to account for the decision to store the weights as a dense matrix. This means may be many rows of zeroes in the weights matrix that do not actually correspond to any features yet, so we slice the matrix to only include first $n$ rows up to the number of features in the model vocabulary.

The ternary conversion is then quite straightforward:

```python
def _ternarize_weights(self, weights: np.ndarray) -> np.ndarray:
    """Convert weights to ternary using ternary threshold.

    The given threshold is used symmetrically in the conversion:
    weight > threshold: 1
    weight < -threshold: -1
    otherwise: 0
    """
    return np.where(
        weights < -self.ternary_threshold,
        -1,
        np.where(weights > self.ternary_threshold, 1, 0),
    )
```

### Emulating the quantization

The key part of QAT is emulating the inference-time quantization during training. To do this we follow these steps in the `predict()` function:

1. Select the latent weights for the active features.
2. Convert those to ternary using the threshold, $\alpha$, calculated above.
3. Calculate the scores based on the ternary weights.

```python
def predict(self, features: set[str], ...) -> tuple[str, float]:
	"""Predict the label for a token described by features set.
	"""
    # Determine the active features.
    feature_indices = self._features_to_idx(features)
	if self.training_mode:
        # Sum weights for active features across all labels at once
        # Shape: (n_labels,) i.e. the summed score for each label.
        if len(feature_indices) > 0:
            # When training we need to convert the active latent weights to ternary
            # to use to calculate the scores.
            active_latent = self.weights[feature_indices]
            ternary_active = self._ternarize_weights(active_latent)
            scores = ternary_active.sum(axis=0)
        else:
            scores = np.zeros(self.n_labels)
```

### Weight simplification

The NumPy implementation of the Averaged Perceptron has a weight simplification step at the end of the training process, after the weights have been averaged. This is a result of the decision to store the weights as a dense matrix where we double the size whenever the number of feature exceeds the current number of rows. The weight simplification step discards any rows that are all zeros.

For this implementation, we have to do this twice. The first time is to remove any pre-allocated rows that are unused. Then we convert the averaged weights to ternary. Then we call `simplify_weights()` a second time to remove any rows that the ternary conversion process has zeroed out.

## Performance Comparison

Comparison of the Numpy and Ternary Averaged Perceptron models, using the same hyperparameters.

| Model                  | Word accuracy   | Sentence accuracy | Model size         | Time    |
| ---------------------- | --------------- | ----------------- | ------------------ | ------- |
| NumPy (full precision) | 98.19%          | 95.18%            | 0.868 MB           | 0:07:07 |
| Ternary                | 97.10% (-1.11%) | 92.11% (-3.23%)   | 0.177 MB (-79.61%) | 2:21:37 |

We can also compare the performance against the NumPy after post training quantization.

| Model                      | Word accuracy    | Sentence accuracy | Model size         | Time    |
| -------------------------- | ---------------- | ----------------- | ------------------ | ------- |
| Numpy (4 bit quantization) | 95.61% (-2.63%)  | 87.02% (-8.57%)   | 0.036 MB (-95.85%) | 0:07:04 |
| Numpy (3 bit quantization) | 80.76% (-17.75%) | 45.58% (-52.11%)  | 0.006 MB (-99.31%) | 0:07:03 |

The effect of QAT is clearly shown here. The post training quantization results in significantly worse accuracy, even when using more bits of precision for the weights.

> [!NOTE]
>
> Note the size of the post training quantized models compared to the Ternary model. The Ternary model is significantly larger. We can infer from this (given the limited options for the values of the weights) that there are more active features in the model. This is a result of the QAT inducing errors during training which result in more updates to feature weights.

We might also consider if the Ternary model requires more training epochs due to the severely limited values the weights can take. The table below shows that more training epochs result in better model accuracy, but only up to a point.

| Model               | Word accuracy   | Sentence accuracy | Model size         | Time    |
| ------------------- | --------------- | ----------------- | ------------------ | ------- |
| Ternary (40 epochs) | 97.57% (-0.63%) | 92.95% (-2.34%)   | 0.195 MB (-77.53%) | 4:12:39 |
| Ternary (50 epochs) | 97.63% (-0.57%) | 93.41% (-1.86%)   | 0.201 MB (-76.84%) | 4:59:45 |
| Ternary (60 epochs) | 97.51% (-0.69%) | 93.10% (-2.19%)   | 0.205 MB (-76.38%) | 5:42:33 |



## References

[^1]: B. Liu, F. Li, X. Wang, B. Zhang, and J. Yan, ‘Ternary Weight Networks’, in ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Rhodes Island, Greece: IEEE, Jun. 2023, pp. 1–5. doi: 10.1109/ICASSP49357.2023.10094626.

