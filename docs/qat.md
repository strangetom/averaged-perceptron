# Quantization Aware Training

## Introduction

Quantization if an effective way of reducing model size and improving execution performance by reducing the precision of the wights, typically from floating point to integers or a lower precision floating point.

The docs [here](model-optimisation.md#quantization) describe the process of quantizing weights as a post training step, which shows that the averaged perceptron model weights can be quanitzed quite aggressively before the model accuracy starts to degrade significantly. The degradation of model accuracy is a result of the quantization happening without knowledge of the structure of the weights - the quantization is applied globally to all weights and eventually the quantized weights no longer resemble the weights that were trained.

Quantization Aware Training (QAT) is a technique that emulates inference-time quantization during the training process. The resolves the issues of post training quantization because the model is trained as if the weights were quantized and therefore the effect of the quantization is captured in the weight values.

## How It Works with the Averaged Perceptron

The core idea of QAT is to emulate the inference-time quantization during training such that effect of the loss of precision is accounted for during the training process. To do this we calculate the quantization scaling factor throughout the training process and use it to scale the active weights for any each prediction.
$$
scale\_factor = \frac{Q_{max}}{W_{max}} \\
Q_{max} = 2^{n-1} - 1
$$
where $n$ is the number of bits we are quantizing to and $W_{max}$ is the maximum absolute weight.

In theory, we should calculate the scaling factor prior to every prediction however this is very computationally expensive. Luckily, because the weight values don't change significantly between prediction, we can get away with recalculating it every ~250 or so sentences without impacting the final model accuracy.

Importantly, we do not apply the quantization to the full weights matrix. The full weights matrix is considered to be *latent* in the sense that we do not directly use it when making predictions. Instead, we quantize the weights for the features that are active for each prediction and calculate the score from those. When the prediction is incorrect and we need to update the model weights, we make the updates to the latent weights.

### Implementation

> [!NOTE]
>
> The implementation of QAT is based on the the NumPy Averaged Perceptron implementation. This is because the the NumPy model makes it very easy to calculate quantization scaling factor and perform the quantization.

The main modification needed to implement QAT is in the `predict()` function of the Averaged Perceptron. There are now two paths in this function:

1. The training path, which emulates the quantization during training.
2. The inference path, which uses the weights as provided.

```python
def predict(features: set[str], ...):
    # Map the features to indices
    # This returns a Numpy array of integers
    feature_indices = self._features_to_idx(features)
    
    if training_mode:
        # Training path.
        # When training we need to quantize the active latent weights to use to
        # calculate the scores.
        active_latent = self.weights[feature_indices]
        quantized_active = self._quantize_weights(active_latent)
        scores = quantized_active.sum(axis=0)
    else:
        # Inference path.
        scores = self.weights[feature_indices].sum(axis=0)
    
    # Find the label with the maximum score
    best_idx = np.argmax(scores)
    best_label = self.labels[best_idx]
    
    return best_label
```

The `quantize_weights` function is where the quantization happens.

```python
def _quantize_weights(self, weights: np.ndarray) -> np.ndarray:
    """Quantize weights to using cached quantization scale factor."""
    scaled = np.round(weights * self.quantisation_scale_factor)
    return np.clip(scaled, -self.q_max, self.q_max)
```

The quantization scale factor is updated throughout the training process by calling  `update_scale_factor()` within the training loop at appropriate intervals.

```python
def update_scale_factor(self) -> None:
    """Update the quantization scale factor based on active latent weights."""
    max_weight = np.max(np.abs(self.weights))
    if max_weight == 0:
        self.quantisation_scale_factor = 1.0
        return

    self.quantisation_scale_factor = self.q_max / max_weight
```

Once the training process is completed, the latent weights are averaged as normal and then the averaged weights are quantized. These are weights that are saved and used for inference.

```python
def quantize(self) -> None:
    """Final quantization of averaged weights to an appropriately sized integer
    type.

    This happens after weight averaging.
    """
    # Choose an appropriate type to minimise model size.
    if self.nbits <= 8:
        type_ = np.int8
    elif self.nbits <= 16:
        type_ = np.int16
    else:
        type_ = np.int32

    self.update_scale_factor()
    self.weights = self._quantize_weights(self.weights).astype(type_)
```

## Performance Comparison

Comparison of training the NumPy and QAT Averaged Perceptron models, using the same hyperparameters, at different quantization levels.

### 8 bit quantization

| Model | Word accuracy | Sentence accuracy | Model size | Time    |
| ----- | ------------- | ----------------- | ---------- | ------- |
| NumPy | 98.18%        | 95.18%            | 0.377 MB   | 0:11:22 |
| QAT   | 98.16%        | 95.18%            | 0.377 MB   | 0:11:11 |

### 6 bit quantization

| Model | Word accuracy | Sentence accuracy | Model size | Time    |
| ----- | ------------- | ----------------- | ---------- | ------- |
| NumPy | 89.13%        | 95.02%            | 0.196 MB   | 0:09:19 |
| QAT   | 98.08%        | 94.91%            | 0.195 MB   | 0:10:10 |

### 4 bit quantization

| Model | Word accuracy | Sentence accuracy | Model size | Time    |
| ----- | ------------- | ----------------- | ---------- | ------- |
| NumPy | 95.39%        | 85.89%            | 0.033 MB   | 0:10:00 |
| QAT   | 96.98%        | 91.92%            | 0.018 MB   | 0:12:11 |

### 3 bit quantization

| Model | Word accuracy | Sentence accuracy | Model size | Time    |
| ----- | ------------- | ----------------- | ---------- | ------- |
| NumPy | 78.08%        | 39.32%            | 0.005 MB   | 0:09:17 |
| QAT   | 92.92%        | 80.24%            | 0.003 MB   | 0:11:56 |

The results show that the more aggressive the quantization level, the more benefit QAT provides. At 4 bits of quantization, QAT shows a substantial improvement in sentence accuracy over post training quantization to the same level. The improvement becomes even more substantial at quantization to a smaller number of bits.

## Ternary Quantization

Ternary quantization is the process of quantizing the model weights so that they have one of 3 values: -1, 0, +1. This is an extreme case of quantization and because of the limited number of values for weights, we have to take a slightly different approach when quantizing the latent weights.

A symmetric threshold, $\alpha$,  is used to determine whether each weight is converted to -1, 0 or 1,  calculated according to[^1]
$$
\alpha = 0.75 E(|W|)
$$
where $E(|W|)$ is the mean of absolute weight values.

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

> [!TIP]
>
> The decision of how often to update the ternary threshold is somewhat arbitrary. Perhaps the *most correct* approach is to recalculate before every training sentence however this is very computationally expensive. 
>
> Since the weights do not change significantly between sentences, then neither will the threshold. Therefore we can get away with calculating it less often than for every sentence. The choice of every 250 sentences was made to have minimal impact on the training run-time whilst still updating the threshold regularly during training.
>
> Technically, how often we update the ternary threshold is a hyper parameter of the model because the choice does impact the model performance.
>
> | Update | Word accuracy | Sentence accuracy |
> | ------ | ------------- | ----------------- |
> | 1000   | 97.34%        | 92.23%            |
> | 250    | 97.48%        | 92.93%            |
> | 50     | 97.47%        | 92.90%            |
> | 10     | 97.45%        | 92.81%            |

The implementation here has to account for the decision to store the weights as a dense matrix. This means may be many rows of zeroes in the weights matrix that do not actually correspond to any features yet, so we slice the matrix to only include first $n$ rows up to the number of features in the model vocabulary.

The ternary conversion is then quite straightforward to apply
$$
W_{ij} = 
\begin{cases}
+1 & W_{ij} > \alpha \\
-1 & W_{ij} < -\alpha \\
0 & otherwise
\end{cases}
$$


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

The `update_ternary_theshold()` function is the equivalent of `update_scale_factor()` and `_ternarize_weights()` is the equivalent of `_quantize_weights()`.

### Ternary Performance Comparison

Comparison of the Numpy and Ternary Averaged Perceptron models, using the same hyperparameters.

> [!NOTE]
>
> We can't compare the Ternary model to a NumPy model that has been quantized to 2 bits of precision because the NumPy model does not work correctly at that level of precision. 
>
> Instead, we compare to the full precision NumPy model to show how (reletively) small degradation in performance.

| Model                  | Word accuracy   | Sentence accuracy | Model size         | Time    |
| ---------------------- | --------------- | ----------------- | ------------------ | ------- |
| NumPy (full precision) | 98.20%          | 95.21%            | 1.622 MB           | 0:07:37 |
| Ternary                | 97.51% (-0.70%) | 92.74% (-2.59%)   | 0.197 MB (-87.85%) | 0:10:53 |

The Ternary model compares favourably with the 4 bit QAT model above, although it is worth noting the differences in model size. The Ternary model is ~10x larger, which likely explains the better accuracy.

We might also consider if the Ternary model requires more training epochs due to the severely limited values the weights can take. The results above where trained to 20 epochs. The table below shows that more training epochs result in better model accuracy, but only up to a point.

| Model               | Word accuracy   | Sentence accuracy | Model size         | Time    |
| ------------------- | --------------- | ----------------- | ------------------ | ------- |
| Ternary (40 epochs) | 97.55% (-0.66%) | 93.07% (-2.25%)   | 0.215 MB (-86.74%) | 0:18:57 |
| Ternary (50 epochs) | 97.53% (-0.68%) | 93.06% (-2.26%)   | 0.220 MB (-86.43%) | 0:23:27 |
| Ternary (60 epochs) | 97.51% (-0.70%) | 92.74% (-2.59%)   | 0.223 MB (-86.25%) | 0:27:08 |

## References

[^1]: B. Liu, F. Li, X. Wang, B. Zhang, and J. Yan, ‘Ternary Weight Networks’, in ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Rhodes Island, Greece: IEEE, Jun. 2023, pp. 1–5. doi: 10.1109/ICASSP49357.2023.10094626.

