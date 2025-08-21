# Model optimisation

## Introduction

Here we discuss different techniques for optimising the model for size or execution performance.

We will assume that the method of storing the weights is to store them in a gzipped JSON file.

```python
def save(self, path: str, compress: bool = True) -> None:
    """Save trained model to given path.

    Parameters
    ----------
    path : str
        Path to save model weights to.
    compress : bool, optional
        If True, compress .json file using gzip.
        Default is True.
    """
    data = {
        "labels": list(self.model.labels),
        "weights": self.model.weights,
        "labeldict": self.labeldict,
    }
    if compress:
        if not path.endswith(".gz"):
            path = path + ".gz"

        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump(data, f)
    else:
        with open(path, "w") as f:
            # The seperator argument removes spaces from the normal defaults
            json.dump(data, f, separators=(",", ":"))
```

## Pruning

Weight pruning is the process of removing weights with values below a set threshold. There are a couple of advantages to doing this:

* Reducing the model size, and therefore improving execution speed.
* Making the model more general by reducing over fitting to the training data that can result in weights with very small values.

The objective of the pruning process is to find the balance between reduction in model size and the decrease in model accuracy that eventually results from too much pruning.

### Performance comparison

The figure in brackets in the accuracy columns shows the change from the baseline results when there is no pruning.

The maximum weight  for the model trained for these examples was 110.43.

| Minimum absolute weight | Word accuracy   | Sentence accuracy | Model size              |
| ----------------------- | --------------- | ----------------- | ----------------------- |
| 0                       | 97.94% (+0%)    | 94.80% (+0%)      | 1,007,206 bytes (+0%)   |
| 0.25                    | 97.94% (+0%)    | 94.80% (+0%)      | 963,583 bytes (-4.33%)  |
| 0.5                     | 97.94% (+0%)    | 94.78% (-0.02%)   | 908,809 bytes (-9.77%)  |
| 1                       | 97.96% (+0.02%) | 94.80% (+0%)      | 684,140 bytes (-32.08%) |
| 5                       | 97.72% (-0.22%) | 93.95% (-0.90%)   | 201,561 bytes (-79.99%) |
| 10                      | 93.66% (-4.37%) | 83.20% (-12.24%)  | 59,343 bytes (-94.11%)  |

In this example, we can reduce the model size by 32% with a no impact on model performance. 

If model size was particularly important, we could go even further and reduce the model size by 80% with less than 1% reduction in sentence level accuracy.

## Quantization

Quantization is a technique to reduce the computational and memory cost of running inference using a model by representing the weights with lower precision data types.

The quantization technique described here is symmetric linear quantization. The weights are scaled linearly relative to the largest absolute weight value, which is mapped to the largest value representable by the chosen lower precision data type i.e.
$$
[-w_{max}, w_{max}] \rightarrow [-q_{max}, q_{max}]
$$
The quantization is done such that an original weight of zero is mapped a quantized weight of zero. This is particularly important for the Averaged Perceptron model, where the majority of the model weights are zero.

Quantization can be performed to an arbitrary level of precision although.

```python
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
```

> [!NOTE]
>
> For some models, storing the scale factor is necessary for use during inference, to rescale the weights back to the original scale (with some loss of precision due to the quantization).
>
> For the Averaged Perceptron, this is not necessary. The Averaged Perceptron uses the weight additively and compares the relative magnitude. We could apply the scale factor to restore the original scale, but since this is multiplicative and therefore would scale all weights equally the relative magnitudes would not change. Therefore, there is no need to store the scale factor.

### Performance comparison

All these results were obtained without any weight pruning. If weight pruning was applied, it would be done before the quantization.

| Quantization bits | Word accuracy   | Sentence accuracy | Model size              |
| ----------------- | --------------- | ----------------- | ----------------------- |
| None              | 97.88% (+0%)    | 94.49% (+0%)      | 970,266 bytes (+0%)     |
| 16                | 97.88% (+0%)    | 94.49% (+0%)      | 867,683 bytes (-10.57%) |
| 8                 | 97.87% (-0.01%) | 94.45% (-0.04%)   | 575,683 bytes (-40.67%) |
| 7                 | 97.85% (-0.03%) | 94.48% (-0.01%)   | 477,485 bytes (-50.79%) |
| 6                 | 97.73% (-0.15%) | 94.13% (-0.38%)   | 400,973 bytes (-58.67%) |
| 5                 | 97.07% (-0.83%) | 92.34% (-2.28%)   | 335,554 bytes (-65.42%) |
| 4                 | 95.07% (-2.87%) | 85.73% (-9.27%)   | 292,533 bytes (-69.85%) |

In this example, we can quantize to 7 bits to reduce the model size by 50% with a negligible impact on model performance. 
