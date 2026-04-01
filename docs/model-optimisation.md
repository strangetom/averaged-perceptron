# Model Optimisation

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

## Weight Pruning

Weight pruning is the process of removing weights with values below a set threshold. There are a couple of advantages to doing this:

* Reducing the model size, and therefore improving execution speed.
* Making the model more general by reducing overfitting to the training data that can result in weights with very small values.

The objective of the pruning process is to find the balance between reduction in model size and the decrease in model accuracy that eventually results from too much pruning.

### Performance Comparison

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

## Feature Pruning

> [!NOTE]
>
> There are two types of feature pruning: 
>
> * Pre-training, where features that occur less than a threshold number of times are ignored.
> * Post-training, where features that have been updated less than a threshold number of times are ignored.
>
> We are discussing the latter here, as described in 
>
> > Goldberg, Yoav, and Michael Elhadad. "Learning sparser perceptron models." *Acl*. MIT Press, 2011.

Post-training feature pruning keeps track of the number of updates for each feature during the training process. While the number of updates remains below a set threshold, the feature is ignored when predicting the label for a token. Updates are made to the feature as usual and, once the number of updates exceeds the threshold, the feature is considered as usual and it comes with its update history.

A filtering step is necessary after training to remove feature whose update count is below the threshold from the weights. Without this step, those features will be saved. When loading a previously saved model, the model will not have the update counts and so will consider all the features loaded from the file.

```python
def filter_features(self) -> None:
    """Filter features from weights dict if they were updated less than than
    min_feat_updates.

    If min_feat_updates is 0, do nothing.

    Returns
    -------
    None
    """
    if self.min_feat_updates == 0:
        # Nothing to filter
        return None

    for feature in list(self.weights.keys()):
        if self._feature_updates.get(feature, 0) < self.min_feat_updates:
            del self.weights[feature]
```



### Performance Comparison

The figure in brackets in the accuracy columns shows the change from the baseline results when there is no pruning.

| Minimum update count | Word accuracy   | Sentence accuracy | Model size              |
| -------------------- | --------------- | ----------------- | ----------------------- |
| 0                    | 98.13% (+0%)    | 95.08% (+0%)      | 866,146 bytes (+0%)     |
| 5                    | 98.09% (-0.04%) | 95.09% (+0.01%)   | 719,536 bytes (-16.93%) |
| 10                   | 98.09% (-0.04%) | 95.07% (-0.01%)   | 635,318 bytes (-26.65%) |
| 20                   | 98.10% (-0.03%) | 95.03% (-0.05%)   | 514,431 bytes (-40.61%) |
| 50                   | 98.08% (-0.05%) | 94.86% (-0.23%)   | 358,718 bytes (-58.58%) |
| 100                  | 97.98% (-0.15%) | 94.57% (-0.54%)   | 268,904 bytes (-68.95%) |
| 500                  | 97.34% (-0.81%) | 92.14% (-3.09%)   | 124,493 bytes (-85.63%) |

## Quantization

Quantization is a technique to reduce the computational and memory cost of running inference using a model by representing the weights with lower precision data types.

The quantization technique described here is symmetric linear quantization. The weights are scaled linearly relative to the largest absolute weight value, which is mapped to the largest value representable by the chosen lower precision datatype i.e.

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

### Performance Comparison

All these results were obtained without any model optimisations. If weight or feature pruning was applied, it would be done before the quantization.

| Quantization bits | Word accuracy    | Sentence accuracy | Model size              |
| ----------------- | ---------------- | ----------------- | ----------------------- |
| None              | 98.20% (+0%)     | 95.21% (+0%)      | 2,207,420 bytes (+0%)   |
| 16                | 98.20% (+0%)     | 95.21% (+0%)      | 939,016 bytes (-50.46%) |
| 8                 | 98.18% (-0.02%)  | 95.18% (-0.03%)   | 601,044 bytes (-72.77%) |
| 7                 | 98.19% (-0.01%)  | 95.21% (+0%)      | 433,462 bytes (-80.36%) |
| 6                 | 98.13% (-0.07%)  | 95.02% (-0.20%)   | 288,526 bytes (-86.93%) |
| 5                 | 97.40% (-0.81%)  | 93.25% (-2.06%)   | 149,517 bytes (-93-22%) |
| 4                 | 95.39% (-2.86%)  | 85.89% (-9.79%)   | 43,559 bytes (-98.03%)  |
| 3                 | 78.08% (-20.49%) | 39.32% (-58.70%)  | 6,427 bytes (-99.71%)   |

In this example, we can quantize to 7 bits to reduce the model size by 80% with a negligible impact on model performance. 
