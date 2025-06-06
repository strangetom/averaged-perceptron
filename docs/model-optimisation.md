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

In this example, we can reduce the model size by 32% with a negligible impact on model performance. 

If model size was particularly important, we could go even further and reduce the model size by 80% with less than 1% reduction in sentence level accuracy.

## Quantisation

### Performance comparison



