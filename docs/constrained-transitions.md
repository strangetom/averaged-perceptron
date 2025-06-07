# Constrained Transitions

## Introduction

The data used to train the Averaged Perceptron model provides the model with examples which the learned weights should ultimately be a representation of. The training data, however, does not necessarily represent the totality of the information we know about the problem we are modelling.

An example of this is whether any label transitions are forbidden. If there are forbidden labels transitions, then they will not appear in the training or evaluation data, but this does not mean the model will consider them forbidden. In general, the learned weights will make the transition very unlikely, but still allowable.

If the labelling scheme being used does forbid certain transitions, then we can apply these rules to ensure the model does predict a label that is forbidden in the context.

> [!IMPORTANT]
>
> We should not apply the constraints when training the model.
>
> Applying label transition constraints during training will prevent the model making prediction errors that would normally update the weights and improve the model's overall performance.

## Analytical constraints

Analytical constraints are constraints on label transitions that arise as a result of how the labelling scheme has been designed.

For example, the label I_NAME_TOK is used to label tokens in an ingredient name, but not the first token of an ingredient name. Therefore, we can constrain the model from predicting I_NAME_TOK if B_NAME_TOK has not already been predicted in the sequence. Note that B_NAME_TOK and I_NAME_TOK do not have to be consecutive tokens.

The NAME_SEP label is used to separate ingredient names, so the above constraint should be modified to prevent I_NAME_TOK being predicted if B_NAME_TOK hasn't been predicted since the last NAME_SEP or the beginning of the sequence.

```python
constrained_labels = set()
if "NAME_SEP" in sequence:
    # Find index of last occurance of NAME_SEP in sequence
    name_sep_idx = max(i for i, v in enumerate(sequence) if v == "NAME_SEP")
    if "B_NAME_TOK" not in sequence[name_sep_idx:]:
        constrained_labels.add("I_NAME_TOK")
else:
    if "B_NAME_TOK" not in sequence:
        constrained_labels.add("I_NAME_TOK")
```



## Data-driven constraints

Data-driven constraints are derived from label transitions never observed in the training data. We need to carefully review these constraints before we use them to make sure the constraint is reasonable and not just a limitation of the training data.

For example if we look at the training data and find the pairs of labels that are never seen, we find that (B_NAME_TOK, B_NAME_TOK) is never seen. This is a reasonable constraints to apply because we would never expect to see two ingredient names adjacent without any intervening tokens like punctuation. Whilst it might technically be possible for us to encounter such as sentence in the wild, the sentence would not be well-formed and it would not be reasonable to assume that this library can automatically handle poorly-formed sentences.

## Performance comparison

Comparison of Averaged Perceptron models, trained with and without label constraints during inference.

| Label constraints? | Word accuracy | Sentence accuracy |
| ------------------ | ------------- | ----------------- |
| With               | 97.96%        | 94.80%            |
| Without            | 97.97%        | 94.54%            |

