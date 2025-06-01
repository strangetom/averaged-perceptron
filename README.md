# Ingredient Parser: Averaged Perceptron

This repository contains a prototype implementation of a pure python Averaged Perceptron sequence tagger, intended as a replacement for the Conditional Random Fields model used by [ingredient-parser](https://github.com/strangetom/ingredient-parser).

The implementation is based on a post by Matthew Honnibal: [A Good Part-of-Speech Tagger in about 200 Lines of Python](https://explosion.ai/blog/part-of-speech-pos-tagger-in-python).

## Set up

This repository relies on some of the files in the [ingredient-parser](https://github.com/strangetom/ingredient-parser) repository, so both must be downloaded to be able to use the implementation here:

```bash
# Clone repos
$ git clones https://github.com/strangetom/ingredient-parser
$ git clone https://github.com/strangetom/averaged-perceptron
# Set up venv
$ cd averaged-perceptron
$ python3 -m venv venv
$ source venv/bin/activate
$ python -m pip install -r requirements.txt
# Train model, outputting html results, detailed results and confusion matrix
$ python train.py train --database train/data/training.sqlite3 --html --detailed --confusion
```

## Performance

The performance of this implementation is show below.

> This is the (greedy) Averaged Perceptron implementation with label transition constraints applied at inference.

```bash
Sentence-level results:
	Accuracy: 94.80%

Word-level results:
	Accuracy 97.95%
	Precision (micro) 97.92%
	Recall (micro) 97.95%
	F1 score (micro) 97.92%

```

The performance of this model exceeds the current Conditional Random Fields model used by [ingredient-parser](https://github.com/strangetom/ingredient-parser) package. I believe this is a combination of two factors:

1. The application of label transition constraints.
2. Using multiple previous labels to predict the current label. This model consider the 3 previous labels to the current label being predicted.

## Model enhancements

The baseline model is the greedy Averaged Perceptron. These enhancements are described relative that baseline.

### Label constraints

When predicting the labels for a sequence of tokens, the labels the model is allowed to select from are constrained based on the labels already predicted in the sequence.

These constraints come from two sources:

1. The label transitions observed in the training data. Certain transitions are never made and we can consider those to be disallowed. These constraints only constrain the current label prediction based on the previous label.
2. We can define constraints based on the logic of the labelling scheme. For example, `I_NAME_TOK` must always follow a `B_NAME_TOK` label (not necessarily consecutively). Therefore if the sequence of labels predicted so far does not include `B_NAME_TOK`, either since the last `NAME_SEP` if there is one or since the beginning of the sequence, `I_NAME_TOK` is disallowed from being predicted.

### Viterbi decoding

The greedy Averaged Perceptron greedily predicts the current label based on the labels with the highest weight, given the token and previous label(s). This is the local optimum, but once a label has been predicted the model commits to that label.

Viterbi decoding offers a method selecting the globally optimum sequence of labels by not committing to any labels until we can calculate the optimum sequence. The implementation if this is quite distinct from the baseline Averaged Perceptron and therefore is implemented separately in the `AveragedPerceptronViterbi` class (and the `IngredientTaggerViterbi` class). The implementation here is 1st order, meaning only the previous label is considered.

Compared to the baseline model (and using the same features only i.e. only considering the previous label), there is good improvement in performance and a considerable increase in training time and inference time.

> Future enhancements will consider the application of label transition constraints and 2nd order Viterbi decoding (although that is likely to be prohibitively slow).

