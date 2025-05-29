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

```bash
Sentence-level results:
	Accuracy: 94.34%

Word-level results:
	Accuracy 97.82%
	Precision (micro) 97.80%
	Recall (micro) 97.82%
	F1 score (micro) 97.80%
```

This compares favourably with current state of the art performance for the [ingredient-parser](https://github.com/strangetom/ingredient-parser) library, although the performance is not quite a good.

## Model enhancements

The Averaged Perceptron model is enhanced in the following ways:

### Label constraints

When predicting the labels for a sequence of tokens, the labels the model is allowed to select from are constrained based on the labels already predicted in the sequence.

These constraints are defined from the labelling scheme. For example, `I_NAME_TOK` must always follow a `B_NAME_TOK` label (not necessarily consecutively). Therefore if the sequence of labels predicted so far does not include `B_NAME_TOK`, either since the last `NAME_SEP` if there is one or since the beginning of the sequence, `I_NAME_TOK` is forbidden from being predicted.

This improves the model performance by ~0.25%, but only when applied during inference. If applied during training, the model performance worsens.
