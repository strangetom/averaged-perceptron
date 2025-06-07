# Ingredient Parser: Averaged Perceptron

This repository contains a prototype implementation of a pure python Averaged Perceptron sequence tagger, intended as a replacement for the Conditional Random Fields model used by [ingredient-parser](https://github.com/strangetom/ingredient-parser).

The implementation is based on a post by Matthew Honnibal: [A Good Part-of-Speech Tagger in about 200 Lines of Python](https://explosion.ai/blog/part-of-speech-pos-tagger-in-python).

See the [docs](docs/README.md) for more details on the implementation in this application.

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
