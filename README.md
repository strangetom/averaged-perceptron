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
# Train model, outputting html results, detailed results and confusion matrix
$ python train.py train --database train/data/training.sqlite3 --html --detailed --confusion
```

## Performance

The performance of this implementation is show below.

```bash
Sentence-level results:
	Accuracy: 94.45%

Word-level results:
	Accuracy 98.06%
	Precision (micro) 98.04%
	Recall (micro) 98.06%
	F1 score (micro) 98.05%
```

This compares favourably with current state of the art performance for the [ingredient-parser](https://github.com/strangetom/ingredient-parser) library, although the performance is not quite a good.



