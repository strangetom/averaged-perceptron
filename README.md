# Ingredient Parser: Averaged Perceptron

This repository contains a prototype implementation of a pure python Averaged Perceptron sequence tagger, intended as a replacement for the Conditional Random Fields model used by [ingredient-parser](https://github.com/strangetom/ingredient-parser).

The implementation is based on a post by Matthew Honnibal: [A Good Part-of-Speech Tagger in about 200 Lines of Python](https://explosion.ai/blog/part-of-speech-pos-tagger-in-python).

## Implementation

The model implementation is in [averaged_perceptron.py](ap/averaged_perceptron.py).

The jupyter notebook [here](notebooks/averaged_perceptron.ipynb) goes through the training and evaluation of the model.

>  ![NOTE]
>  Note that to train the model you will need the database of training data from the `ingredient-parser` library, found [here](https://github.com/strangetom/ingredient-parser/blob/master/train/data/training.sqlite3).

## Performance

The performance of this implementation is show below.

```bash
Sentence-level results:
	Accuracy: 93.51%

Word-level results:
	Accuracy 97.74%
	Precision (micro) 97.76%
	Recall (micro) 97.74%
	F1 score (micro) 97.75%
```

This compares favourably with current state of the art performance for the [ingredient-parser](https://github.com/strangetom/ingredient-parser) library, although the performance is not quite a good.



