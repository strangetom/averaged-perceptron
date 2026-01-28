# Ingredient Parser: Averaged Perceptron

This repository contains implementations of the Averaged Perceptron sequence tagger (and some variations), intended as a replacement for the Conditional Random Fields model used by [ingredient-parser](https://github.com/strangetom/ingredient-parser).

The basic, greedy implementation is based on a post by Matthew Honnibal: [A Good Part-of-Speech Tagger in about 200 Lines of Python](https://explosion.ai/blog/part-of-speech-pos-tagger-in-python), which in turn implements the perceptron algorithm described in [^1].

See the [docs](docs/README.md) for more details of the implementation in this application and the various optimisations and variations that have been investigated.

## Set up

This repository relies on some of the files in the [ingredient-parser](https://github.com/strangetom/ingredient-parser) repository, so both must be downloaded to be able to use the implementation here:

```bash
# Clone repos
$ git clone https://github.com/strangetom/ingredient-parser
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

> This is the (greedy) Averaged Perceptron implementation with the following hyperparameters:
>
> ```
> n_iter=20
> min_abs_weight=2
> min_feat_updates=5
> quantize_bits=8
> 
> seed=170365540
> ```

```bash
╒══════════════════════════╤══════════════════════════╕
│ Sentence-level results   │ Word-level results       │
╞══════════════════════════╪══════════════════════════╡
│ Accuracy: 95.04%         │ Accuracy: 98.17%         │
│                          │ Precision (micro) 98.15% │
│                          │ Recall (micro) 98.17%    │
│                          │ F1 score (micro) 98.15%  │
╘══════════════════════════╧══════════════════════════╛
```

The performance of this model is _similar_ (slightly worse for the sentence level results, slightly better for the work level results) to the current Conditional Random Fields model used by [ingredient-parser](https://github.com/strangetom/ingredient-parser) package. I believe this is a combination of two factors:

1. The application of label transition constraints.
2. Using multiple previous labels to predict the current label. This model consider the 3 previous labels to the current label being predicted.

Additionally, due the applications of feature filtering, weight pruning, and quantization the saved model is more than 5x smaller than the Conditional Random Fields model.

## References

[^1]: M. Collins, ‘Discriminative Training Methods for Hidden Markov Models: Theory and Experiments with Perceptron Algorithms’, in *Proceedings of the 2002 Conference on Empirical Methods in Natural Language Processing (EMNLP 2002)*, Association for Computational Linguistics, Jul. 2002, pp. 1–8. doi: [10.3115/1118693.1118694](https://doi.org/10.3115/1118693.1118694).
