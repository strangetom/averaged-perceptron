#!/usr/bin/env python3

import argparse
import concurrent.futures as cf
import contextlib
from itertools import chain
import random
from statistics import mean, stdev

from sklearn.model_selection import train_test_split
from tqdm import tqdm

from ap.ingredient_tagger import IngredientTagger

from .test_results_to_detailed_results import test_results_to_detailed_results
from .test_results_to_html import test_results_to_html
from .training_utils import (
    DataVectors,
    ModelType,
    Stats,
    confusion_matrix,
    evaluate,
    load_datasets,
)


def get_model_type(cmd_arg: str) -> ModelType:
    """Convert command line argument for model type into enum

    Parameters
    ----------
    cmd_arg : str
        Command line argument for model

    Returns
    -------
    ModelType
    """
    types = {
        "parser": ModelType.PARSER,
        "foundationfoods": ModelType.FOUNDATION_FOODS,
    }
    return types[cmd_arg]


def train_model(
    vectors: DataVectors,
    split: float,
    seed: int | None,
    html: bool,
    detailed_results: bool,
    plot_confusion_matrix: bool,
    model: str,
) -> Stats:
    """Train model using vectors, splitting the vectors into a train and evaluation
    set based on <split>. The trained model is saved to <save_model>.

    Parameters
    ----------
    vectors : DataVectors
        Vectors loaded from training csv files
    split : float
        Fraction of vectors to use for evaluation.
    save_model : str
        Path to save trained model to.
    seed : int | None
        Integer used as seed for splitting the vectors between the training and
        testing sets. If None, a random seed is generated within this function.
    html : bool
        If True, write html file of incorrect evaluation sentences
        and print out details about OTHER labels.
    detailed_results : bool
        If True, write output files with details about how labeling performed on
        the test set.
    plot_confusion_matrix : bool
        If True, plot a confusion matrix of the token labels.
    model : str
        Type of model being trained, used to set the tagger labels.
        'parser' or 'foundationfoods'.

    Returns
    -------
    Stats
        Statistics evaluating the model
    """
    model = get_model_type(model)

    # Generate random seed for the train/test split if none provided.
    if seed is None:
        seed = random.randint(0, 1_000_000_000)

    print(f"[INFO] {seed} is the random seed used for the train/test split.")
    random.seed(seed)

    # Split data into train and test sets
    # The stratify argument means that each dataset is represented proprtionally
    # in the train and tests sets, avoiding the possibility that train or tests sets
    # contain data from one dataset disproportionally.
    (
        _,
        sentences_test,
        features_train,
        features_test,
        truth_train,
        truth_test,
        _,
        source_test,
        _,
        tokens_test,
    ) = train_test_split(
        vectors.sentences,
        vectors.features,
        vectors.labels,
        vectors.source,
        vectors.tokens,
        test_size=split,
        stratify=vectors.source,
        random_state=seed,
    )
    print(f"[INFO] {len(features_train):,} training vectors.")
    print(f"[INFO] {len(features_test):,} testing vectors.")

    print("[INFO] Training model with training data.")
    tagger = IngredientTagger()

    tagger.model.labels = set(chain.from_iterable(truth_train))
    tagger.train(
        features_train,
        truth_train,
        n_iter=15,
        min_abs_weight=0.15,
        quantize=False,
        verbose=False,
    )
    tagger.save(f"{model.name}.json")

    print("[INFO] Evaluating model with test data.")
    labels_pred = []
    scores_pred = []
    for feats in features_test:
        tags = tagger.tag_from_features(feats)
        if not tags:
            continue
        labels, scores = zip(*tags)
        labels_pred.append(list(labels))
        scores_pred.append(list(scores))

    if html:
        test_results_to_html(
            sentences_test,
            tokens_test,
            truth_test,
            labels_pred,
            scores_pred,
            source_test,
        )

    if detailed_results:
        test_results_to_detailed_results(
            sentences_test,
            tokens_test,
            truth_test,
            labels_pred,
            scores_pred,
        )

    if plot_confusion_matrix:
        confusion_matrix(labels_pred, truth_test)

    stats = evaluate(labels_pred, truth_test, seed, model)
    return stats


def train_single(args: argparse.Namespace) -> None:
    """Train CRF model once.

    Parameters
    ----------
    args : argparse.Namespace
        Model training configuration
    """
    vectors = load_datasets(
        args.database, args.table, args.datasets, get_model_type(args.model)
    )
    stats = train_model(
        vectors,
        args.split,
        args.seed,
        args.html,
        args.detailed,
        args.confusion,
        args.model,
    )

    print("Sentence-level results:")
    print(f"\tAccuracy: {100*stats.sentence.accuracy:.2f}%")

    print()
    print("Word-level results:")
    print(f"\tAccuracy {100*stats.token.accuracy:.2f}%")
    print(f"\tPrecision (micro) {100*stats.token.weighted_avg.precision:.2f}%")
    print(f"\tRecall (micro) {100*stats.token.weighted_avg.recall:.2f}%")
    print(f"\tF1 score (micro) {100*stats.token.weighted_avg.f1_score:.2f}%")


def train_multiple(args: argparse.Namespace) -> None:
    """Train model multiple times and calculate average performance
    and performance uncertainty.

    Parameters
    ----------
    args : argparse.Namespace
        Model training configuration
    """
    vectors = load_datasets(
        args.database, args.table, args.datasets, args.model == "foundationfoods"
    )

    # The first None argument is for the seed. This is set to None so each
    # iteration of the training function uses a different random seed.
    arguments = [
        (
            vectors,
            args.split,
            None,
            args.html,
            args.detailed,
            args.confusion,
            args.model,
        )
    ] * args.runs

    eval_results = []
    with contextlib.redirect_stdout(None):  # Suppress print output
        with cf.ProcessPoolExecutor(max_workers=args.processes) as executor:
            futures = [executor.submit(train_model, *a) for a in arguments]
            for future in tqdm(cf.as_completed(futures), total=len(futures)):
                eval_results.append(future.result())

    word_accuracies, sentence_accuracies, seeds = [], [], []
    for result in eval_results:
        sentence_accuracies.append(result.sentence.accuracy)
        word_accuracies.append(result.token.accuracy)
        seeds.append(result.seed)

    sentence_mean = 100 * mean(sentence_accuracies)
    sentence_uncertainty = 3 * 100 * stdev(sentence_accuracies)
    print()
    print("Average sentence-level accuracy:")
    print(f"\t-> {sentence_mean:.2f}% ± {sentence_uncertainty:.2f}%")

    word_mean = 100 * mean(word_accuracies)
    word_uncertainty = 3 * 100 * stdev(word_accuracies)
    print()
    print("Average word-level accuracy:")
    print(f"\t-> {word_mean:.2f}% ± {word_uncertainty:.2f}%")

    index_best = max(
        range(len(sentence_accuracies)), key=sentence_accuracies.__getitem__
    )
    best_sentence = 100 * sentence_accuracies[index_best]
    best_word = 100 * word_accuracies[index_best]
    best_seed = seeds[index_best]
    index_worst = min(
        range(len(sentence_accuracies)), key=sentence_accuracies.__getitem__
    )
    worst_sentence = 100 * sentence_accuracies[index_worst]
    worst_word = 100 * word_accuracies[index_worst]
    worst_seed = seeds[index_worst]
    print()
    print(
        f"Best:  Sentence {best_sentence:.2f}% / Word {best_word:.2f}% (Seed: {best_seed})"
    )
    print(
        f"Worst: Sentence {worst_sentence:.2f}% / Word {worst_word:.2f}% (Seed: {worst_seed})"
    )
