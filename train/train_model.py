#!/usr/bin/env python3

import argparse
import concurrent.futures as cf
import contextlib
import logging
import random
from itertools import chain
from pathlib import Path
from statistics import mean, stdev
from typing import Literal
from uuid import uuid4

from sklearn.model_selection import train_test_split
from tqdm import tqdm

from ap.ingredient_tagger import IngredientTagger, IngredientTaggerViterbi

from .test_results_to_detailed_results import test_results_to_detailed_results
from .test_results_to_html import test_results_to_html
from .training_utils import (
    DataVectors,
    Stats,
    confusion_matrix,
    evaluate,
    load_datasets,
)

DEFAULT_MODEL_LOCATION = "PARSER.json.gz"

logger = logging.getLogger(__name__)


def train_model(
    vectors: DataVectors,
    model_type: Literal["ap", "ap_viterbi"],
    split: float,
    save_model: Path,
    seed: int | None,
    html: bool,
    detailed_results: bool,
    plot_confusion_matrix: bool,
    keep_model: bool = True,
    combine_name_labels: bool = False,
    show_progress: bool = True,
) -> Stats:
    """Train model using vectors, splitting the vectors into a train and evaluation
    set based on <split>. The trained model is saved to <save_model>.

    Parameters
    ----------
    vectors : DataVectors
        Vectors loaded from training csv files
    model_type : Literal["ap", "ap_viterbi"]
        Model type to train.
        ap = AveragedPerceptron.
        ap_viterbi = AveragedPerceptron using viterbi decoding.
    split : float
        Fraction of vectors to use for evaluation.
    save_model : Path
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
    keep_model : bool, optional
        If False, delete model from disk after evaluating it's performance.
        Default is True.
    combine_name_labels : bool, optional
        If True, combine all NAME labels into a single NAME label.
        Default is False
    show_progress: bool, optional
        If True, show progress bar for iterations.
        Default is True.

    Returns
    -------
    Stats
        Statistics evaluating the model
    """
    # Generate random seed for the train/test split if none provided.
    if seed is None:
        seed = random.randint(0, 1_000_000_000)

    logger.info(f"{seed} is the random seed used for the train/test split.")
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
    logger.info(f"{len(features_train):,} training vectors.")
    logger.info(f"{len(features_test):,} testing vectors.")

    logger.info(f'Training "{model_type}" model with training data.')
    if model_type == "ap":
        tagger = IngredientTagger()
    elif model_type == "ap_viterbi":
        tagger = IngredientTaggerViterbi()

    tagger.labels = set(chain.from_iterable(truth_train))
    tagger.model.labels = tagger.labels
    tagger.train(
        features_train,
        truth_train,
        n_iter=15,
        min_abs_weight=1,
        quantize=False,
        make_label_dict=False,
        show_progress=show_progress,
    )
    if keep_model:
        tagger.save(str(save_model))

    logger.info("Evaluating model with test data.")
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

    stats = evaluate(labels_pred, truth_test, seed, combine_name_labels)
    return stats


def train_single(args: argparse.Namespace) -> None:
    """Train CRF model once.

    Parameters
    ----------
    args : argparse.Namespace
        Model training configuration
    """
    vectors = load_datasets(
        args.database,
        args.table,
        args.datasets,
        discard_other=True,
        combine_name_labels=args.combine_name_labels,
    )

    if args.save_model is None:
        save_model = DEFAULT_MODEL_LOCATION
    else:
        save_model = args.save_model

    stats = train_model(
        vectors,
        args.model,
        args.split,
        Path(save_model),
        args.seed,
        args.html,
        args.detailed,
        args.confusion,
        keep_model=True,
        combine_name_labels=args.combine_name_labels,
        show_progress=True,
    )

    print("Sentence-level results:")
    print(f"\tAccuracy: {100 * stats.sentence.accuracy:.2f}%")

    print()
    print("Word-level results:")
    print(f"\tAccuracy {100 * stats.token.accuracy:.2f}%")
    print(f"\tPrecision (micro) {100 * stats.token.weighted_avg.precision:.2f}%")
    print(f"\tRecall (micro) {100 * stats.token.weighted_avg.recall:.2f}%")
    print(f"\tF1 score (micro) {100 * stats.token.weighted_avg.f1_score:.2f}%")


def train_multiple(args: argparse.Namespace) -> None:
    """Train model multiple times and calculate average performance
    and performance uncertainty.

    Parameters
    ----------
    args : argparse.Namespace
        Model training configuration
    """
    vectors = load_datasets(
        args.database,
        args.table,
        args.datasets,
        discard_other=True,
        combine_name_labels=args.combine_name_labels,
    )

    if args.save_model is None:
        save_model = DEFAULT_MODEL_LOCATION
    else:
        save_model = args.save_model

    # The first None argument is for the seed. This is set to None so each
    # iteration of the training function uses a different random seed.
    arguments = [
        (
            vectors,
            args.model,
            args.split,
            Path(save_model).with_stem("model-" + str(uuid4())),
            None,  # Seed
            False,  # html
            False,  # detailed_results
            False,  # plot_confusion_matrix
            False,  # keep_model
            args.combine_name_labels,
            False,  # show_progess
        )
    ] * args.runs

    # Disable logging below INFO level
    log_level = logger.manager.disable
    logging.disable(logging.INFO)

    with contextlib.redirect_stdout(None):  # Suppress print output
        with cf.ProcessPoolExecutor(max_workers=args.processes) as executor:
            futures = [executor.submit(train_model, *a) for a in arguments]
            eval_results = [
                future.result()
                for future in tqdm(cf.as_completed(futures), total=len(futures))
            ]

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
    max_sent = 100 * sentence_accuracies[index_best]
    max_word = 100 * word_accuracies[index_best]
    max_seed = seeds[index_best]
    index_worst = min(
        range(len(sentence_accuracies)), key=sentence_accuracies.__getitem__
    )
    min_sent = 100 * sentence_accuracies[index_worst]
    min_word = 100 * word_accuracies[index_worst]
    min_seed = seeds[index_worst]
    print()
    print(f"Best:  Sentence {max_sent:.2f}% / Word {max_word:.2f}% (Seed: {max_seed})")
    print(f"Worst: Sentence {min_sent:.2f}% / Word {min_word:.2f}% (Seed: {min_seed})")

    # Restore logging to previous setting
    logging.disable(log_level)
