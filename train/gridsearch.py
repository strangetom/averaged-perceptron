#!/usr/bin/env python3

import argparse
import concurrent.futures as cf
import logging
import os
import random
import time
from dataclasses import asdict, dataclass
from datetime import timedelta
from itertools import chain, product
from pathlib import Path
from typing import Literal
from uuid import uuid4

from sklearn.model_selection import train_test_split
from tabulate import tabulate
from tqdm import tqdm

from ap.ingredient_tagger import IngredientTagger, IngredientTaggerViterbi

from .train_model import DEFAULT_MODEL_LOCATION
from .training_utils import (
    DataVectors,
    evaluate,
    load_datasets,
)

logger = logging.getLogger(__name__)


@dataclass
class HyperParameters:
    epochs: int
    only_positive_bool_features: bool
    apply_label_constraints: bool
    min_abs_weight: float
    quantize_bits: int | None
    make_label_dict: bool
    model_type: Literal["ap", "ap_viterbi"]


def default_hyperparams() -> HyperParameters:
    # Create object with defaults
    return HyperParameters(
        epochs=15,
        only_positive_bool_features=False,
        apply_label_constraints=True,
        min_abs_weight=1,
        quantize_bits=None,
        make_label_dict=False,
        model_type="ap",
    )


def param_combos(params: dict) -> list[HyperParameters]:
    """Generate list of dictionaries covering all possible combinations of parameters
    and their values given in the params input.

    Parameters
    ----------
    params : dict
        dict of parameters with list of values for each parameter

    Returns
    -------
    list[HyperParameters]
        list of dicts, where each dict has a single value for each parameter.
        The dicts in the list cover all possible combinations of the input parameters.
    """
    combinations = []
    for combo in product(*params.values()):
        iteration = dict(zip(params.keys(), combo))

        hyperparams = default_hyperparams()
        # modify fields of hyperparams for combination of params in iteration
        for key, value in iteration.items():
            if not hasattr(hyperparams, key):
                raise KeyError(
                    f"'{key}'' is not a hyper-parameter that can be searched over."
                )
            setattr(hyperparams, key, value)

        combinations.append(hyperparams)

    return combinations


def generate_argument_sets(args: argparse.Namespace) -> list[list]:
    """Generate list of lists, where each sublist is the arguments required by the
    train_model_grid_search function:
        parameters
        vectors
        split
        save_model
        seed
        keep_model
        combine_name_labels

    Parameters
    ----------
    args : argparse.Namespace
        Arguments parsed from command line

    Returns
    -------
    list[list]
        list of lists, where each sublist is the arguments for training a model with
        one of the combinations of algorithms and parameters
    """
    vectors = load_datasets(
        args.database,
        args.table,
        args.datasets,
        discard_other=True,
        combine_name_labels=args.combine_name_labels,
    )

    # Generate list of arguments for all combinations parameters for each algorithm
    argument_sets = []

    if args.save_model is None:
        save_model = DEFAULT_MODEL_LOCATION
    else:
        save_model = args.save_model

    # Generate all combinations of parameters
    for parameter_set in param_combos(args.hyperparameters):
        arguments = [
            parameter_set,
            vectors,
            args.split,
            save_model,
            args.seed,
            args.keep_models,
            args.combine_name_labels,
        ]
        argument_sets.append(arguments)

    return argument_sets


def train_model_grid_search(
    parameters: HyperParameters,
    vectors: DataVectors,
    split: float,
    save_model: str,
    seed: int,
    keep_model: bool,
    combine_name_labels: bool,
) -> dict:
    """Train model using given training parameters,
    returning model performance statistics, model parameters and elapsed training time.

    Parameters
    ----------
    parameters : HyperParameters
        Hyper-parameters values to grid search over.
    vectors : DataVectors
        Vectors loaded from training csv files
    split : float
        Fraction of vectors to use for evaluation.
    save_model : str
        Path to save trained model to.
    seed : int
        Integer used as seed for splitting the vectors between the training and
        testing sets.
    keep_model : bool
        If True, keep model after evaluation, otherwise delete it.

    Returns
    -------
    dict
        Statistics from evaluating the model
    """
    random.seed(seed)
    start_time = time.monotonic()

    # Split data into train and test sets
    # The stratify argument means that each dataset is represented proprtionally
    # in the train and tests sets, avoiding the possibility that train or tests sets
    # contain data from one dataset disproportionally.
    (
        _,
        _,
        features_train,
        features_test,
        truth_train,
        truth_test,
        _,
        _,
    ) = train_test_split(
        vectors.sentences,
        vectors.features,
        vectors.labels,
        vectors.source,
        test_size=split,
        stratify=vectors.source,
        random_state=seed,
    )

    # Make model name unique
    uuid = str(uuid4())
    save_model_path = Path(save_model).with_stem("model-" + uuid)

    # Train model
    if parameters.model_type == "ap":
        tagger = IngredientTagger(
            only_positive_bool_features=parameters.only_positive_bool_features,
            apply_label_constraints=parameters.apply_label_constraints,
        )
    elif parameters.model_type == "ap_viterbi":
        tagger = IngredientTaggerViterbi(
            only_positive_bool_features=parameters.only_positive_bool_features,
        )
    else:
        raise ValueError(f"{parameters.model_type} is unknown model type.")

    tagger.labels = set(chain.from_iterable(truth_train))
    tagger.model.labels = tagger.labels
    tagger.train(
        features_train,
        truth_train,
        n_iter=parameters.epochs,
        min_abs_weight=parameters.min_abs_weight,
        quantize_bits=parameters.quantize_bits,
        make_label_dict=parameters.make_label_dict,
        show_progress=False,
    )
    tagger.save(str(save_model_path))

    # Get model size, in MB
    model_size = os.path.getsize(save_model_path) / 1024**2

    # Evaluate model
    labels_pred = []
    scores_pred = []
    for feats in features_test:
        tags = tagger.tag_from_features(feats)
        if not tags:
            continue
        labels, scores = zip(*tags)
        labels_pred.append(list(labels))
        scores_pred.append(list(scores))
    stats = evaluate(labels_pred, truth_test, seed, combine_name_labels)

    if not keep_model:
        save_model_path.unlink(missing_ok=True)

    return {
        "model_size": model_size,
        "uuid": uuid,
        "params": parameters,
        "stats": stats,
        "time": time.monotonic() - start_time,
    }


def grid_search(args: argparse.Namespace):
    """Perform a grid search over the specified hyperparameters and return the model
    performance statistics for each combination of parameters.

    Parameters
    ----------
    args : argparse.Namespace
        Grid search configuration
    """
    arguments = generate_argument_sets(args)

    logger.info(f"Grid search over {len(arguments)} hyperparameters combinations.")
    logger.info(f"{args.seed} is the random seed used for the train/test split.")

    with cf.ProcessPoolExecutor(max_workers=args.processes) as executor:
        futures = [executor.submit(train_model_grid_search, *a) for a in arguments]
        eval_results = [
            future.result()
            for future in tqdm(cf.as_completed(futures), total=len(futures))
        ]

    # Sort with highest sentence accuracy first
    eval_results = sorted(
        eval_results, key=lambda x: x["stats"].sentence.accuracy, reverse=True
    )

    headers = [
        "Algorithm",
        "Parameters",
        "Token accuracy",
        "Sentence accuracy",
        "Time",
        "Size (MB)",
        "uuid",
    ]
    table = []
    for result in eval_results:
        algo = result["params"].model_type
        params = asdict(result["params"])
        stats = result["stats"]
        size = result["model_size"]
        time = timedelta(seconds=int(result["time"]))
        uuid = result["uuid"]
        table.append(
            [
                algo,
                ", ".join([f"{k}={v}" for k, v in params.items()]),
                f"{100 * stats.token.accuracy:.2f}%",
                f"{100 * stats.sentence.accuracy:.2f}%",
                str(time),
                f"{size:.2f}",
                uuid[:6],
            ]
        )

    print(
        tabulate(
            table,
            headers=headers,
            tablefmt="fancy_grid",
            maxcolwidths=[None, 120, None, None, None, None],
        )
    )
