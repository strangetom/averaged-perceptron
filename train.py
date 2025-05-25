#!/usr/bin/env python3

import argparse
import logging
import os
import sys

from train import (
    train_multiple,
    train_single,
)

LOGGING_LEVEL = {
    0: logging.INFO,
    1: logging.DEBUG,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a CRF model to parse label token from recipe \
                    ingredient sentences."
    )
    subparsers = parser.add_subparsers(dest="command", help="Training commands")

    train_parser = subparsers.add_parser("train", help="Train A{} model.")
    train_parser.add_argument(
        "--database",
        help="Path to database of training data",
        type=str,
        dest="database",
        required=True,
    )
    train_parser.add_argument(
        "--database-table",
        help="Name of table in database containing training data",
        type=str,
        dest="table",
        default="en",
    )
    train_parser.add_argument(
        "--datasets",
        help="Datasets to use in training and evaluating the model",
        dest="datasets",
        nargs="*",
        default=["bbc", "cookstr", "nyt", "allrecipes", "tc"],
    )
    train_parser.add_argument(
        "--split",
        default=0.20,
        type=float,
        help="Fraction of data to be used for testing",
    )
    train_parser.add_argument(
        "--seed",
        default=None,
        type=int,
        help="Seed value used for train/test split.",
    )
    train_parser.add_argument(
        "--html",
        action="store_true",
        help="Output a markdown file containing detailed results.",
    )
    train_parser.add_argument(
        "--detailed",
        action="store_true",
        help="Output a file containing detailed results about accuracy.",
    )
    train_parser.add_argument(
        "--confusion",
        action="store_true",
        help="Plot confusion matrix of token labels.",
    )
    train_parser.add_argument(
        "-v",
        help="Enable verbose output.",
        action="count",
        default=0,
        dest="verbose",
    )

    multiple_parser_help = "Average AP performance across multiple training cycles."
    multiple_parser = subparsers.add_parser("multiple", help=multiple_parser_help)
    multiple_parser.add_argument(
        "--database",
        help="Path to database of training data",
        type=str,
        dest="database",
        required=True,
    )
    multiple_parser.add_argument(
        "--database-table",
        help="Name of table in database containing training data",
        type=str,
        dest="table",
        default="en",
    )
    multiple_parser.add_argument(
        "--datasets",
        help="Datasets to use in training and evaluating the model",
        dest="datasets",
        nargs="*",
        default=["bbc", "cookstr", "nyt", "allrecipes", "tc"],
    )
    multiple_parser.add_argument(
        "--split",
        default=0.20,
        type=float,
        help="Fraction of data to be used for testing",
    )
    multiple_parser.add_argument(
        "--html",
        action="store_true",
        help="Output a markdown file containing detailed results.",
    )
    multiple_parser.add_argument(
        "--detailed",
        action="store_true",
        help="Output a file containing detailed results about accuracy.",
    )
    multiple_parser.add_argument(
        "--confusion",
        action="store_true",
        help="Plot confusion matrix of token labels.",
    )
    multiple_parser.add_argument(
        "-r",
        "--runs",
        default=10,
        type=int,
        help="Number of times to run the training and evaluation of the model.",
    )
    multiple_parser.add_argument(
        "-p",
        "--processes",
        default=os.cpu_count() - 1,
        type=int,
        help="Number of processes to spawn. Default to number of cpu cores.",
    )
    multiple_parser.add_argument(
        "-v",
        help="Enable verbose output.",
        action="count",
        default=0,
        dest="verbose",
    )

    args = parser.parse_args()

    logging.basicConfig(
        stream=sys.stdout,
        level=LOGGING_LEVEL[args.verbose],
        format="[%(levelname)s] (%(module)s) %(message)s",
    )

    if args.command == "train":
        train_single(args)
    elif args.command == "multiple":
        train_multiple(args)
