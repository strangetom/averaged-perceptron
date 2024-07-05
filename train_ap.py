#!/usr/env python3

import random
import sys

sys.path.append("../ingredient-parser")

from sklearn.model_selection import train_test_split

from ap.ingredient_tagger import IngredientTagger

from train.training_utils import load_datasets, evaluate
from train.test_results_to_html import test_results_to_html

SEED = 561528489

if __name__ == "__main__":
    vectors = load_datasets(
        "../ingredient-parser/train/data/training.sqlite3",
        "en",
        ["bbc", "cookstr", "nyt", "allrecipes"],
    )
    (
        sentences_train,
        sentences_test,
        features_train,
        features_test,
        truth_train,
        truth_test,
        source_train,
        source_test,
    ) = train_test_split(
        vectors.sentences,
        vectors.features,
        vectors.labels,
        vectors.source,
        test_size=0.2,
        random_state=SEED,
        stratify=vectors.source,
    )

    random.seed(SEED)
    tagger = IngredientTagger()
    tagger.model.labels = {
        "QTY",
        "UNIT",
        "NAME",
        "PREP",
        "COMMENT",
        "PURPOSE",
        "PUNC",
        "SIZE",
    }
    tagger.train(features_train, truth_train, n_iter=15, min_abs_weight=0.25)
    tagger.save("ap.pickle")

    predicted_labels = []
    for feats in features_test:
        labels = [label for label, _ in tagger.tag_from_features(feats)]
        predicted_labels.append(labels)

    stats = evaluate(truth_test, predicted_labels)
    print("Sentence-level results:")
    print(f"\tAccuracy: {100*stats.sentence.accuracy:.2f}%")

    print()
    print("Word-level results:")
    print(f"\tAccuracy {100*stats.token.accuracy:.2f}%")
    print(f"\tPrecision (micro) {100*stats.token.weighted_avg.precision:.2f}%")
    print(f"\tRecall (micro) {100*stats.token.weighted_avg.recall:.2f}%")
    print(f"\tF1 score (micro) {100*stats.token.weighted_avg.f1_score:.2f}%")

    test_results_to_html(
        sentences_test,
        truth_test,
        predicted_labels,
        [[1.0] * len(t) for t in truth_test],
        source_test,
        lambda x: x >= 1,
    )
