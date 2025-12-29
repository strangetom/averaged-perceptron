#!/usr/bin/env python3

import argparse
import sqlite3
import time

from ingredient_parser.en import PostProcessor, PreProcessor
from ingredient_parser.en._utils import pluralise_units

from ap import IngredientTagger

TAGGER = IngredientTagger("PARSER.json.gz")


def parse_ingredient(sentence):
    processed_sentence = PreProcessor(sentence)
    labels, scores = zip(
        *TAGGER.tag_from_features(processed_sentence.sentence_features())
    )
    tokens = [t.text for t in processed_sentence.tokenized_sentence]
    pos_tags = [t.pos_tag for t in processed_sentence.tokenized_sentence]
    labels = list(labels)
    scores = list(scores)

    # Re-pluralise tokens that were singularised if the label isn't UNIT
    # For tokens with UNIT label, we'll deal with them below
    for idx in processed_sentence.singularised_indices:
        token = tokens[idx]
        label = labels[idx]
        if label != "UNIT":
            tokens[idx] = pluralise_units(token)

    postprocessed_sentence = PostProcessor(
        sentence,
        tokens,
        pos_tags,
        labels,
        scores,
        discard_isolated_stop_words=True,
        string_units=False,
        volumetric_units_system="imperial",
    )
    return postprocessed_sentence.parsed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingredient Parser benchmark")
    parser.add_argument(
        "-n", type=int, help="Number of sentences to use from each source.", default=100
    )
    parser.add_argument(
        "--iterations", "-i", type=int, help="Number of iterations to run.", default=500
    )
    args = parser.parse_args()

    with sqlite3.connect(
        "train/data/training.sqlite3", detect_types=sqlite3.PARSE_DECLTYPES
    ) as conn:
        c = conn.cursor()
        # Select `n` sentences from each source
        c.execute(
            """
            SELECT sentence FROM 
            (
                SELECT *, row_number() OVER (
                    PARTITION BY source ORDER BY rowid
                ) 
                AS rn FROM en
            )
            WHERE rn <= ?
            ORDER BY source, rn
            """,
            (args.n,),
        )
        sentences = [sent for (sent,) in c.fetchall()]
    conn.close()

    start = time.time()
    for i in range(args.iterations):
        for sent in sentences:
            parse_ingredient(sent)

    duration = time.time() - start
    total_sentences = args.iterations * len(sentences)
    print(f"Elapsed time: {duration:.2f} s.")
    print(f"{total_sentences} total iterations.")
    print(f"{1e6 * duration / total_sentences:.2f} us/sentence.")
    print(f"{total_sentences / duration:.2f} sentences/second.")
