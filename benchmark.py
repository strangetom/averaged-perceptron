#!/usr/bin/env python3
import argparse
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
        imperial_units=False,
    )
    return postprocessed_sentence.parsed


if __name__ == "__main__":
    sentences = [
        ("&frac12; cup warm water (105°F)", "0.5 cup warm water (105°F)"),
        ("3 1/2 chilis anchos", "3.5 chilis anchos"),
        ("six eggs", "6 eggs"),
        ("thumbnail-size piece ginger", "thumbnail-size piece ginger"),
        (
            "2 cups flour – white or self-raising",
            "2 cups flour - white or self-raising",
        ),
        ("3–4 sirloin steaks", "3-4 sirloin steaks"),
        ("three large onions", "3 large onions"),
        ("twelve bonbons", "12 bonbons"),
        ("1&frac34; cups tomato ketchup", "1.75 cups tomato ketchup"),
        ("1/2 cup icing sugar", "0.5 cup icing sugar"),
        ("2 3/4 pound chickpeas", "2.75 pound chickpeas"),
        ("1 and 1/2 tsp fine grain sea salt", "1.5 tsp fine grain sea salt"),
        ("1 and 1/4 cups dark chocolate morsels", "1.25 cups dark chocolate morsels"),
        ("½ cup icing sugar", "0.5 cup icing sugar"),
        ("3⅓ cups warm water", "3.333 cups warm water"),
        ("¼-½ teaspoon", "0.25-0.5 teaspoon"),
        ("100g green beans", "100 g green beans"),
        ("2-pound red peppers, sliced", "2 pound red peppers, sliced"),
        ("2lb1oz cherry tomatoes", "2 lb 1 oz cherry tomatoes"),
        ("2lb-1oz cherry tomatoes", "2 lb - 1 oz cherry tomatoes"),
        ("1 tsp. garlic powder", "1 tsp garlic powder"),
        ("5 oz. chopped tomatoes", "5 oz chopped tomatoes"),
        ("1 to 2 mashed bananas", "1-2 mashed bananas"),
        ("5- or 6- large apples", "5-6- large apples"),
        ("227 g - 283.5 g/8-10 oz duck breast", "227-283.5 g/8-10 oz duck breast"),
        ("400-500 g/14 oz - 17 oz rhubarb", "400-500 g/14-17 oz rhubarb"),
        ("8 x 450 g/1 lb live lobsters", "8x 450 g/1 lb live lobsters"),
        ("4 x 100 g wild salmon fillet", "4x 100 g wild salmon fillet"),
        (
            "½ - ¾ cup heavy cream, plus extra for brushing the tops of the scones",
            "0.5-0.75 cup heavy cream, plus extra for brushing the tops of the scones",
        ),
    ]

    parser = argparse.ArgumentParser(description="Ingredient Parser benchmark")
    parser.add_argument(
        "--iterations", "-i", type=int, help="Number of iterations to run.", default=500
    )
    args = parser.parse_args()

    start = time.time()
    for i in range(args.iterations):
        for sent, _ in sentences:
            parse_ingredient(sent)

    total_sentences = args.iterations * len(sentences)
    duration = time.time() - start
    print(f"Elapsed time: {duration:.2f} s")
    print(f"{1e6 * duration / total_sentences:.2f} us/sentence")
    print(f"{int(total_sentences / duration)} sentences/second")
