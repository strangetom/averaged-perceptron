# Documentation

## Introduction

This documentation is intended to document and explain the Averaged Perceptron models implemented in this repository, their features and their variations. The intention is also to document the effect of model accuracy and execution performance of the different variations and features too.

The main motivation behind this to investigate ways of improving the accuracy of the [ingredient-parser](https://github.com/strangetom/ingredient-parser) library. This library uses a third party library ([python-crfsuite](https://github.com/scrapinghub/python-crfsuite)) to implement a Conditional Random Fields sequence labelling model, but the nature of using a third party library is that I am limited to the functionality it provides.

The main audience of this documentation is me, but in the future when I can't remember why I implemented something in a particular way or at all. 

Hopefully some of the information captured here will be useful for others too.

## Setting up the problem

The objective is a sequence labelling model that will label the tokens of a recipe ingredient sentence to indicate the part of the ingredient sentence to which the token belongs. 

The possible labels are as follows:

| Label      | Description                                                  |
| ---------- | ------------------------------------------------------------ |
| QTY        | Quantity of the ingredient (e.g. 200, 1-2).                  |
| UNIT       | Unit of a quantity for the ingredient (e.g. grams, cups).    |
| SIZE       | Physical size of the ingredient (e.g. large, small).         |
| PREP       | Preparation instructions for the ingredient (e.g. finely chopped). |
| PURPOSE    | Purpose of the ingredient (e.g. for garnish).                |
| PUNC       | Any punctuation tokens.                                      |
| B_NAME_TOK | The first token of an ingredient name.                       |
| I_NAME_TOK | A token within an ingredient name that is not the first token. |
| NAME_VAR   | A token that creates a variant of the ingredient name.<br />This is used in cases such as **beef or chicken stock**. **beef** and **chicken** are labelled with NAME_VAR as they indicate variations of the ingredient name **stock**. |
| NAME_MOD   | A token that modifies multiple ingredient names in the sentence.<br />For example in **dried apply and pears**, **dried** is labelled as NAME_MOD because it modified the two ingredient names, **apples** and **pears**. |
| NAME_SEP   | A token that separates different ingredient names and isn't PUNC, typically **or**. |
| COMMENT    | Additional information in the sentence that does not fall in one of the labels. |

This is the problem that is at the core of [ingredient-parser](https://github.com/strangetom/ingredient-parser) library. 

For the purposes of the implementations here, we will reuse the training data, tokenization and features used by the ingredient-parser library and concern ourselves only with the implementation of models to solve this sequence labelling problem.

## Approach

The basic approach is to implement an Averaged Perceptron model, which will be based on the the following blog post by Matthew Honnibal: [A Good Part-of-Speech Tagger in about 200 Lines of Python](https://explosion.ai/blog/part-of-speech-pos-tagger-in-python).

We will then investigate improvements various improvements to this.

* [Averaged Perceptron](averaged-perceptron.md) 
  * [Model optimisation](model-optimisation.md)
  * [Constrained transitions](constrained-transitions.md)
  * [Viterbi decoding](viterbi.md)
  * ...

Where possible, model accuracy results will be provided to show the relative improvement (or not) of each technique discussed. It will not necessarily be valid to compare the performance results across different techniques as there won't be any guarantee that the training conditions were the same when the results were generated. 
