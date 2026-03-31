#!/usr/bin/env python3

from dataclasses import dataclass


@dataclass
class ModelHyperParameters:
    model_type: str
    epochs: int
    only_positive_bool_features: bool
    apply_label_constraints: bool
    min_abs_weight: float
    min_feat_updates: int
    quantize_bits: int | None
    make_label_dict: bool
    datetime: str
