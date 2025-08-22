#!/usr/bin/env python3

# Dict of illegal transitions.
# The key is the previous label, the values are the set of labels that cannot be
# predicted for the next label.
# This are generated from the training data: these transition never occur in the
# training data.
ILLEGAL_TRANSITIONS = {
    "B_NAME_TOK": {"B_NAME_TOK", "NAME_MOD"},
    "I_NAME_TOK": {"NAME_MOD"},
    "NAME_MOD": {"COMMENT", "I_NAME_TOK", "PURPOSE", "QTY", "UNIT"},
    "NAME_SEP": {"I_NAME_TOK", "PURPOSE"},
    "NAME_VAR": {"COMMENT", "I_NAME_TOK", "NAME_MOD", "PURPOSE", "QTY", "UNIT"},
    "PREP": {"NAME_SEP"},
    "PURPOSE": {
        "B_NAME_TOK",
        "I_NAME_TOK",
        "NAME_MOD",
        "NAME_SEP",
        "NAME_VAR",
        "PREP",
        "QTY",
        "SIZE",
        "UNIT",
    },
    "QTY": {"NAME_SEP", "PURPOSE"},
    "SIZE": {"NAME_SEP", "PURPOSE"},
}
