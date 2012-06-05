#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright (c) 2011 Christopher D. Lasher
#
# This software is released under the MIT License. Please see
# LICENSE.txt for details.

# This module largely exists largely because I couldn't solve circular
# imports involved with mcmcbpn and other modules, particularly those
# above this package (e.g., cli.py).

"""Defaults for the MCMC BPN program."""

import math


# Set this to True if we need very detailed statements for debugging
# purposes
SUPERDEBUG_MODE = False
SUPERDEBUG = 5

# This controls how frequently the logger will broadcast the status on
# the percent of the steps completed during execution
BROADCAST_PERCENT = 1

BURN_IN = 1000000
NUM_STEPS = 10000000
TRANSITION_TYPE_RATIO = 0.9

TEMPERATURE = 100000
END_TEMPERATURE = 0.1

# If you alter this value, make sure you update the help string in
# cli.py!
ACTIVITY_THRESHOLD = -math.log10(0.05)

OUTPUT_BUFFER_SIZE = 1000
TRANSITIONS_BUFFER_SIZE = 100000

LINKS_FIELDNAMES = ('term1', 'term2', 'probability')
PARAMETERS_OUTFILE = 'parameter_results.tsv'
PARAMETERS_FIELDNAMES = ('parameter', 'value', 'probability')
TRANSITIONS_OUTTFILE = 'transitions.tsv'
TRANSITIONS_FIELDNAMES = [
        'transition_type',
        'log_transition_ratio',
        'log_state_likelihood',
        'accepted',
        'log_rejection_prob'
]
DETAILED_TRANSITIONS_FIELDNAMES = TRANSITIONS_FIELDNAMES + [
        'link_false_pos',
        'link_false_neg',
        'link_prior',
        'num_selected_links',
        'num_unselected_links',
        'num_selected_active_interactions',
        'num_selected_inactive_interactions',
        'num_unselected_active_interactions',
        'num_unselected_inactive_interactions'
]

FREQUENCIES_OUTFILE = 'state_frequencies.tsv'

TERMS_OUTFILE = 'terms_results.tsv'
TERMS_BASED_TRANSITIONS_FIELDNAMES = DETAILED_TRANSITIONS_FIELDNAMES[:]
TERMS_BASED_FIELDS = [
        'term_prior',
        'num_selected_terms',
        'num_unselected_terms'
]
TERMS_BASED_TRANSITIONS_FIELDNAMES[10:10] = TERMS_BASED_FIELDS[1:]
INDEPENDENT_TERMS_BASED_TRANSITIONS_FIELDNAMES = (
        TERMS_BASED_TRANSITIONS_FIELDNAMES[:])
INDEPENDENT_TERMS_BASED_TRANSITIONS_FIELDNAMES.insert(10,
        TERMS_BASED_FIELDS[0])

GENES_BASED_TRANSITIONS_FIELDNAMES = (
        INDEPENDENT_TERMS_BASED_TRANSITIONS_FIELDNAMES + [
            'num_selected_active_genes',
            'num_selected_inactive_genes',
            'num_unselected_active_genes',
            'num_unselected_inactive_genes'
        ])

GENES_BASED_TRANSITIONS_FIELDNAMES[10:10] = [
        'term_false_pos',
        'term_false_neg',
]

TERMS_FIELDNAMES = ('term', 'probability')
