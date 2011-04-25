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

BURN_IN = 20000
NUM_STEPS = 1000000
TEMPERATURE = 100000

# If you alter this value, make sure you update the help string in
# cli.py!
ACTIVITY_THRESHOLD = -math.log10(0.05)

LINKS_FIELDNAMES = ('term1', 'term2', 'probability')
PARAMETERS_OUTFILE = 'parameter_results.tsv'
PARAMETERS_FIELDNAMES = ('parameter', 'value', 'probability')
TRANSITIONS_OUTTFILE = 'transitions.tsv'
TRANSITIONS_FIELDNAMES = [
        'transition_type',
        'log_transition_ratio',
        'log_state_likelihood',
        'accepted'
]
DETAILED_TRANSITIONS_FIELDNAMES = TRANSITIONS_FIELDNAMES + [
        'alpha',
        'beta',
        'link_prior',
        'num_selected_links',
        'num_unselected_links',
        'num_selected_active_interactions',
        'num_selected_inactive_interactions',
        'num_unselected_active_interactions',
        'num_unselected_inactive_interactions'
]

TERMS_OUTFILE = 'terms_results.tsv'
TERMS_BASED_TRANSITIONS_FIELDNAMES = DETAILED_TRANSITIONS_FIELDNAMES[:]
TERMS_BASED_FIELDS = [
        'term_prior',
        'num_selected_terms',
        'num_unselected_terms'
]
TERMS_BASED_TRANSITIONS_FIELDNAMES.insert(6, TERMS_BASED_FIELDS[0])
TERMS_BASED_TRANSITIONS_FIELDNAMES[8:8] = TERMS_BASED_FIELDS[1:]

TERMS_FIELDNAMES = ('term', 'probability')
