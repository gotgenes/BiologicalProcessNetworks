#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright (c) 2011 Christopher D. Lasher
#
# This software is released under the MIT License. Please see
# LICENSE.txt for details.


"""A program to detect Process Linkage Networks using a Markov chain
Monte Carlo technique.

"""


import collections
import itertools
import sys

from convutils import convutils

import bpn.cli
import bpn.structures
from defaults import (
        SUPERDEBUG,
        SUPERDEBUG_MODE,
        LINKS_FIELDNAMES,
        PARAMETERS_FIELDNAMES,
        TRANSITIONS_FIELDNAMES,
        DETAILED_TRANSITIONS_FIELDNAMES
)

# Configure all the logging stuff
import logging
logger = logging.getLogger('bpn.mcmcbpn')

if SUPERDEBUG_MODE:
    # A logging level below logging.DEBUG
    logging.addLevelName(SUPERDEBUG, 'SUPERDEBUG')
    logger.setLevel(SUPERDEBUG)
    #stream_handler.setLevel(SUPERDEBUG)

import chains
import states
import recorders


def main(argv=None):
    cli_parser = bpn.cli.McmcCli()
    input_data = cli_parser.parse_args(argv)

    logger.info("Constructing supporting data structures; this may "
            "take a while...")
    annotated_interactions = bpn.structures.AnnotatedInteractionsArray(
            input_data.interaction_graph,
            input_data.annotations_dict
    )
    logger.info("Considering %d candidate links in total." %
            annotated_interactions.calc_num_links())

    logger.info("Constructing the Markov chain.")
    if input_data.free_parameters:
        logger.info("Using free parameter transitions.")
        parameters_state_class = states.RandomTransitionParametersState
    else:
        parameters_state_class = states.PLNParametersState
    if input_data.disable_swaps:
        logger.info("Disabling swap transitions.")
        links_state_class = states.NoSwapArrayLinksState
    else:
        links_state_class = states.ArrayLinksState
    if input_data.detailed_transitions:
        logger.info("Recording extra information for each state.")
        state_recorder_class = recorders.DetailedArrayStateRecorder
        transitions_csvfile = convutils.make_csv_dict_writer(
                input_data.transitions_outfile,
                DETAILED_TRANSITIONS_FIELDNAMES
        )
    else:
        state_recorder_class = recorders.ArrayStateRecorder
        transitions_csvfile = convutils.make_csv_dict_writer(
                input_data.transitions_outfile,
                TRANSITIONS_FIELDNAMES
        )
    markov_chain = chains.ArrayMarkovChain(
            annotated_interactions,
            input_data.activity_threshold,
            input_data.transition_ratio,
            num_steps=input_data.steps,
            burn_in=input_data.burn_in,
            state_recorder_class=state_recorder_class,
            parameters_state_class=parameters_state_class,
            links_state_class=links_state_class
    )
    logger.info("Beginning to run through states in the chain. This "
            "may take a while...")
    markov_chain.run()
    logger.info("Run completed.")

    logger.info("Writing link results to %s" %
            input_data.links_outfile.name)
    links_out_csvwriter = convutils.make_csv_dict_writer(
            input_data.links_outfile, LINKS_FIELDNAMES)
    markov_chain.state_recorder.write_links_probabilities(
            links_out_csvwriter)
    logger.info("Writing parameter results to %s" % (
            input_data.parameters_outfile.name))
    parameters_out_csvwriter = convutils.make_csv_dict_writer(
            input_data.parameters_outfile, PARAMETERS_FIELDNAMES)
    markov_chain.state_recorder.write_parameters_probabilities(
            parameters_out_csvwriter)
    logger.info("Writing transitions data to %s." % (
            input_data.transitions_outfile.name))
    markov_chain.state_recorder.write_transition_states(
            transitions_csvfile)
    logger.info("Finished.")


if __name__ == '__main__':
    main()

