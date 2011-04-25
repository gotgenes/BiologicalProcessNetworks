#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright (c) 2011 Chris D. Lasher & Phillip Whisenhunt
#
# This software is released under the MIT License. Please see
# LICENSE.txt for details.


"""A program to detect Process Linkage Networks using
Simulated Annealing.

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
logger = logging.getLogger('bpn.sabpn')

if SUPERDEBUG_MODE:
    # A logging level below logging.DEBUG
    logging.addLevelName(SUPERDEBUG, 'SUPERDEBUG')
    logger.setLevel(SUPERDEBUG)
    #stream_handler.setLevel(SUPERDEBUG)

import simulatedannealing
import states
import recorders

def main(argv=None):
    cli_parser = bpn.cli.SaCli()
    input_data = cli_parser.parse_args(argv)

    logger.info("Constructing supporting data structures; this may "
            "take a while...")
    annotated_interactions = bpn.structures.AnnotatedInteractionsArray(
            input_data.interactions_graph,
            input_data.annotations_dict
    )
    logger.info("Considering %d candidate links in total." %
            annotated_interactions.calc_num_links())

    logger.info("Constructing Simulated Annealing")
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
        transitions_csvfile = convutils.make_csv_dict_writer(
                input_data.transitions_outfile,
                DETAILED_TRANSITIONS_FIELDNAMES
        )
    else:
        transitions_csvfile = convutils.make_csv_dict_writer(
                input_data.transitions_outfile,
                TRANSITIONS_FIELDNAMES
        )
    sa = simulatedannealing.ArraySimulatedAnnealing(
            annotated_interactions,
            input_data.activity_threshold,
            input_data.transition_ratio,
            num_steps=input_data.steps,
            temperature=input_data.temperature,
            end_temperature=input_data.end_temperature,
	    parameters_state_class=parameters_state_class,
            links_state_class=links_state_class
    )
    logger.info("Beginning to Anneal. This may take a while...")
    sa.run()
    logger.info("Run completed.")

    logger.info("Writing link results to %s" %
            input_data.links_outfile.name)
    links_out_csvwriter = convutils.make_csv_dict_writer(
            input_data.links_outfile, LINKS_FIELDNAMES)
    logger.info("Writing parameter results to %s" % (
            input_data.parameters_outfile.name))
    parameters_out_csvwriter = convutils.make_csv_dict_writer(
            input_data.parameters_outfile, PARAMETERS_FIELDNAMES)
    logger.info("Writing transitions data to %s." % (
            input_data.transitions_outfile.name))
    logger.info("Finished.")


if __name__ == '__main__':
    main()
