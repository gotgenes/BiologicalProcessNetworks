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
import networkx

import bpn.cli
import bpn.structures
from defaults import (
        SUPERDEBUG,
        SUPERDEBUG_MODE,
        LINKS_FIELDNAMES,
        PARAMETERS_FIELDNAMES,
        TRANSITIONS_FIELDNAMES,
        DETAILED_TRANSITIONS_FIELDNAMES,
        TERMS_BASED_TRANSITIONS_FIELDNAMES,
        TERMS_FIELDNAMES
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


def check_link_components(annotated_interactions):
    num_terms = annotated_interactions.calc_num_terms()
    num_links = annotated_interactions.calc_num_links()
    potential_links_graph = networkx.Graph()
    potential_links_graph.add_edges_from(annotated_interactions.get_all_links())
    components = networkx.algorithms.components.connected_component_subgraphs(
            potential_links_graph)
    num_components = len(components)
    component_sizes = [(c.number_of_nodes(), c.number_of_edges()) for c in
            components]
    logger.info("{0} co-annotating pairs from {1} terms.".format(
            num_terms, num_links))
    logger.info(("Co-annotation network forms {0} connected "
        "component(s)").format(num_components))
    logger.info("Component sizes:\nNodes\tEdges\n{0}".format(
            '\n'.join('{0}\t{1}'.format(*sizes) for sizes in
                component_sizes))
    )
    if num_components > 1:
        logger.warning("WARNING! Potential links do not form a single"
                "connected component; algorithm may be unsuitable!")


def main(argv=None):
    cli_parser = bpn.cli.McmcCli()
    input_data = cli_parser.parse_args(argv)

    logger.info("Constructing supporting data structures; this may "
            "take a while...")
    if input_data.terms_based:
        if input_data.intraterms:
            annotated_interactions = (
                    bpn.structures.IntratermInteractions2dArray(
                            input_data.interactions_graph,
                            input_data.annotations_dict
                    )
            )
        else:
            annotated_interactions = (
                    bpn.structures.AnnotatedInteractions2dArray(
                            input_data.interactions_graph,
                            input_data.annotations_dict
                    )
            )
    else:
        annotated_interactions = (
                bpn.structures.AnnotatedInteractionsArray(
                        input_data.interactions_graph,
                        input_data.annotations_dict
                )
        )
    # Check to see whether the potential links form a single connected
    # component.
    check_link_components(annotated_interactions)

    logger.info("Constructing the Markov chain.")

    # Present the seed_links as indices.
    if input_data.seed_links:
        seed_links = [annotated_interactions.get_link_index(*link)
                for link in input_data.seed_links]
    else:
        seed_links = None

    if input_data.free_parameters:
        logger.info("Using free parameter transitions.")
        parameters_state_class = states.RandomTransitionParametersState
    else:
        parameters_state_class = states.PLNParametersState

    if input_data.terms_based:
        logger.info("Using terms-based model.")
        state_recorder_class = recorders.TermsBasedStateRecorder
        transitions_csvfile = convutils.make_csv_dict_writer(
                input_data.transitions_outfile,
                TERMS_BASED_TRANSITIONS_FIELDNAMES
        )
        if input_data.independent_terms:
            logger.info("Using independent-terms model.")
            links_state_class = states.IndependentTermsAndLinksState
        elif input_data.intraterms:
            logger.info("Considering intra-term interactions.")
            links_state_class = states.IntraTermsAndLinksState
        else:
            links_state_class = states.TermsAndLinksState
        markov_chain = chains.TermsBasedMarkovChain(
                annotated_interactions,
                input_data.activity_threshold,
                input_data.transition_ratio,
                num_steps=input_data.steps,
                burn_in=input_data.burn_in,
                state_recorder_class=state_recorder_class,
                links_state_class=links_state_class,
                seed_links_indices=seed_links
        )
    else:
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
                links_state_class=links_state_class,
                seed_links_indices=seed_links
        )

    logger.info("Beginning to run through states in the chain. This "
            "may take a while...")
    markov_chain.run()
    logger.info("Run completed.")

    logger.info("Writing link results to {0}".format(
            input_data.links_outfile.name))
    links_out_csvwriter = convutils.make_csv_dict_writer(
            input_data.links_outfile, LINKS_FIELDNAMES)
    markov_chain.state_recorder.write_links_probabilities(
            links_out_csvwriter)
    logger.info("Writing parameter results to {0}".format(
            input_data.parameters_outfile.name))
    parameters_out_csvwriter = convutils.make_csv_dict_writer(
            input_data.parameters_outfile, PARAMETERS_FIELDNAMES)
    markov_chain.state_recorder.write_parameters_probabilities(
            parameters_out_csvwriter)
    logger.info("Writing transitions data to {0}.".format(
            input_data.transitions_outfile.name))
    markov_chain.state_recorder.write_transition_states(
            transitions_csvfile)
    if input_data.terms_based:
        logger.info("Writing terms data to {0}.".format(
            input_data.terms_outfile.name))
        terms_out_csvwriter = convutils.make_csv_dict_writer(
                input_data.terms_outfile, TERMS_FIELDNAMES)
        markov_chain.state_recorder.write_terms_probabilities(
                terms_out_csvwriter)
    logger.info("Finished.")


if __name__ == '__main__':
    main()

