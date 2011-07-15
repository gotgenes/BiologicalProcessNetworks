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
import datetime
import itertools
import random
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
        TERMS_FIELDNAMES,
        INDEPENDENT_TERMS_BASED_TRANSITIONS_FIELDNAMES,
        GENES_BASED_TRANSITIONS_FIELDNAMES
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


def create_seed_value():
    """Creates a seed value for the `random` module, should one not be
    provided by the user.

    This code is simply a reproduction of the code from random.py in the
    standard library.

    Returns a seed value.

    """
    from binascii import hexlify as _hexlify
    from os import urandom as _urandom
    try:
        seed = long(_hexlify(_urandom(16)), 16)
    except NotImplementedError:
        import time
        seed = long(time.time() * 256) # use fractional seconds
    return seed


def check_link_components(annotated_interactions):
    logger.info("Checking components and sizes of potential BPN.")
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
            num_links, num_terms))
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
    starting_time = datetime.datetime.now()
    cli_parser = bpn.cli.McmcCli()
    input_data = cli_parser.parse_args(argv)

    logger.info("Constructing supporting data structures; this may "
            "take a while...")
    if input_data.terms_based:
        annotated_interactions = (
                bpn.structures.AnnotatedInteractions2dArray(
                        input_data.interactions_graph,
                        input_data.annotations_dict,
                        stringent_coannotations=input_data.stringent_coannotations
                )
        )
    else:
        annotated_interactions = (
                bpn.structures.AnnotatedInteractionsArray(
                        input_data.interactions_graph,
                        input_data.annotations_dict,
                        stringent_coannotations=input_data.stringent_coannotations
                )
        )
    # Check to see whether the potential links form a single connected
    # component.
    check_link_components(annotated_interactions)

    # TODO: check a command line option to see if the user input a seed;
    # for now, we'll just generate one all the time and report it.
    random_seed = create_seed_value()
    logger.info("The random seed value for this run is {0}.".format(
            random_seed))
    random.seed(random_seed)

    logger.info("Constructing the Markov chain.")
    # Prepare the CSV writers for the state recorder.
    links_out_csvwriter = convutils.make_csv_dict_writer(
            input_data.links_outfile, LINKS_FIELDNAMES)
    parameters_out_csvwriter = convutils.make_csv_dict_writer(
            input_data.parameters_outfile, PARAMETERS_FIELDNAMES)
    if input_data.terms_based:
        terms_out_csvwriter = convutils.make_csv_dict_writer(
                input_data.terms_outfile, TERMS_FIELDNAMES)

    # Present the seed_links as indices.
    if input_data.seed_links:
        seed_links = [annotated_interactions.get_link_index(*link)
                for link in input_data.seed_links]
    else:
        seed_links = None

    # Choose the appropriate parameters class.
    if input_data.fixed_distributions:
        logger.info("Using fixed distributions for all parameters.")
        if input_data.terms_based:
            parameters_state_class = states.FixedTermPriorParametersState
        else:
            parameters_state_class = (
                    states.FixedDistributionParametersState)
    elif input_data.terms_based:
        parameters_state_class = states.TermPriorParametersState
    elif input_data.free_parameters:
        logger.info("Using free parameter transitions.")
        parameters_state_class = states.RandomTransitionParametersState
    else:
        parameters_state_class = states.PLNParametersState

    if input_data.terms_based:
        logger.info("Using terms-based model.")

        if input_data.independent_terms or input_data.genes_based:
            if input_data.seed_terms:
                seed_terms = [
                        annotated_interactions.get_term_index(term) for
                        term in input_data.seed_terms
                ]
            else:
                seed_terms = None
        #else:
            #seed_terms = None

        if input_data.genes_based:
            if input_data.detailed_transitions:
                transitions_out_csvwriter = convutils.make_csv_dict_writer(
                        input_data.transitions_outfile,
                        GENES_BASED_TRANSITIONS_FIELDNAMES
                )
                state_recorder = (
                        recorders.DetailedGenesBasedStateRecorder(
                                annotated_interactions,
                                parameters_out_csvwriter,
                                links_out_csvwriter,
                                terms_out_csvwriter,
                                transitions_out_csvwriter
                        )
                )
            else:
                transitions_out_csvwriter = convutils.make_csv_dict_writer(
                        input_data.transitions_outfile,
                        TRANSITIONS_FIELDNAMES
                )
                state_recorder = recorders.TermsBasedStateRecorder(
                        annotated_interactions,
                        parameters_out_csvwriter,
                        links_out_csvwriter,
                        terms_out_csvwriter,
                        transitions_out_csvwriter
                )
            logger.info("Assessing term overlap through genes.")
            markov_chain = chains.GenesBasedMarkovChain(
                    state_recorder,
                    input_data.burn_in,
                    input_data.steps,
                    annotated_interactions,
                    input_data.activity_threshold,
                    transition_type_ratio=input_data.transition_ratio,
                    seed_terms_indices=seed_terms,
                    seed_links_indices=seed_links,
                    link_false_pos=input_data.link_false_pos,
                    link_false_neg=input_data.link_false_neg,
                    link_prior=input_data.link_prior,
                    term_false_pos=input_data.term_false_pos,
                    term_false_neg=input_data.term_false_neg,
                    term_prior=input_data.term_prior,
            )
        else:
            if input_data.independent_terms:
                if input_data.detailed_transitions:
                    transitions_out_csvwriter = (
                            convutils.make_csv_dict_writer(
                                input_data.transitions_outfile,
                                INDEPENDENT_TERMS_BASED_TRANSITIONS_FIELDNAMES
                            )
                    )
                    state_recorder = (
                            recorders.DetailedIndependentTermsBasedStateRecorder(
                                    annotated_interactions,
                                    parameters_out_csvwriter,
                                    links_out_csvwriter,
                                    terms_out_csvwriter,
                                    transitions_out_csvwriter
                            )
                    )
                else:
                    transitions_out_csvwriter = (
                            convutils.make_csv_dict_writer(
                                    input_data.transitions_outfile,
                                    TRANSITIONS_FIELDNAMES
                            )
                    )
                    state_recorder = (
                            recorders.TermsBasedStateRecorder(
                                    annotated_interactions,
                                    parameters_out_csvwriter,
                                    links_out_csvwriter,
                                    terms_out_csvwriter,
                                    transitions_out_csvwriter
                            )
                    )
                logger.info("Using independent-terms model.")
                markov_chain = chains.IndependentTermsBasedMarkovChain(
                        state_recorder,
                        input_data.burn_in,
                        input_data.steps,
                        annotated_interactions,
                        input_data.activity_threshold,
                        transition_type_ratio=input_data.transition_ratio,
                        seed_terms_indices=seed_terms,
                        seed_links_indices=seed_links,
                        link_false_pos=input_data.link_false_pos,
                        link_false_neg=input_data.link_false_neg,
                        link_prior=input_data.link_prior,
                        term_prior=input_data.term_prior,
                        parameters_state_class=parameters_state_class
                )
            else:
                if input_data.detailed_transitions:
                    transitions_out_csvwriter = (
                            convutils.make_csv_dict_writer(
                                    input_data.transitions_outfile,
                                    TERMS_BASED_TRANSITIONS_FIELDNAMES
                            )
                    )
                    state_recorder = (
                            recorders.DetailedTermsBasedStateRecorder(
                                    annotated_interactions,
                                    parameters_out_csvwriter,
                                    links_out_csvwriter,
                                    terms_out_csvwriter,
                                    transitions_out_csvwriter
                            )
                    )
                else:
                    transitions_out_csvwriter = (
                            convutils.make_csv_dict_writer(
                                    input_data.transitions_outfile,
                                    TRANSITIONS_FIELDNAMES
                            )
                    )
                    state_recorder = (
                            recorders.TermsBasedStateRecorder(
                                    annotated_interactions,
                                    parameters_out_csvwriter,
                                    links_out_csvwriter,
                                    terms_out_csvwriter,
                                    transitions_out_csvwriter
                            )
                    )
                if input_data.intraterms:
                    logger.info("Considering intra-term interactions.")
                    links_state_class = states.IntraTermsAndLinksState
                else:
                    links_state_class = states.TermsAndLinksState

                markov_chain = chains.TermsBasedMarkovChain(
                        state_recorder,
                        input_data.burn_in,
                        input_data.steps,
                        annotated_interactions,
                        input_data.activity_threshold,
                        transition_type_ratio=input_data.transition_ratio,
                        seed_links_indices=seed_links,
                        link_false_pos=input_data.link_false_pos,
                        link_false_neg=input_data.link_false_neg,
                        link_prior=input_data.link_prior,
                        term_prior=input_data.term_prior,
                        parameters_state_class=parameters_state_class,
                        links_state_class=links_state_class,
                )
    else:
        if input_data.disable_swaps:
            logger.info("Disabling swap transitions.")
            links_state_class = states.NoSwapArrayLinksState
        else:
            links_state_class = states.ArrayLinksState
        if input_data.detailed_transitions:
            logger.info("Recording extra information for each state.")
            transitions_out_csvwriter = convutils.make_csv_dict_writer(
                    input_data.transitions_outfile,
                    DETAILED_TRANSITIONS_FIELDNAMES
            )
            if input_data.record_frequencies:
                logger.info("Recording frequency information for each "
                "state.")
                state_recorder = recorders.FrequencyDetailedArrayStateRecorder(
                        annotated_interactions,
                        parameters_out_csvwriter,
                        links_out_csvwriter,
                        transitions_out_csvwriter
                )
            else:
                state_recorder = recorders.DetailedArrayStateRecorder(
                        annotated_interactions,
                        parameters_out_csvwriter,
                        links_out_csvwriter,
                        transitions_out_csvwriter
                )
        else:
            transitions_out_csvwriter = convutils.make_csv_dict_writer(
                    input_data.transitions_outfile,
                    TRANSITIONS_FIELDNAMES,
                    # TODO: This is a hack to force
                    # FrequencyDetailedArrayStateRecorder to work
                    # without the details transitions flag
                    extrasaction="ignore"
            )
            if input_data.record_frequencies:
                logger.info("Recording frequency information for each "
                "state.")
                state_recorder = recorders.FrequencyDetailedArrayStateRecorder(
                        annotated_interactions,
                        parameters_out_csvwriter,
                        links_out_csvwriter,
                        transitions_out_csvwriter
                )
            else:
                state_recorder = recorders.ArrayStateRecorder(
                    annotated_interactions,
                    parameters_out_csvwriter,
                    links_out_csvwriter,
                    transitions_out_csvwriter
                )
        markov_chain = chains.ArrayMarkovChain(
                state_recorder,
                input_data.burn_in,
                input_data.steps,
                annotated_interactions,
                input_data.activity_threshold,
                transition_type_ratio=input_data.transition_ratio,
                seed_links_indices=seed_links,
                link_false_pos=input_data.link_false_pos,
                link_false_neg=input_data.link_false_neg,
                link_prior=input_data.link_prior,
                parameters_state_class=parameters_state_class,
                links_state_class=links_state_class,
        )

    logger.debug("""\
Chain information:
    Chain class: {chain.__class__}
    Overall class: {chain.current_state.__class__}
    Links class: {chain.current_state.links_state.__class__}
    Parameters class: {chain.current_state.parameters_state.__class__}\
""".format(chain=markov_chain))
    logger.info("Beginning to run through states in the chain. This "
            "may take a while...")

    markov_chain.run()
    logger.info("Run completed.")

    logger.info("Writing link results to {0}".format(
            input_data.links_outfile.name))
    markov_chain.state_recorder.write_links_probabilities()
    logger.info("Writing parameter results to {0}".format(
            input_data.parameters_outfile.name))
    markov_chain.state_recorder.write_parameters_probabilities()
    if input_data.terms_based:
        logger.info("Writing terms data to {0}.".format(
            input_data.terms_outfile.name))
        markov_chain.state_recorder.write_terms_probabilities()
    markov_chain.state_recorder.write_transition_states()
    logger.info("Transitions data written to {0}.".format(
            input_data.transitions_outfile.name))
    if input_data.record_frequencies:
        logger.info("Writing state frequencies to {0}".format(
                input_data.frequencies_outfile.name))
        if "ArrayMarkovChain" in markov_chain.__class__.__name__:
            markov_chain.state_recorder.write_state_frequencies(
                    input_data.frequencies_outfile,
                    input_data.activity_threshold,
                    input_data.transition_ratio,
                    input_data.link_false_pos,
                    input_data.link_false_neg,
                    input_data.link_prior,
                    parameters_state_class,
                    links_state_class
            )
        logger.info("State frequencies written.")

    ending_time = datetime.datetime.now()
    logger.info("Finished.")
    running_time = ending_time - starting_time
    hours, remainder = divmod(running_time.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    hours += running_time.days * 24
    logger.info("Running time: {0}h {1}m {2}s".format(hours, minutes,
        seconds))

if __name__ == '__main__':
    main()

