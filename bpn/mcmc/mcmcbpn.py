#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright (c) 2011 Christopher D. Lasher
#
# This software is released under the MIT License. Please see
# LICENSE.txt for details.


"""A program to detect Process Linkage Networks using a Markov chain
Monte Carlo technique.

"""

# Set this to True if we need very detailed statements for debugging
# purposes
SUPERDEBUG_MODE = False
SUPERDEBUG = 5

# This controls how frequently the logger will broadcast the status on
# the percent of the steps completed during execution
BROADCAST_PERCENT = 1

BURN_IN = 20000
NUM_STEPS = 1000000

# If you alter this value, make sure you update the help string in
# cli.py!
import math
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


import collections
import itertools
import sys

from convutils import convutils

import bpn.cli

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


class AnnotatedInteractionsGraph(object):
    """A class that provides access to a mapping from process links
    (pairs of annotations) to the interactions which they co-annotate.

    A co-annotation is defined where, for two genes incident on an
    interaction edge, the first gene is annotated with one of the two
    processes, and the second gene is annotated with the other process.

    This class also provides information such as the number of gene-gene
    interactions, and which of those interactions are considered
    "active" according to a threshold.

    """
    def __init__(
            self,
            interaction_graph,
            annotations_dict,
            links_of_interest=None
        ):
        """Create a new instance.

        :Parameters:
        - `interaction_graph`: graph containing the gene-gene or gene
          product-gene product interactions
        - `annotations_dict`: a dictionary with annotation terms as keys
          and `set`s of genes as values
        - `links_of_interest`: a `set` of links in which the user is
          only interested; restricts the lookup keys to this set of
          interactions, potentially significantly reducing the memory
          usage. [NOTE: Each link's terms MUST be sorted alphabetically
          (e.g., `('term1', 'term2')` and NOT `('term2',
          'term1')`!]

        """
        self._interaction_graph = interaction_graph
        self._annotations_dict = annotations_dict
        # We'll use the following variable to cache the number of
        # interactions present, since this is apparently not cached by
        # the NetworkX Graph class.
        self._num_interactions = None
        # We use this dictionary for fast lookup of what interactions
        # are co-annotated by any given pair of annotation terms.
        self._annotations_to_interactions = collections.defaultdict(set)
        self._create_interaction_annotations(links_of_interest)
        self._num_annotation_pairs = None


    def _create_interaction_annotations(self, links_of_interest=None):
        """Convert all the node annotations into pair-wise annotations
        of interactions.

        :Parameters:
        - `links_of_interest`: a `set` of links in which the user is
          only interested; restricts the lookup keys to this set of
          interactions

        """
        total_num_interactions = self.calc_num_interactions()
        broadcast_percent_complete = 0
        for i, edge in enumerate(self._interaction_graph.edges_iter()):
            gene1_annotations = self._annotations_dict.get_item_keys(
                    edge[0])
            gene2_annotations = self._annotations_dict.get_item_keys(
                    edge[1])
            pairwise_combinations = itertools.product(gene1_annotations,
                    gene2_annotations)
            for gene1_annotation, gene2_annotation in \
                    pairwise_combinations:
                # If these are the same term, skip it.
                if gene1_annotation == gene2_annotation:
                    continue
                # We want to preserve alphabetical order of the
                # annotations.
                if gene1_annotation > gene2_annotation:
                    gene1_annotation, gene2_annotation = \
                            gene2_annotation, gene1_annotation
                link = (gene1_annotation, gene2_annotation)
                if links_of_interest is not None:
                    if link not in links_of_interest:
                        continue
                if SUPERDEBUG_MODE:
                    logger.log(SUPERDEBUG, "Adding interactions "
                            "for link %s" % (link,))
                self._annotations_to_interactions[(gene1_annotation,
                    gene2_annotation)].add(edge)

            percent_complete = int(100 * float(i + 1) /
                    total_num_interactions)
            if percent_complete >= (broadcast_percent_complete +
                    BROADCAST_PERCENT):
                broadcast_percent_complete = percent_complete
                logger.info("%d%% of interactions processed." % (
                        percent_complete))


    def get_all_links(self):
        """Returns a list of all the annotation pairs annotating the
        interactions.

        """
        return self._annotations_to_interactions.keys()


    def calc_num_links(self):
        """Returns the number of annotation pairs annotating the
        interactions.

        """
        if self._num_annotation_pairs is None:
            self._num_annotation_pairs = \
                    len(self._annotations_to_interactions)
        return self._num_annotation_pairs


    def calc_num_interactions(self):
        """Returns the total number of interactions."""
        if self._num_interactions is None:
            self._num_interactions = \
                    self._interaction_graph.number_of_edges()
        return self._num_interactions


    def get_interactions_annotated_by(self, annotation1, annotation2):
        """Returns a `set` of all interactions for which one adjacent
        gene is annotated with `annotation1`, and the other adjacent
        gene is annotated with `annotation2`.

        :Parameters:
        - `annotation1`: an annotation term
        - `annotation2`: an annotation term

        """
        if annotation1 > annotation2:
            annotation1, annotation2 = annotation2, annotation1
        interactions = self._annotations_to_interactions[(annotation1,
                annotation2)]
        return interactions


    def get_active_interactions(self, cutoff, greater=True):
        """Returns a `set` of all "active" interactions: those for which
        both incident genes pass a cutoff for differential gene
        expression.

        :Parameters:
        - `cutoff`: a numerical threshold value for determining whether
          a gene is active or not
        - `greater`: if `True`, consider a gene "active" if its
          differential expression value is greater than or equal to the
          `cutoff`; if `False`, consider a gene "active" if its value is
          less than or equal to the `cutoff`.

        """
        active_interactions = set()
        for edge in self._interaction_graph.edges_iter():
            gene1_expr = self._interaction_graph.node[edge[0]]['weight']
            gene2_expr = self._interaction_graph.node[edge[1]]['weight']
            if greater:
                if gene1_expr >= cutoff and gene2_expr >= cutoff:
                    active_interactions.add(edge)
            if not greater:
                if gene1_expr <= cutoff and gene2_expr <= cutoff:
                    active_interactions.add(edge)
        return active_interactions


class ShoveAnnotatedInteractionsGraph(AnnotatedInteractionsGraph):
    def __init__(
            self,
            interaction_graph,
            annotations_dict,
            store
        ):
        """Create a new instance.

        :Parameters:
        - `interaction_graph`: graph containing the gene-gene or gene
          product-gene product interactions
        - `annotations_dict`: a dictionary with annotation terms as keys and
          `set`s of genes as values
        - `store`: a `Shove` instance

        """
        self._interaction_graph = interaction_graph
        self._annotations_dict = annotations_dict
        self._store = store
        self._annotations_to_interactions = store


    def _create_interaction_annotations(self):
        """Convert all the node annotations into pair-wise annotations
        of interactions

        """
        for interaction in self._interaction_graph.edges_iter():
            gene1_annotations = self._annotations_dict[interaction[0]]
            gene2_annotations = self._annotations_dict[interaction[1]]
            annotation_pairs = itertools.product(gene1_annotations,
                    gene2_annotations)
            for annotation1, annotation2 in annotation_pairs:
                if annotation1 == annotation2:
                    continue
                if annotation1 > annotation2:
                    # Swap the annotations so they are in alphabetical
                    # order
                    annotation1, annotation2 = annotation2, annotation1
                # Get the interactions this pair of terms annotates
                annotated_interactions = \
                        self._annotations_to_interactions.get(
                                (annotation1, annotation2), set())
                # Add this interaction
                annotated_interactions.add(interaction)
                # Update the store
                self._annotations_to_interactions[
                        (annotation1, annotation2)] = \
                                annotated_interactions


class AnnotatedInteractionsArray(AnnotatedInteractionsGraph):
    """Similar to `AnnotatedInteractionsGraph`, however, it stores
    links, and their associated interactions, in linear arrays (lists),
    which are accessed by integer indices.

    """
    def __init__(
            self,
            interaction_graph,
            annotations_dict,
            links_of_interest=None
        ):
        """Create a new instance.

        :Parameters:
        - `interaction_graph`: graph containing the gene-gene or gene
          product-gene product interactions
        - `annotations_dict`: a dictionary with annotation terms as keys
          and `set`s of genes as values
        - `links_of_interest`: a `set` of links in which the user is
          only interested; restricts the lookup keys to this set of
          interactions, potentially significantly reducing the memory
          usage. [NOTE: Each link's terms MUST be sorted alphabetically
          (e.g., `('term1', 'term2')` and NOT `('term2',
          'term1')`!]

        """
        AnnotatedInteractionsGraph.__init__(
                self,
                interaction_graph,
                annotations_dict,
                links_of_interest
        )
        logger.info("Converting to more efficient data structures.")
        # Set two new attributes below, one being the list of all links,
        # and the other being the list of each link's corresponding
        # interactions, so that we can access them in linear time using
        # indices.
        #
        # Note that we rely on a property of Python dictionaries that,
        # so long as they are not modified between accesses, the lists
        # returned by dict.keys() and dict.values() correspond. See
        # http://docs.python.org/library/stdtypes.html#dict.items
        self._links, self._link_interactions = (
                self._annotations_to_interactions.keys(),
                self._annotations_to_interactions.values()
        )
        # Delete the dictionary mapping since we will not use it
        # hereafter.
        del self._annotations_to_interactions


    def get_all_links(self):
        """Returns a list of all the annotation pairs annotating the
        interactions.

        """
        return self._links


    def calc_num_links(self):
        """Returns the number of annotation pairs annotating the
        interactions.

        """
        if self._num_annotation_pairs is None:
            self._num_annotation_pairs = \
                    len(self._links)
        return self._num_annotation_pairs


    def get_interactions_annotated_by(self, link_index):
        """Returns a `set` of all interactions for which one adjacent
        gene is annotated with one term in the link and the other
        gene is annotated with the second term in the link.

        :Parameters:
        - `link_index`: the index of the link whose interactions are
          sought

        """
        return self._link_interactions[link_index]


def main(argv=None):
    cli_parser = bpn.cli.McmcCli()
    input_data = cli_parser.parse_args(argv)

    logger.info("Constructing supporting data structures; this may "
            "take a while...")
    annotated_interactions = AnnotatedInteractionsArray(
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

