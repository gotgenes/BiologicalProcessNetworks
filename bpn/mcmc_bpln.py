#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright (c) 2011 Christopher D. Lasher
#
# This software is released under the MIT License. Please see
# LICENSE.txt for details.


"""A program to detect Process Linkage Networks using a Markov chain
Monte Carlo technique.

"""

__author__ = 'Chris Lasher'
__email__ = 'chris DOT lasher <AT> gmail DOT com'


import bisect
import collections
import copy
import itertools
import math
import random

from convutils import convutils
import numpy

import cli

# Configure all the logging stuff
import logging
logger = logging.getLogger('bpln.mcmc_bpln')

# Set this to True if we need very detailed statements for debugging
# purposes
SUPERDEBUG_MODE = False

if SUPERDEBUG_MODE:
    # A logging level below logging.DEBUG
    SUPERDEBUG = 5
    logging.addLevelName(SUPERDEBUG, 'SUPERDEBUG')
    logger.setLevel(SUPERDEBUG)
    #stream_handler.setLevel(SUPERDEBUG)

# This controls how frequently the logger will broadcast the status on
# the percent of the steps completed during execution
BROADCAST_PERCENT = 1

BURN_IN = 20000
NUM_STEPS = 1000000
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


class ParameterNotInDistributionError(ValueError):
    """Exception raised when a parameter is not available within the
    allowed distribution.

    """
    pass


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


class MarkovChain(object):
    pass


class State(object):
    """Represents a state in a Markov Chain"""

    def copy(self):
        """Create a copy of this state instance."""
        newcopy = copy.copy(self)
        return newcopy


    def create_new_state(self):
        """Creates a new state on the basis of this state instance."""
        raise NotImplementedError


class PLNParametersState(State):
    """Represents the possible parameter space for the likelihood
    function of the Process Linkage Network.

    """
    parameter_names = ('alpha', 'beta', 'link_prior')
    # I'm mostly placing these here to get pylint to stop complaining
    alpha = None
    beta = None
    link_prior = None
    # Now set the bounds for alpha (and beta)
    _alpha_min = 0
    _alpha_max = 1
    # This controls how many discrete settings we can choose from for
    # the false-positive rate
    _size_alpha_distribution = 19
    _alpha_distribution = [0.05 * k for k in range(1,
            _size_alpha_distribution + 1)]
    if SUPERDEBUG_MODE:
        logger.log(SUPERDEBUG, "Alpha distribution: %s" %
                _alpha_distribution)
    _beta_distribution = _alpha_distribution
    _beta_min = _alpha_min
    _beta_max = _alpha_max
    # This controls the maximum number of discrete settings we can
    # choose from for the link prior rate
    _size_link_prior_distribution = 20
    _link_prior_max = 0.5

    def __init__(
            self,
            number_of_links,
            alpha=None,
            beta=None,
            link_prior=None
        ):
        """Create a new instance.

        :Parameters:
        - `number_of_links`: the total number of links being considered
        - `alpha`: the false-positive rate, the portion of gene-gene
          interactions which were included, but shouldn't have been
        - `beta`: the false-negative rate, the portion of gene-gene
          interactions which weren't included, but should have been
        - `link_prior`: the assumed probability we would pick any one
          link as being active

        """
        # We must first set up the link prior distribution; this cannot
        # be known beforehand, because it depends on knowing the number
        # links beforehand.
        if number_of_links >= self._size_link_prior_distribution:
            self._link_prior_distribution = [
                    float(k) / number_of_links for k in range(1,
                        self._size_link_prior_distribution + 1)
            ]
        else:
            self._link_prior_distribution = []
            for k in range(1, number_of_links):
                value = float(k) / number_of_links
                if value <= self._link_prior_max:
                    self._link_prior_distribution.append(value)
                else:
                    break

        if SUPERDEBUG_MODE:
            logger.log(SUPERDEBUG, "link_prior_distribution: %s" %
                    self._link_prior_distribution)

        # We know the entire parameter space, now.
        self.parameter_space_size = len(self._alpha_distribution) + \
                len(self._beta_distribution) + \
                len(self._link_prior_distribution)

        # Set all parameters, if not set already, and validate them.
        self._set_parameters_at_init(alpha, beta, link_prior)
        logger.debug("Initial parameter settings: "
                "alpha=%s, beta=%s, link_prior=%s" % (self.alpha,
                    self.beta, self.link_prior)
        )

        # This variable is used to store the previous state that the
        # current state arrived from. When set, it should be a tuple
        # containing the name of the parameter, and the index in the
        # distribution that it had previous to this state.
        self._delta = None


    def _set_parameters_at_init(self, alpha, beta, link_prior):
        """A helper function to verify and set the parameters at
        instantiation.

        :Parameters:
        - `alpha`: the false-positive rate, the portion of gene-gene
          interactions which were included, but shouldn't have been
        - `beta`: the false-negative rate, the portion of gene-gene
          interactions which weren't included, but should have been
        - `link_prior`: the assumed probability we would pick any one
          link as being active

        """
        given_values = locals()
        for param_name in self.parameter_names:
            value = given_values[param_name]
            param_distribution = getattr(self, '_%s_distribution' %
                    param_name)
            param_index_name = '_%s_index' % param_name
            if value is None:
                # The user did not define the value ahead of time;
                # select one randomly
                rand_index = random.randint(0,
                        len(param_distribution) - 1)
                # Set the index
                setattr(self, param_index_name, rand_index)
                # Set the value of the parameter
                value = param_distribution[rand_index]
                setattr(self, param_name, value)
            else:
                # The user defined this parameter
                try:
                    # Get the index for this value and set it for the
                    # instance
                    index = param_distribution.index(value)
                    setattr(self, param_index_name, index)
                except ValueError:
                    # The user gave a value for the parameter that is
                    # not available in the distribution; re-raise the
                    # exception and notify the user to supply a better
                    # value
                    error_message = ("The value of %s for %s is "
                            "not available in the distribution" % (
                                value, param_name)
                    )
                    raise ParameterNotInDistributionError(error_message)
                # Set the value of the parameter
                setattr(self, param_name, value)


    def _get_parameter_neighboring_indices(self, parameter_name):
        """Returns the indices for the possible neighboring values for
        `parameter`

        :Parameters:
        - `parameter_name`: the name of the parameter, either `'alpha'`,
          `'beta'`, or `'link_prior'`

        """
        param_index = getattr(self, '_%s_index' % parameter_name)
        param_distribution = getattr(self, '_%s_distribution' %
                parameter_name)
        if param_index == 0:
            # The current index is at the minimum end of the
            # distribution; return only the next, higher index
            neighboring_indices = (1,)
        elif param_index == len(param_distribution) - 1:
            # The current index is at the maximum end of the
            # distribution; return only the next, lower index
            neighboring_indices = (param_index - 1,)
        else:
            # The current index isn't at either end of the
            # distribution; return the immediate smaller and greater
            # indices
            neighboring_indices = (param_index - 1, param_index + 1)
        return neighboring_indices


    def _get_all_parameter_neighboring_indices(self):
        """Returns a dictionary where the keys are the names of the
        parameters and the values are the neighboring indices of the
        parameters' current indices.

        """
        all_neighboring_indices = {}
        for param in self.parameter_names:
            all_neighboring_indices[param] = \
                    self._get_parameter_neighboring_indices(param)
        return all_neighboring_indices


    def get_parameter_distributions(self):
        """Returns a dictionary with the parameter names as keys and
        their distribution of possible values as values.

        """
        parameter_distributions = {}
        for param_name in self.parameter_names:
            parameter_distributions[param_name] = getattr(
                    self, '_%s_distribution' % param_name)
        return parameter_distributions


    def _calc_num_neighbors_per_parameter(self):
        """Calculates the number of neighboring values there are for
        each parameter.

        Returns a dictionary with the parameter names as keys and the
        number of neighboring values as values.

        """
        num_neighbors_per_parameter = {}
        neighboring_indices = self._get_all_parameter_neighboring_indices()
        for param_name, indices in neighboring_indices.items():
            num_neighbors_per_parameter[param_name] = len(indices)
        return num_neighbors_per_parameter


    def calc_num_neighboring_states(self):
        """Returns the count of the number of parameter states
        neighboring this one.

        """
        num_neighboring_states = sum(
                self._calc_num_neighbors_per_parameter().values())
        return num_neighboring_states


    def _construct_param_selection_cutoffs(self):
        """Creates a list containing the cutoffs to be used for random
        selection of the parameter whose value will be changed.

        """
        num_neighbors_per_param = \
                self._calc_num_neighbors_per_parameter()
        # Although we could call self._calc_num_neighboring states, it
        # is more efficient to just sum those per neighbor, so we avoid
        # looping twice
        total_num_neighbors = sum(num_neighbors_per_param.values())
        running_total = 0
        cutoffs = []
        for param_name in self.parameter_names:
            num_neighbors = num_neighbors_per_param[param_name]
            running_total += float(num_neighbors) / total_num_neighbors
            if running_total > 1:
                running_total = 1
            cutoffs.append(running_total)
        # The last cutoff should be at 1, so that we don't try and
        # index outside of the distribution
        cutoffs[-1] = 1
        if SUPERDEBUG_MODE:
            logger.log(SUPERDEBUG, "Parameter selection random "
                    "distribution: %s" % (cutoffs,))
        return cutoffs


    def _choose_parameter_to_alter(self):
        """Returns a string providing the name of the parameter to
        alter.

        """
        # We need to establish the distribution cutoffs for selecting
        # which parameter to alter for the new state
        cutoffs = self._construct_param_selection_cutoffs()
        # Now flip a coin to decide which parameter to change
        param_coin = random.random()
        if SUPERDEBUG_MODE:
            logger.log(SUPERDEBUG, "Parameter coin: %s" % param_coin)
        cutoff_index = bisect.bisect_left(cutoffs, param_coin)
        parameter_to_alter = self.parameter_names[cutoff_index]
        return parameter_to_alter


    def create_new_state(self):
        """Creates a new state on the basis of this state instance."""
        new_state = self.copy()
        parameter_to_alter = self._choose_parameter_to_alter()
        # Now flip another coin to decide which new value the parameter
        # should get (by picking the next index to use)
        neighboring_indices = self._get_parameter_neighboring_indices(
                parameter_to_alter)
        current_index = getattr(self, '_%s_index' % parameter_to_alter)
        new_index = random.choice(neighboring_indices)
        # Now set the parameter and its index in the new state
        setattr(new_state, '_%s_index' % parameter_to_alter,
                new_index)
        new_value = getattr(self, '_%s_distribution' %
                parameter_to_alter)[new_index]
        setattr(new_state, parameter_to_alter, new_value)
        new_state._delta = (parameter_to_alter, current_index)
        logger.debug("Changing parameter %s from %s to %s\n"
                "New state: alpha=%s, beta=%s, link_prior=%s" % (
                parameter_to_alter, getattr(self, parameter_to_alter),
                new_value, new_state.alpha, new_state.beta,
                new_state.link_prior)
        )
        return new_state


class RandomTransitionParametersState(PLNParametersState):
    """Similar to `PLNParametersState`, however, parameters are allowed
    to transition randomly to any other value in the distribution,
    rather than being restricted on transitioning to neighboring values.

    """
    def __init__(
            self,
            number_of_links,
            alpha=None,
            beta=None,
            link_prior=None
        ):
        """Create a new instance.

        :Parameters:
        - `number_of_links`: the total number of links being considered
        - `alpha`: the false-positive rate, the portion of gene-gene
          interactions which were included, but shouldn't have been
        - `beta`: the false-negative rate, the portion of gene-gene
          interactions which weren't included, but should have been
        - `link_prior`: the assumed probability we would pick any one
          link as being active

        """
        PLNParametersState.__init__(
            self,
            number_of_links,
            alpha,
            beta,
            link_prior
        )
        # We'll use this as a cache of the cutoffs, since it will never
        # change
        self._parameter_selection_cutoffs = None


    def _construct_param_selection_cutoffs(self):
        """Creates a list containing the cutoffs to be used for random
        selection of the parameter whose value will be changed.

        """
        if self._parameter_selection_cutoffs is None:
            if SUPERDEBUG_MODE:
                logger.log(SUPERDEBUG, "Caching parameter selection "
                        "cutoffs.")
            cutoffs = PLNParametersState._construct_param_selection_cutoffs(self)
            self._parameter_selection_cutoffs = cutoffs
        return self._parameter_selection_cutoffs


    def _calc_num_neighbors_per_parameter(self):
        """Calculates the number of neighboring values there are for
        each parameter.

        Returns a dictionary with the parameter names as keys and the
        number of neighboring values as values.

        """
        num_neighbors_per_parameter = {}
        for param_name in self.parameter_names:
            num_other_values_in_distribution = \
                    len(getattr(self, '_%s_distribution' % param_name)) - 1
            num_neighbors_per_parameter[param_name] = \
                    num_other_values_in_distribution
        return num_neighbors_per_parameter


    def create_new_state(self):
        """Creates a new state on the basis of this state instance."""
        new_state = self.copy()
        parameter_to_alter = self._choose_parameter_to_alter()
        parameter_distribution = getattr(self,
                '_%s_distribution' % parameter_to_alter)
        size_parameter_distribution = len(parameter_distribution)
        current_index = getattr(self, '_%s_index' % parameter_to_alter)
        # Roll a die to get the new parameter index and value.
        new_index = random.randint(0, size_parameter_distribution - 1)
        # We could by random chance end up with the same index (and
        # value) as the current one; keep sampling until we have a
        # different index.
        while new_index == current_index:
            new_index = random.randint(0, size_parameter_distribution - 1)
        # Now set the parameter and its index in the new state
        setattr(new_state, '_%s_index' % parameter_to_alter,
                new_index)
        new_value = parameter_distribution[new_index]
        setattr(new_state, parameter_to_alter, new_value)
        new_state._delta = (parameter_to_alter, current_index)
        logger.debug("Changing parameter %s from %s to %s\n"
                "New state: alpha=%s, beta=%s, link_prior=%s" % (
                parameter_to_alter, getattr(self, parameter_to_alter),
                new_value, new_state.alpha, new_state.beta,
                new_state.link_prior)
        )
        return new_state


class PLNLinksState(State):
    def __init__(
            self,
            process_links,
            selected_links,
            annotated_interactions,
            active_interactions
        ):
        """Create a new PLNLinksState instance

        :Parameters:
        - `process_links`: the set of all possible links between
          biological processes
        - `selected_links`: the subset of links being considered as
          "selected" initially in the process linkage network
        - `annotated_interactions`: an `AnnotatedInteractionsGraph`
          instance
        - `active_interactions`: a set of interactions that are
          considered "active"

        """
        self.process_links = process_links
        self._annotated_interactions = annotated_interactions
        self._active_interactions = active_interactions
        # _interaction_selection_counts maintains the number of times a
        # gene-gene interaction has been "covered" by the selected
        # process links
        self._interaction_selection_counts = collections.defaultdict(
                int)
        self._num_selected_active_interactions = 0

        # This variable is used to store the previous state that the
        # current state arrived from. When set, it should be a tuple
        # where the first item is a string describing the type of
        # transition, and the second item is the link or pair of links
        # used in the transition. Acceptable strings for first item are:
        # 'selection', 'unselection', and 'swap'. For 'selection' and
        # 'unselection', the second item is a tuple of (annotation1,
        # annotation2); for 'swap', the second item is a tuple of links
        # ((annotation1, annotation2), (annotation3, annotation4))
        #
        # If self._delta is None, it indicates this is an origin state
        # with no ancestor.
        self._delta = None

        # Now get the appropriate tallies for the selected interactions
        self.unselected_links = set(self.process_links)
        self.selected_links = set()
        for annotation1, annotation2 in selected_links:
            self.select_link(annotation1, annotation2)
        # We need to return the delta to None
        self._delta = None


    def calc_num_links(self):
        """Returns thet number of possible links."""
        return len(self.process_links)


    def calc_num_selected_links(self):
        """Returns the number of links that have been selected."""
        return len(self.selected_links)


    def calc_num_unselected_links(self):
        """Returns the number of links that have not been selected."""
        return len(self.unselected_links)


    def calc_num_neighboring_states(self):
        """Calculates the number of possible neighboring states."""
        num_selected_links = self.calc_num_selected_links()
        num_unselected_links = self.calc_num_unselected_links()
        total_num_links = num_selected_links + num_unselected_links
        num_neighboring_states = total_num_links + (num_selected_links *
                num_unselected_links)
        return num_neighboring_states


    def select_link(self, annotation1, annotation2):
        """Add a link to the set of selected links.

        :Parameters:
        - `annotation1`: the first annotation of the link
        - `annotation2`: the second annotation of the link

        """
        if annotation1 > annotation2:
            annotation1, annotation2 = annotation2, annotation1
        logger.debug("Selecting link (%s, %s)" % (annotation1,
                annotation2))
        if (annotation1, annotation2) in self.selected_links:
            raise ValueError("The link (%s, %s) has already been selected."
                    % (annotation1, annotation2))
        self.selected_links.add((annotation1, annotation2))
        self.unselected_links.remove((annotation1, annotation2))
        link_annotated_interactions = \
            self._annotated_interactions.get_interactions_annotated_by(
                    annotation1, annotation2)
        for interaction in link_annotated_interactions:
            self._interaction_selection_counts[interaction] += 1
            # If this is the first time selecting this interaction,
            # and this interaction is noted as active, increment the
            # active count
            if (self._interaction_selection_counts[interaction] == 1) \
                    and (interaction in self._active_interactions):
                self._num_selected_active_interactions += 1
        # Finally, we note in the delta this selection
        self._delta = ('selection', (annotation1, annotation2))


    def unselect_link(self, annotation1, annotation2):
        """Remove a link from the set of selected links.

        :Parameters:
        - `annotation1`: the first annotation of the link
        - `annotation2`: the second annotation of the link

        """
        if annotation1 > annotation2:
            annotation1, annotation2 = annotation2, annotation1
        logger.debug("Unselecting link (%s, %s)" % (annotation1,
                annotation2))
        if (annotation1, annotation2) not in self.selected_links:
            raise ValueError("The link (%s, %s) has not been selected."
                    % annotation1, annotation2)
        self.unselected_links.add((annotation1, annotation2))
        self.selected_links.remove((annotation1, annotation2))
        link_annotated_interactions = \
            self._annotated_interactions.get_interactions_annotated_by(
                    annotation1, annotation2)
        for interaction in link_annotated_interactions:
            self._interaction_selection_counts[interaction] -= 1
            # If we have removed the only link which selected this
            # interaction, remove the interaction from the dictionary
            if not self._interaction_selection_counts[interaction]:
                del self._interaction_selection_counts[interaction]
                # Further, if it was an active interaction, deduct from
                # the count of selected active interactions
                if interaction in self._active_interactions:
                    self._num_selected_active_interactions -= 1
        # Finally, we note in the delta this unselection
        self._delta = ('unselection', (annotation1, annotation2))


    def swap_links(self, selected_link, unselected_link):
        """Swap an selected link for an unselected link

        Unselects the selected link and selects the unselected link.

        :Parameters:
        - `selected_link`: a tuple of two annotation terms representing
          the process link to be unselected
        - `unselected_link`: a tuple of two annotation terms
          representing the process link to selected

        """
        selected_annotation1, selected_annotation2 = selected_link
        if selected_annotation1 > selected_annotation2:
            selected_annotation1, selected_annotation2 = \
                    selected_annotation2, selected_annotation1
        selected_link = (selected_annotation1, selected_annotation2)
        unselected_annotation1, unselected_annotation2 = unselected_link
        if unselected_annotation1 > unselected_annotation2:
            unselected_annotation1, unselected_annotation2 = \
                    unselected_annotation2, unselected_annotation1
        unselected_link = (unselected_annotation1, unselected_annotation2)
        logger.debug("Swapping selected link (%s, %s) for unselected "
                "link (%s, %s)" % (selected_annotation1,
                    selected_annotation2, unselected_annotation1,
                    unselected_annotation2)
        )
        # If these two links are the same, raise an error
        if selected_link == unselected_link:
            raise ValueError("Links are same (%s, %s)" % selected_link)

        self.select_link(unselected_annotation1, unselected_annotation2)
        self.unselect_link(selected_annotation1, selected_annotation2)
        self._delta = ('swap', (selected_link, unselected_link))


    def copy(self):
        """Create a copy of this state instance."""
        newcopy = copy.copy(self)
        # We need to make a separate (shallow) copy of the selected
        # links structure, so that when the copy updates its selected
        # links, it doesn't have the side-effect of affecting this
        # instance's structure.
        newcopy.selected_links = copy.copy(self.selected_links)
        newcopy.unselected_links = copy.copy(self.unselected_links)
        # Similarly with the selection counts for the interactions
        newcopy._interaction_selection_counts = copy.copy(
                self._interaction_selection_counts)
        return newcopy


    def calc_transition_cutoffs(self):
        """Calculate the appropriate cutoffs for the possible
        transitions.

        Returns a tuple where the first item is the probability cutoff
        for using a swap transition, the second item is the probability
        cutoff for using a selection transition, and the third item is
        the probability cutoff for an unselection transition.

        """
        # First, we get the individual probabilities of each type of
        # transition as its percentage of the total number of
        # transitions
        total_transitions = self.calc_num_neighboring_states()
        # The swap transition will have the highest probability, so we
        # calculate it first in the distribution.
        # The probability of making a swap transition depends on
        # both the number of selected and unselected links
        num_selected_links = self.calc_num_selected_links()
        num_unselected_links = self.calc_num_unselected_links()
        swap_prob = float(num_selected_links * num_unselected_links) / \
                total_transitions
        # The probability of making a selection transition depends on
        # the number of links presently unselected
        selection_prob = float(num_unselected_links) / total_transitions
        # Since all probabilities should add up to 1, we don't need to
        # calculate the unselection probability.

        # Now create the cutoffs
        transition_cutoffs = (
                swap_prob,
                swap_prob + selection_prob,
                1
        )
        if SUPERDEBUG_MODE:
            logger.log(SUPERDEBUG, ("Transition choice distribution: "
                    "%s" % (transition_cutoffs,)))
        return transition_cutoffs


    def choose_random_selected_link(self):
        """Chooses a selected link uniformly at random and returns it.

        """
        selected_link = random.choice(list(self.selected_links))
        return selected_link


    def choose_random_unselected_link(self):
        """Chooses an unselected link uniformly at random and returns
        it.

        """
        unselected_link = random.choice(list(self.unselected_links))
        return unselected_link


    def create_new_state(self):
        """Creates a new state on the basis of this state instance."""
        logger.debug("Creating a new links state.")
        # First, get an identical copy of this state
        new_state = self.copy()
        # Then decide which type of transition we're going to do
        transition_cutoffs = self.calc_transition_cutoffs()
        transition_dice = random.random()
        if SUPERDEBUG_MODE:
            logger.log(SUPERDEBUG, "Transition dice was %s" % (
                    transition_dice))
        if transition_dice <= transition_cutoffs[0]:
            # Swap links
            logger.debug("Chose swap transition.")
            selected_link = self.choose_random_selected_link()
            unselected_link = self.choose_random_unselected_link()
            new_state.swap_links(selected_link, unselected_link)
        elif transition_dice <= transition_cutoffs[1]:
            # Select an unselected link
            logger.debug("Chose selection transition.")
            unselected_link = self.choose_random_unselected_link()
            new_state.select_link(*unselected_link)
        else:
            # Unselect a selected link
            logger.debug("Chose unselection transition.")
            selected_link = self.choose_random_selected_link()
            new_state.unselect_link(*selected_link)

        return new_state


    def calc_num_selected_interactions(self):
        """Returns the number of interactions covered by at least one
        selected link.

        """
        # The keys to _interaction_selection_counts are those
        # interactions which have been covered by at least one selected
        # process link
        return len(self._interaction_selection_counts)


    def calc_num_unselected_interactions(self):
        """Returns the number of interactions covered by no selected
        link.

        """
        return self._annotated_interactions.calc_num_interactions() -\
                self.calc_num_selected_interactions()


    def calc_num_selected_active_interactions(self):
        """Returns the number of active interactions covered by at least
        one selected link.

        """
        return self._num_selected_active_interactions


    def calc_num_unselected_active_interactions(self):
        """Returns the number of active interactions covered by no
        selected link.

        """
        return len(self._active_interactions) -\
                self._num_selected_active_interactions


    def calc_num_selected_inactive_interactions(self):
        """Returns the number of inactive interactions covered by at
        least one selected link.

        """
        return self.calc_num_selected_interactions() -\
                self.calc_num_selected_active_interactions()


    def calc_num_unselected_inactive_interactions(self):
        """Returns the number of inactive interactions covered by no
        selected link.

        """
        return self.calc_num_unselected_interactions() -\
                self.calc_num_unselected_active_interactions()


class ArrayLinksState(PLNLinksState):
    """Similar to `PLNLinksState`, however, uses array data structures
    to track which links are selected and unselected.

    """
    def __init__(
            self,
            annotated_interactions,
            selected_links_indices,
            active_interactions
        ):
        """Create a new PLNLinksState instance

        :Parameters:
        - `annotated_interactions`: an `AnnotatedInteractionsArray`
          instance
        - `selected_links_indices`: indices for the subset of links
          being considered as "selected" initially in the process
          linkage network
        - `active_interactions`: a set of interactions that are
          considered "active"

        """
        self._annotated_interactions = annotated_interactions
        self._num_links = annotated_interactions.calc_num_links()
        self._num_selected_links = 0
        self._active_interactions = active_interactions
        self._interaction_selection_counts = collections.defaultdict(
                int)
        self._num_selected_active_interactions = 0
        self._delta = None
        # _link_selections is a 1-dimensional `numpy.ndarray` of boolean
        # data type, where the value at each index is `True` if the link
        # represented by that index has been selected, or `False` if the
        # link is unselected.
        self.link_selections = numpy.zeros(self._num_links, numpy.bool)
        for index in selected_links_indices:
            self.select_link(index)
        self._delta = None


    def select_link(self, index):
        """Mark a link as selected.

        :Parameters:
        - `index`: the index of the link to mark as selected

        """
        if self.link_selections[index]:
            raise ValueError("The link at index %d has already been "
                    "marked selected." % index)
        self.link_selections[index] = True
        self._num_selected_links += 1
        link_annotated_interactions = \
            self._annotated_interactions.get_interactions_annotated_by(
                    index)
        for interaction in link_annotated_interactions:
            self._interaction_selection_counts[interaction] += 1
            # If this is the first time selecting this interaction,
            # and this interaction is noted as active, increment the
            # active count
            if (self._interaction_selection_counts[interaction] == 1) \
                    and (interaction in self._active_interactions):
                self._num_selected_active_interactions += 1
        # Finally, we note in the delta this selection
        self._delta = ('selection', index)


    def unselect_link(self, index):
        """Mark a link as unselected.

        :Parameters:
        - `index`: the index of the link to mark as unselected

        """
        if not self.link_selections[index]:
            raise ValueError("The link at index %d has already been "
                    "marked unselected." % index)
        self.link_selections[index] = False
        self._num_selected_links -= 1
        link_annotated_interactions = \
            self._annotated_interactions.get_interactions_annotated_by(
                    index)
        for interaction in link_annotated_interactions:
            self._interaction_selection_counts[interaction] -= 1
            # If we have removed the only link which selected this
            # interaction, remove the interaction from the dictionary
            if not self._interaction_selection_counts[interaction]:
                del self._interaction_selection_counts[interaction]
                # Further, if it was an active interaction, deduct from
                # the count of selected active interactions
                if interaction in self._active_interactions:
                    self._num_selected_active_interactions -= 1
        # Finally, we note in the delta this unselection
        self._delta = ('unselection', index)


    def swap_links(self, selected_link_index, unselected_link_index):
        """Swap an selected link for an unselected link

        Unselects the selected link and selects the unselected link.

        :Parameters:
        - `selected_link_index`: index of the selected link to be
          unselected
        - `unselected_link_index`: index of the unselected link to be
          selected

        """
        # If these two links are the same, raise an error
        if selected_link_index == unselected_link_index:
            raise ValueError("Same link indices given (%d)"
                    % selected_link_index)

        self.select_link(unselected_link_index)
        self.unselect_link(selected_link_index)
        self._delta = ('swap', (selected_link_index,
            unselected_link_index))


    def copy(self):
        """Create a copy of this state instance."""
        newcopy = copy.copy(self)
        newcopy.link_selections = self.link_selections.copy()
        newcopy._interaction_selection_counts = \
                self._interaction_selection_counts.copy()
        return newcopy


    def calc_num_links(self):
        """Returns thet number of possible links."""
        return self._num_links


    def calc_num_selected_links(self):
        """Returns the number of links that have been selected."""
        return self._num_selected_links


    def calc_num_unselected_links(self):
        """Returns the number of links that have not been selected."""
        return self._num_links - self._num_selected_links


    def calc_num_neighboring_states(self):
        """Calculates the number of possible neighboring states."""
        num_selected_links = self.calc_num_selected_links()
        num_unselected_links = self.calc_num_unselected_links()
        total_num_links = num_selected_links + num_unselected_links
        num_neighboring_states = total_num_links + (num_selected_links *
                num_unselected_links)
        return num_neighboring_states


    def choose_random_selected_link(self):
        """Chooses the index of a selected link uniformly at random and
        returns it.

        """
        # This returns all the indices where value in the array is
        # `True`.
        indices_of_selected = numpy.nonzero(self.link_selections)[0]
        selected_link_index = random.choice(indices_of_selected)
        return selected_link_index


    def choose_random_unselected_link(self):
        """Chooses the index of an unselected link uniformly at random
        and returns it.

        """
        # By inverting the values on the matrix, we get the indices
        # where the values are `False` in the original.
        indices_of_unselected = numpy.nonzero(
                -self.link_selections)[0]
        unselected_link_index = random.choice(indices_of_unselected)
        return unselected_link_index


    def create_new_state(self):
        """Creates a new state on the basis of this state instance."""
        logger.debug("Creating a new links state.")
        # First, get an identical copy of this state
        new_state = self.copy()
        # Then decide which type of transition we're going to do
        transition_cutoffs = self.calc_transition_cutoffs()
        transition_dice = random.random()
        if SUPERDEBUG_MODE:
            logger.log(SUPERDEBUG, "Transition dice was %s" % (
                    transition_dice))
        if transition_dice <= transition_cutoffs[0]:
            # Swap links
            logger.debug("Chose swap transition.")
            selected_link_index = self.choose_random_selected_link()
            unselected_link_index = self.choose_random_unselected_link()
            new_state.swap_links(selected_link_index,
                    unselected_link_index)
        elif transition_dice <= transition_cutoffs[1]:
            # Select an unselected link
            logger.debug("Chose selection transition.")
            unselected_link_index = self.choose_random_unselected_link()
            new_state.select_link(unselected_link_index)
        else:
            # Unselect a selected link
            logger.debug("Chose unselection transition.")
            selected_link_index = self.choose_random_selected_link()
            new_state.unselect_link(selected_link_index)

        return new_state


class NoSwapArrayLinksState(ArrayLinksState):
    """Similar to `ArrayLinksState`, however, it does not perform any
    swap operations.

    """
    def swap_links(self, selected_link_index, unselected_link_index):
        """Raises `NotImplementedError`."""
        raise NotImplementedError


    def calc_num_neighboring_states(self):
        """Calculates the number of possible neighboring states."""
        return self.calc_num_links()


    def calc_transition_cutoffs(self):
        """Calculate the appropriate cutoffs for the possible
        transitions.

        Returns a tuple where the first item is the probability cutoff
        for using a swap transition, the second item is the probability
        cutoff for using a selection transition, and the third item is
        the probability cutoff for an unselection transition.

        """
        num_selected_links = self.calc_num_selected_links()
        num_unselected_links = self.calc_num_unselected_links()
        total_transitions = num_selected_links + num_unselected_links
        selection_prob = float(num_unselected_links) / total_transitions
        # Since all probabilities should add up to 1, we don't need to
        # calculate the unselection probability.

        # Now create the cutoffs
        transition_cutoffs = (selection_prob, 1)
        if SUPERDEBUG_MODE:
            logger.log(SUPERDEBUG, ("Transition choice distribution: "
                    "%s" % (transition_cutoffs,)))
        return transition_cutoffs


    def create_new_state(self):
        """Creates a new state on the basis of this state instance."""
        logger.debug("Creating a new links state.")
        # First, get an identical copy of this state
        new_state = self.copy()
        # Then decide which type of transition we're going to do
        transition_cutoffs = self.calc_transition_cutoffs()
        transition_dice = random.random()
        if SUPERDEBUG_MODE:
            logger.log(SUPERDEBUG, "Transition dice was %s" % (
                    transition_dice))
        if transition_dice <= transition_cutoffs[0]:
            # Select an unselected link
            logger.debug("Chose selection transition.")
            unselected_link_index = self.choose_random_unselected_link()
            new_state.select_link(unselected_link_index)
        else:
            # Unselect a selected link
            logger.debug("Chose unselection transition.")
            selected_link_index = self.choose_random_selected_link()
            new_state.unselect_link(selected_link_index)

        return new_state


class PLNOverallState(State):
    def __init__(
            self,
            annotated_interactions,
            active_gene_threshold,
            transition_ratio,
            selected_links=None,
            alpha=None,
            beta=None,
            link_prior=None,
            parameters_state_class=PLNParametersState
        ):
        """Create a new instance.

        :Parameters:
        - `annotated_interactions`: an `AnnotatedInteractionsGraph`
          instance
        - `active_gene_threshold`: the threshold at or above which a
          gene is considered "active"
        - `transition_ratio`: a `float` indicating the ratio of link
          transitions to parameter transitions
        - `selected_links`: a user-defined seed of links to start as
          selected
        - `alpha`: the false-positive rate; see `PLNParametersState` for
          more information
        - `beta`: the false-negative rate; see `PLNParametersState` for
          more information
        - `link_prior`: the assumed probability we would pick any one
          link as being active; see `PLNParametersState` for more
          information
        - `parameters_state_class`: the class of the parameters state to
          use [default: `PLNParametersState`]

        """
        process_links = annotated_interactions.get_all_links()
        # Choose some sample of the process links to be selected
        # initially, if none have been provided
        if selected_links is None:
            # NOTE: It's important process_links be a list for this step,
            # because random.sample doesn't work on sets
            selected_links = random.sample(process_links,
                    random.randint(0, len(process_links)))
        # We need to convert process links into a set, now.
        process_links = frozenset(process_links)
        # Next, figure out which interactions are active
        logger.info("Determining active interactions.")
        active_interactions = \
                annotated_interactions.get_active_interactions(
                        active_gene_threshold
        )
        self.links_state = PLNLinksState(
                process_links,
                selected_links,
                annotated_interactions,
                active_interactions
        )
        self.parameters_state = parameters_state_class(
                len(process_links),
                alpha,
                beta,
                link_prior
        )
        self.transition_ratio = transition_ratio
        # This is used to track how we arrived at the current state from
        # the previous one. It is `None` for the first state, but for
        # any new created states, it is set to the `_delta` attribute of
        # either the `links_state` or the `parameters_state`, depending
        # on which was used for the transition.
        self._delta = None


    def calc_log_prob_observed_given_selected(self):
        """Calculates the log base 10 of the probability (likelihood?)
        of the state of the active interactions given the selected
        process links.

        """
        # TODO: Add some more documentation to this docstring
        num_unselected_active_interactions = \
                self.links_state.calc_num_unselected_active_interactions()
        num_unselected_inactive_interactions = \
                self.links_state.calc_num_unselected_inactive_interactions()
        alpha = self.parameters_state.alpha
        log_unselected_probability = \
                (num_unselected_active_interactions * \
                math.log10(alpha)) + \
                (num_unselected_inactive_interactions * \
                math.log10(1 - alpha))

        num_selected_inactive_interactions = \
                self.links_state.calc_num_selected_inactive_interactions()
        num_selected_active_interactions = \
                self.links_state.calc_num_selected_active_interactions()
        beta = self.parameters_state.beta
        log_selected_probability = \
                (num_selected_inactive_interactions) * \
                math.log10(beta) + \
                (num_selected_active_interactions * \
                math.log10(1 - beta))

        log_prob_observed_given_selected = log_unselected_probability + \
                log_selected_probability
        return log_prob_observed_given_selected


    def calc_log_prob_selected(self):
        """Calculates the log base 10 of the probability that the number
        of terms selected would be as large as it is given the link
        prior.

        """
        # TODO: Add some more documentation to this docstring
        num_selected_links = self.links_state.calc_num_selected_links()
        num_unselected_links = \
                self.links_state.calc_num_unselected_links()
        num_total_links = num_selected_links + num_unselected_links
        link_prior = self.parameters_state.link_prior
        log_prob_of_selected = \
                (num_selected_links * \
                math.log10(link_prior)) + \
                (num_unselected_links * \
                math.log10(1 - link_prior))
        return log_prob_of_selected


    def calc_num_neighboring_states(self):
        """Returns the total number of states neighboring this one."""
        num_neighboring_states = \
                self.links_state.calc_num_neighboring_states() * \
                self.parameters_state.calc_num_neighboring_states()
        return num_neighboring_states


    def calc_log_likelihood(self):
        """Returns the log of the likelihood of this current state given
        the observed data.

        """
        # TODO: Add some more documentation to this docstring
        log_prob_obs_given_sel = \
                self.calc_log_prob_observed_given_selected()
        log_prob_sel = self.calc_log_prob_selected()
        num_neighbor_states = \
                self.calc_num_neighboring_states()
        log_likelihood = log_prob_obs_given_sel + log_prob_sel - \
                math.log10(num_neighbor_states)
        return log_likelihood


    def create_new_state(self):
        """Creates a new state on the basis of this state instance."""
        logger.debug("Creating new overall state.")
        new_state = self.copy()
        # Flip a coin to decide if we're going to consider altering the
        # parameters or the links state
        coin = random.random()
        logger.debug("Transition coin was %s" % coin)
        if coin <= self.transition_ratio:
            logger.debug("Chose link transition.")
            # Propose a transition to a new links state
            new_state.links_state = self.links_state.create_new_state()
            new_state._delta = new_state.links_state._delta
        else:
            # Propose a transition to a new parameters state
            logger.debug("Chose parameter transition.")
            new_state.parameters_state = \
                    self.parameters_state.create_new_state()
            new_state._delta = new_state.parameters_state._delta
        return new_state


class ArrayOverallState(PLNOverallState):
    """Similar to `PLNOverallState`, but using `ArrayLinksState` as the
    links state class, and `AnnotatedInteractionsArray` as the annotated
    interactions class.

    """
    def __init__(
            self,
            annotated_interactions,
            active_gene_threshold,
            transition_ratio,
            selected_links_indices=None,
            alpha=None,
            beta=None,
            link_prior=None,
            parameters_state_class=PLNParametersState,
            links_state_class=ArrayLinksState
        ):
        """Create a new instance.

        :Parameters:
        - `annotated_interactions`: an `AnnotatedInteractionsArray`
          instance
        - `active_gene_threshold`: the threshold at or above which a
          gene is considered "active"
        - `transition_ratio`: a `float` indicating the ratio of link
          transitions to parameter transitions
        - `selected_links_indices`: a user-defined seed of indices to
          links to start as selected
        - `alpha`: the false-positive rate; see `PLNParametersState` for
          more information
        - `beta`: the false-negative rate; see `PLNParametersState` for
          more information
        - `link_prior`: the assumed probability we would pick any one
          link as being active; see `PLNParametersState` for more
          information
        - `parameters_state_class`: the class of the parameters state to
          use [default: `PLNParametersState`]
        - `links_state_class`: the class of the links state to use
          [default: `ArrayLinksState`]

        """
        num_process_links = len(annotated_interactions.get_all_links())
        if selected_links_indices is None:
            # Note that we're randomly selecting a random number of
            # indices here.
            selected_links_indices = random.sample(
                    xrange(num_process_links),
                    random.randint(0, num_process_links)
            )
        # Next, figure out which interactions are active
        logger.info("Determining active interactions.")
        active_interactions = \
                annotated_interactions.get_active_interactions(
                        active_gene_threshold
        )
        self.links_state = links_state_class(
                annotated_interactions,
                selected_links_indices,
                active_interactions
        )
        self.parameters_state = parameters_state_class(
                num_process_links,
                alpha,
                beta,
                link_prior
        )
        self.transition_ratio = transition_ratio
        # This is used to track how we arrived at the current state from
        # the previous one. It is `None` for the first state, but for
        # any new created states, it is set to the `_delta` attribute of
        # either the `links_state` or the `parameters_state`, depending
        # on which was used for the transition.
        self._delta = None


class StateRecorder(object):
    pass


class PLNStateRecorder(StateRecorder):
    """A class to record the states of a `PLNMarkovChain` instance."""

    def __init__(self, links, parameter_distributions):
        """Create a new instance.

        :Parameters:
        - `links`: all the links which can possibly be selected
        - `parameter_distributions`: a dictionary with the names of the
          parameters and their possible distribution values

        """
        self.records_made = 0
        self.selected_links_tallies = dict.fromkeys(links, 0)
        self.parameter_tallies = {}
        for param_name, distribution in parameter_distributions.items():
            distribution_dict = dict.fromkeys(distribution, 0)
            self.parameter_tallies[param_name] = distribution_dict
        self._transitions_data = []


    def record_links_state(self, links_state):
        """Record the links selected in this state.

        :Parameters:
        - `links_state`: a `PLNLinksState` instance

        """
        for selected_link in links_state.selected_links:
            self.selected_links_tallies[selected_link] += 1


    def record_parameters_state(self, parameters_state):
        """Record the parameters in this state.

        :Parameters:
        - `parameters_state`: a `PLNParametersState` instance

        """
        for param_name in self.parameter_tallies.keys():
            param_value = getattr(parameters_state, param_name)
            self.parameter_tallies[param_name][param_value] += 1


    def record_transition(self, transition_info):
        """Records the information about the previous transition that
        led to this state.

        :Parameters:
        - `transition_info`: a tuple with the first item is a string
          representing the transition type, the second item is a real
          value representing the log of the transition ratio, and the
          third item is either `True` or `False`, representing whether
          or not that transition was accepted.

        """
        self._transitions_data.append(transition_info)


    def record_state(self, markov_chain):
        """Record the features of the current state.

        :Parameters:
        - `markov_chain`: a `PLNMarkovChain` instance

        """
        self.record_links_state(markov_chain.current_state.links_state)
        self.record_parameters_state(
                markov_chain.current_state.parameters_state)
        self.record_transition(markov_chain.last_transition_info)
        self.records_made += 1


    def write_links_probabilities(self, out_csvfile, buffer_size=100):
        """Output the final probabilities for the links.

        :Parameters:
        - `out_csvfile`: a `csv.DictWriter` instance with these fields:
          `term1`, `term2`, `probability`
        - `buffer_size`: the number of records to write to disk at once

        """
        num_total_steps = float(self.records_made)
        output_records = []

        for i, link_tally in enumerate(
                self.selected_links_tallies.iteritems()):
            term1, term2 = link_tally[0]
            link_selection_count = link_tally[1]
            link_probability = link_selection_count / num_total_steps
            output_records.append(
                    {
                        'term1': term1,
                        'term2': term2,
                        'probability': link_probability
                    }
            )
            # Periodically flush results to disk
            if not ((i + 1) % buffer_size):
                out_csvfile.writerows(output_records)
                # Flush the scores
                output_records = []

        # Flush any remaining records
        if output_records:
            out_csvfile.writerows(output_records)


    def write_parameters_probabilities(self, out_csvfile):
        """Output the final probabilities for the parameters.

        :Parameters:
        - `out_csvfile`: a `csv.DictWriter` instance with these fields:
          `parameter`, `value`, `probability`

        """
        num_total_steps = float(self.records_made)
        output_records = []
        param_names = self.parameter_tallies.keys()
        param_names.sort()
        for param_name in param_names:
            distribution = self.parameter_tallies[param_name]
            param_values = distribution.keys()
            param_values.sort()
            for param_value in param_values:
                value_probability = distribution[param_value] / \
                        num_total_steps
                output_records.append(
                        {
                            'parameter': param_name,
                            'value': param_value,
                            'probability': value_probability
                        }
                )
        out_csvfile.writerows(output_records)


    def write_transition_states(
            self,
            out_csvfile,
            buffer_size=100
        ):
        """Writes the transition state information for the Markov chain
        to CSV files.

        :Parameters:
        - `out_csvfile`: a `csv.DictWriter` instance to output the
          transition information for the burn-in period, with these
          fields: `transition_type`, `log_transition_ratio`, `accepted`
        - `buffer_size`: the number of records to write to disk at once

        """
        output_records = []

        for i, transition_info in enumerate(
                self._transitions_data):
            record = {
                    'transition_type': transition_info[0],
                    'log_transition_ratio': str(transition_info[1]),
                    'log_state_likelihood': str(transition_info[2]),
                    'accepted': str(transition_info[3]).lower()
            }
            output_records.append(record)
            # Periodically flush results to disk
            if not ((i + 1) % buffer_size):
                out_csvfile.writerows(output_records)
                # Flush the scores
                output_records = []

        # Flush any remaining records
        if output_records:
            out_csvfile.writerows(output_records)


class ArrayStateRecorder(PLNStateRecorder):
    """Similar to `PLNStateRecorder`, however, adapted to record the
    state of `ArrayLinksState` instances.

    """
    def __init__(self, links, parameter_distributions):
        """Create a new instance.

        :Parameters:
        - `links`: an list of the links that can be selected. NOTE:
          Links should appear in the same order in this list as they do
          for the construction of the corresponding
          `AnnotatedInteractionsArray` instance.
        - `parameter_distributions`: a dictionary with the names of the
          parameters and their possible distribution values

        """
        self.records_made = 0
        self.links = links
        self.selected_links_tallies = numpy.zeros(len(links),
                numpy.int)
        self.parameter_tallies = {}
        for param_name, distribution in parameter_distributions.items():
            distribution_dict = dict.fromkeys(distribution, 0)
            self.parameter_tallies[param_name] = distribution_dict
        self._transitions_data = []


    def record_links_state(self, links_state):
        """Record the links selected in this state.

        :Parameters:
        - `links_state`: an `ArrayLinksState` instance

        """
        self.selected_links_tallies += links_state.link_selections


    def write_links_probabilities(self, out_csvfile, buffer_size=100):
        """Output the final probabilities for the links.

        :Parameters:
        - `out_csvfile`: a `csv.DictWriter` instance with these fields:
          `term1`, `term2`, `probability`
        - `buffer_size`: the number of records to write to disk at once

        """
        num_total_steps = float(self.records_made)
        link_probabilities = self.selected_links_tallies / \
                num_total_steps
        output_records = []

        for i, link in enumerate(self.links):
            term1, term2 = link
            link_probability = link_probabilities[i]
            output_records.append(
                    {
                        'term1': term1,
                        'term2': term2,
                        'probability': link_probability
                    }
            )
            # Periodically flush results to disk
            if not ((i + 1) % buffer_size):
                out_csvfile.writerows(output_records)
                # Flush the scores
                output_records = []

        # Flush any remaining records
        if output_records:
            out_csvfile.writerows(output_records)


class DetailedArrayStateRecorder(ArrayStateRecorder):
    """Similar to `ArrayStateRecorder`, only it records more information
    about each step.

    """
    def __init__(self, links, parameter_distributions):
        """Create a new instance.

        :Parameters:
        - `links`: an list of the links that can be selected. NOTE:
          Links should appear in the same order in this list as they do
          for the construction of the corresponding
          `AnnotatedInteractionsArray` instance.
        - `parameter_distributions`: a dictionary with the names of the
          parameters and their possible distribution values

        """
        super(DetailedArrayStateRecorder, self).__init__(links,
                parameter_distributions)
        # We're using this variable to store the information of the
        # overall state.
        self._overall_data = []


    def record_overall_state(self, overall_state):
        """Record all the numbers of the current state.

        :Parameters:
        - `overall_state`: an `ArrayOverallState` instance

        """
        state_info = {}
        parameters_state = overall_state.parameters_state
        state_info['alpha'] = parameters_state.alpha
        state_info['beta'] = parameters_state.beta
        state_info['link_prior'] = parameters_state.link_prior
        links_state = overall_state.links_state
        state_info['num_selected_links'] = \
                links_state.calc_num_selected_links()
        state_info['num_unselected_links'] = \
                links_state.calc_num_unselected_links()
        state_info['num_selected_active_interactions'] = \
                links_state.calc_num_selected_active_interactions()
        state_info['num_selected_inactive_interactions'] = \
                links_state.calc_num_selected_inactive_interactions()
        state_info['num_unselected_active_interactions'] = \
                links_state.calc_num_unselected_active_interactions()
        state_info['num_unselected_inactive_interactions'] = \
                links_state.calc_num_unselected_inactive_interactions()
        self._overall_data.append(state_info)


    def record_state(self, markov_chain):
        """Record the features of the current state.

        :Parameters:
        - `markov_chain`: an `ArrayMarkovChain` instance

        """
        super(DetailedArrayStateRecorder, self).record_state(
                markov_chain)
        self.record_overall_state(markov_chain.current_state)


    def write_transition_states(
            self,
            out_csvfile,
            buffer_size=100
        ):
        """Writes the transition state information for the Markov chain
        to CSV files.

        :Parameters:
        - `out_csvfile`: a `csv.DictWriter` instance to output the
          transition information for the burn-in period, with these
          fields: `transition_type`, `log_transition_ratio`, `accepted`
        - `buffer_size`: the number of records to write to disk at once

        """
        output_records = []

        for i, transition_info in enumerate(
                self._transitions_data):
            record = {
                    'transition_type': transition_info[0],
                    'log_transition_ratio': str(transition_info[1]),
                    'log_state_likelihood': str(transition_info[2]),
                    'accepted': str(transition_info[3]).lower()
            }
            record.update(self._overall_data[i])
            output_records.append(record)
            # Periodically flush results to disk
            if not ((i + 1) % buffer_size):
                out_csvfile.writerows(output_records)
                # Flush the scores
                output_records = []

        # Flush any remaining records
        if output_records:
            out_csvfile.writerows(output_records)


class ShovePLNStateRecorder(PLNStateRecorder):
    """A `PLNStateRecorder` which uses a `shove` store to maintain
    tallies.

    """

    # We will need this class implemente when we move up to the full
    # annotations of MSigDB.

    def __init__(self, links, store):
        """Create a new instance.

        :Parameters:
        - `links`: all the links which can possibly be selected
        - `parameter_distributions`: a dictionary with the names of the
          parameters and their possible distribution values
        - `store`: a `Shove` instance

        """
        pass


class PLNMarkovChain(MarkovChain):
    """A class representing the Markov chain for process linkage
    networks.

    """
    def __init__(
            self,
            annotated_interactions,
            active_gene_threshold,
            transition_ratio,
            num_steps=NUM_STEPS,
            burn_in=BURN_IN,
            selected_links=None,
            alpha=None,
            beta=None,
            link_prior=None,
            state_recorder_class=PLNStateRecorder,
            parameters_state_class=PLNParametersState
        ):
        """Create a new instance.

        :Parameters:
        - `annotated_interactions`: an `AnnotatedInteractionsGraph`
          instance
        - `active_gene_threshold`: the threshold at or above which a
          gene is considered "active"
        - `transition_ratio`: a `float` indicating the ratio of link
          transitions to parameter transitions
        - `num_steps`: the number of steps to take in the Markov chain
        - `burn_in`: the number of steps to take before recording state
          information about the Markov chain (state records are
          discarded until complete)
        - `selected_links`: a user-defined seed of links to start as
          selected
        - `alpha`: the false-positive rate; see `PLNParametersState` for
          more information
        - `beta`: the false-negative rate; see `PLNParametersState` for
          more information
        - `link_prior`: the assumed probability we would pick any one
          link as being active; see `PLNParametersState` for more
          information
        - `state_recorder_class`: the class of the state recorder to use
          [default: `PLNStateRecorder`]
        - `parameters_state_class`: the class of the parameters state to
          use [default: `PLNParametersState]`

        """
        self.current_state = PLNOverallState(
                annotated_interactions,
                active_gene_threshold,
                transition_ratio,
                selected_links,
                alpha,
                beta,
                link_prior,
                parameters_state_class

        )
        self.state_recorder = state_recorder_class(
                self.current_state.links_state.process_links,
                self.current_state.parameters_state.get_parameter_distributions()
        )
        self.burn_in_steps = burn_in
        self.num_steps = num_steps
        self.burn_in_period = True
        # This attribute will keep track of how we transition through
        # the Markov chain by storing a tuple for the previous
        # transition. The first item for the transition information is
        # the type of transition performed, which is obtained from the
        # PLNOverallState._delta attribute's key. The second item of the
        # tuple is a floating point value representing the log of the
        # transition ratio computed in calc_log_transition_ratio(). The
        # third item in the tuple is either `True`, representing that
        # the transition was rejected, or `False`, representing that the
        # transition was accepted.
        self.last_transition_info = None


    def next_state(self):
        """Move to the next state in the Markov chain.

        This method creates a proposed state for a transition; it then
        assesses the "fitness" of this new state by comparing the
        likelihood of the proposed state to the likelihood of the
        current state as a (log of the) ratio of the two likelihoods.

        If this ratio is greater than 1 (i.e., the log of the ratio is
        positive, we accept the proposed state and transition to it. If,
        instead, the ratio is less than 1 (i.e., the log of the ratio is
        negative), we flip a rejection coin. If the ratio is still
        greater than the rejection coin, we accept the less likely
        proposed state, anyway (a feature which allows us to exit local
        maxima); otherwise we reject the proposed state and continue
        with the current state.

        """
        proposed_state = self.current_state.create_new_state()
        proposed_transition_type = proposed_state._delta[0]
        current_log_likelihood = \
                self.current_state.calc_log_likelihood()
        proposed_log_likelihood = \
                proposed_state.calc_log_likelihood()
        log_transition_ratio = proposed_log_likelihood - \
                current_log_likelihood
        logger.debug("Log of transition ratio: %s" % (
                log_transition_ratio))

        # Flip a coin to see if we'll accept the transition
        log_rejection_coin = math.log10(random.random())
        logger.debug("Log rejection coin: %s" % log_rejection_coin)
        if log_rejection_coin < log_transition_ratio:
            # We accept the proposed state!
            self.current_state = proposed_state
            logger.debug("Accepted proposed state.")
            accepted = True
            log_state_likelihood = proposed_log_likelihood
        else:
            logger.debug("Rejected proposed state.")
            accepted = False
            log_state_likelihood = current_log_likelihood

        logger.debug("Log of state likelihood: %s" % (
                log_state_likelihood))
        self.last_transition_info = (
                proposed_transition_type,
                log_transition_ratio,
                log_state_likelihood,
                accepted
        )


    def run(self):
        """Step through the states of the Markov chain.

        Runs for a total number of iterations equal to `burn_in` +
        `num_steps`.

        """
        logger.info("Beginning burn-in of %d steps." %
                self.burn_in_steps)
        broadcast_percent_complete = 0
        for i in xrange(self.burn_in_steps):
            logger.debug("Burn-in step %d of %d" % (i + 1,
                    self.burn_in_steps))
            self.next_state()
            percent_complete = int(100 * float(i + 1) /
                    self.burn_in_steps)
            if percent_complete >= (broadcast_percent_complete +
                    BROADCAST_PERCENT):
                broadcast_percent_complete = percent_complete
                logger.info("%d%% of burn-in complete." % (
                        percent_complete))

        logger.info("Beginning run of %d steps." % self.num_steps)
        self.burn_in_period = False
        broadcast_percent_complete = 0
        for i in xrange(self.num_steps):
            logger.debug("Step %d of %d" % (i + 1, self.num_steps))
            self.next_state()
            self.state_recorder.record_state(self)
            percent_complete = int(100 * float(i + 1) / self.num_steps)
            if percent_complete >= (broadcast_percent_complete +
                    BROADCAST_PERCENT):
                broadcast_percent_complete = percent_complete
                logger.info("%d%% of steps complete." % (
                        percent_complete))


class ArrayMarkovChain(PLNMarkovChain):
    """Similar to `PLNMarkovChain`, but using `numpy` arrays to track
    state information.

    """
    def __init__(
            self,
            annotated_interactions,
            active_gene_threshold,
            transition_ratio,
            num_steps=NUM_STEPS,
            burn_in=BURN_IN,
            selected_links_indices=None,
            alpha=None,
            beta=None,
            link_prior=None,
            state_recorder_class=ArrayStateRecorder,
            parameters_state_class=PLNParametersState,
            links_state_class=ArrayLinksState
        ):
        """Create a new instance.

        :Parameters:
        - `annotated_interactions`: an `AnnotatedInteractionsArray`
          instance
        - `active_gene_threshold`: the threshold at or above which a
          gene is considered "active"
        - `transition_ratio`: a `float` indicating the ratio of link
          transitions to parameter transitions
        - `num_steps`: the number of steps to take in the Markov chain
        - `burn_in`: the number of steps to take before recording state
          information about the Markov chain (state records are
          discarded until complete)
        - `selected_links_indices`: a user-defined seed of indices to
          links to start as selected
        - `alpha`: the false-positive rate; see `PLNParametersState` for
          more information
        - `beta`: the false-negative rate; see `PLNParametersState` for
          more information
        - `link_prior`: the assumed probability we would pick any one
          link as being active; see `PLNParametersState` for more
          information
        - `state_recorder_class`: the class of the state recorder to use
          [default: `ArrayStateRecorder`]
        - `parameters_state_class`: the class of the parameters state to
          use [default: `PLNParametersState`]
        - `links_state_class`: the class of the links state to use
          [default: `ArrayLinksState`]

        """
        self.current_state = ArrayOverallState(
                annotated_interactions,
                active_gene_threshold,
                transition_ratio,
                selected_links_indices,
                alpha,
                beta,
                link_prior,
                parameters_state_class,
                links_state_class
        )
        self.state_recorder = state_recorder_class(
                annotated_interactions.get_all_links(),
                self.current_state.parameters_state.get_parameter_distributions()
        )
        self.burn_in_steps = burn_in
        self.num_steps = num_steps
        self.burn_in_period = True
        self.last_transition_info = None


def main(argv=None):
    cli_parser = cli.McmcCli()
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
        parameters_state_class = RandomTransitionParametersState
    else:
        parameters_state_class = PLNParametersState
    if input_data.disable_swaps:
        logger.info("Disabling swap transitions.")
        links_state_class = NoSwapArrayLinksState
    else:
        links_state_class = ArrayLinksState
    if input_data.detailed_transitions:
        logger.info("Recording extra information for each state.")
        state_recorder_class = DetailedArrayStateRecorder
        transitions_csvfile = convutils.make_csv_dict_writer(
                input_data.transitions_outfile,
                DETAILED_TRANSITIONS_FIELDNAMES
        )
    else:
        state_recorder_class = ArrayStateRecorder
        transitions_csvfile = convutils.make_csv_dict_writer(
                input_data.transitions_outfile,
                TRANSITIONS_FIELDNAMES
        )
    markov_chain = ArrayMarkovChain(
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

