#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright (c) 2011 Christopher D. Lasher
#
# This software is released under the MIT License. Please see
# LICENSE.txt for details.


"""States for the BPN Markov chain."""


import bisect
import collections
import copy
import math
import random

import numpy
import scipy

import bpn.structures

import logging
logger = logging.getLogger('bpn.mcmcbpn.states')

from defaults import SUPERDEBUG, SUPERDEBUG_MODE


class ParameterNotInDistributionError(ValueError):
    """Exception raised when a parameter is not available within the
    allowed distribution.

    """
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
        - `link_prior`: the assumed probability we would select any one
          link

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
        - `link_prior`: the assumed probability we would select any one
          link

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
        - `link_prior`: the assumed probability we would select any one
          link

        """
        super(RandomTransitionParametersState, self).__init__(
            self,
            number_of_links,
            alpha=alpha,
            beta=beta,
            link_prior=link_prior
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
            cutoffs = super(RandomTransitionParametersState, self)._construct_param_selection_cutoffs()
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
            num_other_values_in_distribution = (len(
                    getattr(self, '_%s_distribution' % param_name)) - 1)
            num_neighbors_per_parameter[param_name] = (
                    num_other_values_in_distribution)
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


class TermPriorParametersState(RandomTransitionParametersState):
    """Similar to `RandomTransitionParametersState`, but includes an
    extra parameter, the term prior probability.

    Intended for use with `TermsAndLinksState`.

    """
    parameter_names = ('alpha', 'beta', 'link_prior', 'term_prior')
    # This controls the maximum number of discrete settings we can
    # choose from for the link prior rate
    _size_term_prior_distribution = 20
    _term_prior_max = 0.5

    def __init__(
            self,
            number_of_links,
            number_of_terms,
            alpha=None,
            beta=None,
            link_prior=None,
            term_prior=None
        ):
        """Create a new instance.

        :Parameters:
        - `number_of_links`: the total number of links being considered
        - `number_of_terms`: the total number of terms being considered
        - `alpha`: the false-positive rate, the portion of gene-gene
          interactions which were included, but shouldn't have been
        - `beta`: the false-negative rate, the portion of gene-gene
          interactions which weren't included, but should have been
        - `link_prior`: the assumed probability we would select any one
          link
        - `term_prior`:the assumed probability we would select any one
          term


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

        # We must first set up the term prior distribution; this cannot
        # be known beforehand, because it depends on knowing the number
        # terms beforehand.
        if number_of_terms >= self._size_term_prior_distribution:
            self._term_prior_distribution = [
                    float(k) / number_of_terms for k in range(1,
                        self._size_term_prior_distribution + 1)
            ]
        else:
            self._term_prior_distribution = []
            for k in range(1, number_of_terms):
                value = float(k) / number_of_terms
                if value <= self._term_prior_max:
                    self._term_prior_distribution.append(value)
                else:
                    break

        if SUPERDEBUG_MODE:
            logger.log(SUPERDEBUG, "term_prior_distribution: %s" %
                    self._term_prior_distribution)

        # Set all parameters, if not set already, and validate them.
        self._set_parameters_at_init(alpha, beta, link_prior,
                term_prior)
        logger.debug("Initial parameter settings: "
                "alpha=%s, beta=%s, link_prior=%s, term_prior=%s" % (
                    self.alpha, self.beta, self.link_prior,
                    self.term_prior
                )
        )

        self._parameter_selection_cutoffs = None

        # This variable is used to store the previous state that the
        # current state arrived from. When set, it should be a tuple
        # containing the name of the parameter, and the index in the
        # distribution that it had previous to this state.
        self._delta = None


    def _set_parameters_at_init(self, alpha, beta, link_prior,
            term_prior):
        """A helper function to verify and set the parameters at
        instantiation.

        :Parameters:
        - `alpha`: the false-positive rate, the portion of gene-gene
          interactions which were included, but shouldn't have been
        - `beta`: the false-negative rate, the portion of gene-gene
          interactions which weren't included, but should have been
        - `link_prior`: the assumed probability we would select any one
          link
        - `term_prior`:the assumed probability we would select any one
          term

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
        #
        # TODO: Copying this default dict is slow; refactor this code to
        # instead use a numpy array, by converting each interaction from
        # a tuple of gene names to an index (integer) in the array.
        self._interaction_selection_counts = collections.defaultdict(
                int)
        # _num_selected_active_interactions keeps a cache of how many
        # active interactions are currently selected
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


    def _mark_interactions_selected(self, interactions):
        """Marks interactions as being selected by coannotating
        links.

        """
        for interaction in interactions:
            self._interaction_selection_counts[interaction] += 1
            # If this is the first time selecting this interaction,
            # and this interaction is noted as active, increment the
            # active count
            if (self._interaction_selection_counts[interaction] == 1) \
                    and (interaction in self._active_interactions):
                self._num_selected_active_interactions += 1


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
        link_annotated_interactions = (
            self._annotated_interactions.get_coannotated_interactions(
                    annotation1, annotation2))
        self._mark_interactions_selected(link_annotated_interactions)
        # Finally, we note in the delta this selection
        self._delta = ('selection', (annotation1, annotation2))


    def _mark_interactions_unselected(self, interactions):
        """Marks interactions as being unselected by coannotating
        links.

        """
        for interaction in interactions:
            self._interaction_selection_counts[interaction] -= 1
            # If we have removed the only link which selected this
            # interaction, remove the interaction from the dictionary
            if not self._interaction_selection_counts[interaction]:
                del self._interaction_selection_counts[interaction]
                # Further, if it was an active interaction, deduct from
                # the count of selected active interactions
                if interaction in self._active_interactions:
                    self._num_selected_active_interactions -= 1


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
        link_annotated_interactions = (
            self._annotated_interactions.get_coannotated_interactions(
                    annotation1, annotation2))
        self._mark_interactions_unselected(link_annotated_interactions)
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
        """Create a new ArrayLinksState instance

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
            raise ValueError(("The link at index {0} has already been "
                    "marked selected.").format(index))
        self.link_selections[index] = True
        self._num_selected_links += 1
        link_annotated_interactions = (
            self._annotated_interactions.get_coannotated_interactions(
                    index))
        self._mark_interactions_selected(link_annotated_interactions)
        # Finally, we note in the delta this selection
        self._delta = ('selection', index)


    def unselect_link(self, index):
        """Mark a link as unselected.

        :Parameters:
        - `index`: the index of the link to mark as unselected

        """
        if not self.link_selections[index]:
            raise ValueError(("The link at index %d has already been "
                    "marked unselected.").format(index))
        self.link_selections[index] = False
        self._num_selected_links -= 1
        link_annotated_interactions = (
            self._annotated_interactions.get_coannotated_interactions(
                    index))
        self._mark_interactions_unselected(link_annotated_interactions)
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


class TermsAndLinksState(NoSwapArrayLinksState):
    """A representation of the terms and links selected to explain
    observed data.

    """
    def __init__(
            self,
            annotated_interactions,
            selected_links_indices,
            active_interactions
        ):
        """Create a new ArrayLinksState instance

        :Parameters:
        - `annotated_interactions`: an `AnnotatedInteractions2dArray`
          instance
        - `selected_links_indices`: indices for the subset of links
          being considered as "selected" initially in the process
          linkage network
        - `active_interactions`: a set of interactions that are
          considered "active"

        """
        self._annotated_interactions = annotated_interactions
        self._num_terms = annotated_interactions.calc_num_terms()
        self._num_selected_terms = 0
        self._num_links = annotated_interactions.calc_num_links()
        self._num_selected_links = 0
        self._active_interactions = active_interactions
        self._interaction_selection_counts = collections.defaultdict(
                int)
        self._num_selected_active_interactions = 0

        # link_selections is a 2-dimensional symmetric array of boolean
        # data type, where the value at each index is `True` if the link
        # represented by that index has been selected, or `False` if the
        # link is not selected.
        self.link_selections = bpn.structures.symzeros(self._num_terms,
                bool)

        # _term_selections is an array of integers, where a positive
        # value indicates the term is selected.
        self._term_selections = numpy.zeros(self._num_terms, int)

        if selected_links_indices:
            for index in selected_links_indices:
                self.select_link(index)
        else:
            # We have to have at least one link selected.
            interactions = None
            random_term1 = None
            random_term2 = None
            while (random_term1 == random_term2) or (interactions is
                    None):
                random_term1 = random.randrange(self._num_terms)
                random_term2 = random.randrange(self._num_terms)
                interactions = (
                        self._annotated_interactions.get_coannotated_interactions(
                            (random_term1, random_term2))
                )
            self.select_link((random_term1, random_term2))

        self._delta = None


    def copy(self):
        """Create a copy of this state instance."""
        newcopy = copy.copy(self)
        newcopy._term_selections = self._term_selections.copy()
        newcopy.link_selections = self.link_selections.copy()
        newcopy._interaction_selection_counts = \
                self._interaction_selection_counts.copy()
        return newcopy


    def calc_num_neighboring_states(self):
        """Calculates the number of possible neighboring states."""
        # The number of neighboring states depends on two scenarios:
        #
        # The first scenario is a selected term is chosen first; then,
        # the number of terms considered possible to partner in a
        # link-based transition (selection or unselection) is all of the
        # other terms. There is only one option per pairing: marking the
        # link as selected if it's not already, or unmarking it if it is
        # already marked.
        num_selected_term_neighboring_states = (
                self._num_selected_terms * (self._num_terms - 1))
        # The second scenario is an unselected term is chosen first;
        # then, it may only partner with selected terms (and the only
        # option is to mark their link selected).
        num_unselected_term_neighboring_states = (
                (self._num_terms - self._num_selected_terms) *
                self._num_selected_terms
        )
        num_neighboring_states = (num_selected_term_neighboring_states +
                num_unselected_term_neighboring_states)
        return num_neighboring_states


    def calc_num_terms(self):
        """Returns the number of terms available."""
        return self._num_terms


    def calc_num_selected_terms(self):
        """Returns the number of selected terms."""
        return self._num_selected_terms


    def calc_num_unselected_terms(self):
        """Returns the number of unselected terms."""
        return self._num_terms - self._num_selected_terms


    def calc_num_links(self):
        """Returns the number of links available in the current state.

        Note that the number of links available is different than the
        number of valid links (co-annotating pairs of terms). More
        specifically, the number of links available is based instead on
        the number of terms currently selected in the state.
        Specifically, the value is (``number_of_selected_terms`` choose
        ``2``).

        """
        num_possible_links = scipy.comb(self._num_selected_terms, 2,
                exact=True)
        return num_possible_links


    def select_term(self, term):
        """Mark a term as selected (again)."""
        if not self._term_selections[term]:
            self._num_selected_terms += 1
        self._term_selections[term] += 1


    def unselect_term(self, term):
        """Unmark a term as selected (again)."""
        self._term_selections[term] -= 1
        if not self._term_selections[term]:
            self._num_selected_terms -= 1


    def select_link(self, index):
        """Mark a link as selected.

        :Parameters:
        - `index`: the 2-dimensional index of the link to mark as
          selected

        """
        super(TermsAndLinksState, self).select_link(index)
        self.select_term(index[0])
        self.select_term(index[1])


    def unselect_link(self, index):
        """Mark a link as unselected.

        :Parameters:
        - `index`: the 2-dimensional index of the link to mark as
          unselected

        """
        super(TermsAndLinksState, self).unselect_link(index)
        self.unselect_term(index[0])
        self.unselect_term(index[0])


    def _draw_random_valid_link(self):
        """Draws a random link from among those valid ones.

        The draw is done discriminantly, with the first draw of one term
        dictating the pool from which the second term for the link may
        be drawn.

        """
        num_choices_discarded = -1
        term1 = None
        term2 = None
        interactions = None
        selected_terms = self._term_selections.nonzero()[0]
        while (term1 == term2) or (interactions is None):
            num_choices_discarded += 1
            # Select a term uniformly at random.
            term1 = random.randrange(self._num_terms)
            # Determine if that term is already in the set of selected
            # terms.
            if self._term_selections[term1]:
                # The term is in the set of selected; choose another
                # term uniformly at random from all other terms.
                term2 = random.randrange(self._num_terms)
            else:
                # The term is not in the set of selected; choose another
                # term uniformly at random from the set of selected
                # terms.
                term2 = random.choice(selected_terms)
            interactions = (
                    self._annotated_interactions.get_coannotated_interactions(
                        (term1, term2))
            )

        if num_choices_discarded:
            logger.debug("Discarded {0} choices.".format(
                    num_choices_discarded))

        return (term1, term2)


    def create_new_state(self):
        """Creates a new state on the basis of this state instance."""
        logger.debug("Creating a new links state.")
        # First, get an identical copy of this state
        new_state = self.copy()
        random_link = new_state._draw_random_valid_link()
        if new_state.link_selections[random_link]:
            # Remove the link if it's already a candidate.
            new_state.unselect_link(random_link)
        else:
            # Add the link if it's not a candidate.
            new_state.select_link(random_link)
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
        - `link_prior`: the assumed probability we would select any one
          link; see `PLNParametersState` for more information
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
        - `link_prior`: the assumed probability we would select any one
          link; see `PLNParametersState` for more information
        - `parameters_state_class`: the class of the parameters state to
          use [default: `PLNParametersState`]
        - `links_state_class`: the class of the links state to use
          [default: `ArrayLinksState`]

        """
        num_process_links = annotated_interactions.calc_num_links()
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


