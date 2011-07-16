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
    parameter_names = ('link_false_pos', 'link_false_neg', 'link_prior')
    # I'm mostly placing these here to get pylint to stop complaining
    link_false_pos = None
    link_false_neg = None
    link_prior = None
    # Now set the bounds for link_false_pos (and link_false_neg)
    _link_false_pos_min = 0
    _link_false_pos_max = 1
    # This controls how many discrete settings we can choose from for
    # the false-positive rate
    _size_link_false_pos_distribution = 19
    _link_false_pos_distribution = [0.05 * k for k in range(1,
            _size_link_false_pos_distribution + 1)]
    if SUPERDEBUG_MODE:
        logger.log(SUPERDEBUG, ("link_false_pos distribution: "
                "{0}").format(_link_false_pos_distribution))
    _link_false_neg_distribution = _link_false_pos_distribution
    _link_false_neg_min = _link_false_pos_min
    _link_false_neg_max = _link_false_pos_max
    # This controls the maximum number of discrete settings we can
    # choose from for the link prior rate
    _size_link_prior_distribution = 20
    _link_prior_max = 0.5

    def __init__(
            self,
            number_of_links,
            link_false_pos=None,
            link_false_neg=None,
            link_prior=None
        ):
        """Create a new instance.

        :Parameters:
        - `number_of_links`: the total number of links being considered
        - `link_false_pos`: the false-positive rate for links, the
          portion of gene-gene interactions which were included, but
          shouldn't have been
        - `link_false_neg`: the false-negative rate for links, the
          portion of gene-gene interactions which weren't included, but
          should have been
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
            logger.log(SUPERDEBUG, ("link_prior_distribution: "
                    "{0}").format(self._link_prior_distribution))

        # Set all parameters, if not set already, and validate them.
        self._set_parameters_at_init(
                link_false_pos=link_false_pos,
                link_false_neg=link_false_neg,
                link_prior=link_prior
        )
        logger.debug(("Initial parameter settings: "
                "link_false_pos={0}, link_false_neg={1}, "
                "link_prior={2}").format(self.link_false_pos,
                    self.link_false_neg, self.link_prior)
        )

        # This variable is used to store the previous state that the
        # current state arrived from. When set, it should be a tuple
        # containing the name of the parameter, and the index in the
        # distribution that it had previous to this state.
        self._delta = None


    def _get_closest_parameter_index_and_value(self, desired_value,
            distribution):
        """Returns the index and value of the parameter value closest to
        the desired value.

        :Parameters:
        - `desired_value`: value for which the nearest match is sought
        - `distribution`: parameter distribution from which the nearest
          value is sought

        """
        closest_index, closest_value = min(enumerate(distribution),
                key=lambda d: abs(d[1] - desired_value))
        return (closest_index, closest_value)


    def _set_parameters_at_init(self, **params):
        """A helper function to verify and set the parameters at
        instantiation.

        :Parameters:
        - `**params`: the parameters to set

        """
        for param_name in self.parameter_names:
            value = params[param_name]
            param_distribution = getattr(self,
                    '_{0}_distribution'.format(param_name))
            param_index_name = '_{0}_index'.format(param_name)
            if value is None:
                # The user did not define the value ahead of time;
                # select one randomly
                rand_index = random.randrange(len(param_distribution))
                # Set the index
                setattr(self, param_index_name, rand_index)
                # Set the value of the parameter
                value = param_distribution[rand_index]
                setattr(self, param_name, value)
            else:
                # The user defined this parameter.
                # Get the value closest to this parameter.
                param_index, param_value = (
                        self._get_closest_parameter_index_and_value(
                                value, param_distribution))
                setattr(self, param_index_name, param_index)
                setattr(self, param_name, param_value)


    def _get_parameter_neighboring_indices(self, parameter_name):
        """Returns the indices for the possible neighboring values for
        `parameter`

        :Parameters:
        - `parameter_name`: the name of the parameter

        """
        param_index = getattr(self, '_{0}_index'.format(parameter_name))
        param_distribution = getattr(self, '_{0}_distribution'.format(
                parameter_name))
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
                    self, '_{0}_distribution'.format(param_name))
        return parameter_distributions


    def _calc_num_neighbors_per_parameter(self):
        """Calculates the number of neighboring values there are for
        each parameter.

        Returns a dictionary with the parameter names as keys and the
        number of neighboring values as values.

        """
        num_neighbors_per_parameter = {}
        neighboring_indices = (
                self._get_all_parameter_neighboring_indices())
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
        num_neighbors_per_param = (
                self._calc_num_neighbors_per_parameter())
        # Although we could call self._calc_num_neighboring states, it
        # is more efficient to just sum those per neighbor, so we avoid
        # looping twice
        total_num_neighbors = sum(num_neighbors_per_param.values())
        running_total = 0
        cutoffs = []
        for param_name in self.parameter_names:
            num_neighbors = num_neighbors_per_param[param_name]
            running_total += float(num_neighbors) / total_num_neighbors
            cutoffs.append(running_total)
        # The last cutoff should be at 1, so that we don't try and
        # index outside of the distribution
        cutoffs[-1] = 1
        if SUPERDEBUG_MODE:
            logger.log(SUPERDEBUG, ("Parameter selection random "
                    "distribution: {0}").format(cutoffs))
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
            logger.log(SUPERDEBUG, "Parameter coin: {0}".format(
                param_coin))
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
        current_index = getattr(self, '_{0}_index'.format(
                parameter_to_alter))
        new_index = random.choice(neighboring_indices)
        # Now set the parameter and its index in the new state
        setattr(new_state, '_{0}_index'.format(parameter_to_alter),
                new_index)
        new_value = getattr(self, '_{0}_distribution'.format(
                parameter_to_alter))[new_index]
        setattr(new_state, parameter_to_alter, new_value)
        new_state._delta = (parameter_to_alter, current_index)
        logger.debug("Changing parameter {0} from {1} to {2}\n".format(
                    parameter_to_alter,
                    getattr(self, parameter_to_alter),
                    new_value
                )
        )
        logger.debug(
                ("New state: link_false_pos={0.link_false_pos}, "
                    "link_false_neg={0.link_false_neg}, "
                    "link_prior={0.link_prior}"
                ).format(new_state)
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
            link_false_pos=None,
            link_false_neg=None,
            link_prior=None
        ):
        """Create a new instance.

        :Parameters:
        - `number_of_links`: the total number of links being considered
        - `link_false_pos`: the false-positive rate for links, the
          portion of gene-gene interactions which were included, but
          shouldn't have been
        - `link_false_neg`: the false-negative rate for links, the
          portion of gene-gene interactions which weren't included, but
          should have been
        - `link_prior`: the assumed probability we would select any one
          link

        """
        super(RandomTransitionParametersState, self).__init__(
            self,
            number_of_links,
            link_false_pos=link_false_pos,
            link_false_neg=link_false_neg,
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
            num_other_values_in_distribution = (
                    len(getattr(self, '_{0}_distribution'.format(
                        param_name))) - 1
            )
            num_neighbors_per_parameter[param_name] = (
                    num_other_values_in_distribution)
        return num_neighbors_per_parameter


    def create_new_state(self):
        """Creates a new state on the basis of this state instance."""
        new_state = self.copy()
        parameter_to_alter = self._choose_parameter_to_alter()
        parameter_distribution = getattr(self,
                '_{0}_distribution'.format(parameter_to_alter))
        size_parameter_distribution = len(parameter_distribution)
        current_index = getattr(self, '_{0}_index'.format(
                parameter_to_alter))
        # Roll a die to get the new parameter index and value.
        new_index = random.randrange(size_parameter_distribution)
        # We could by random chance end up with the same index (and
        # value) as the current one; keep sampling until we have a
        # different index.
        while new_index == current_index:
            new_index = random.randrange(size_parameter_distribution)
        # Now set the parameter and its index in the new state
        setattr(new_state, '_{0}_index'.format(parameter_to_alter),
                new_index)
        new_value = parameter_distribution[new_index]
        setattr(new_state, parameter_to_alter, new_value)
        new_state._delta = (parameter_to_alter, current_index)
        logger.debug("Changing parameter {0} from {1} to {2}\n".format(
                    parameter_to_alter,
                    getattr(self, parameter_to_alter),
                    new_value
                )
        )
        logger.debug(
                ("New state: link_false_pos={0.link_false_pos}, "
                    "link_false_neg={0.link_false_neg}, "
                    "link_prior={0.link_prior}"
                ).format(new_state)
        )
        return new_state


class FixedDistributionParametersState(RandomTransitionParametersState):
    """Similar to `RandomTransitionParametersState`, but with a fixed
    distribution for the link prior.

    """
    _link_prior_distribution = [
            0.00001,
            0.00005,
            0.0001,
            0.0005,
            0.001,
            0.005,
            0.01,
            ] + [
            # This latter portion gives a range [0.05, 0.5], inclusive.
            0.05 * k for k in range(1, 11)]
    def __init__(
            self,
            number_of_links,
            link_false_pos=None,
            link_false_neg=None,
            link_prior=None,
        ):
        """Create a new instance.

        :Parameters:
        - `number_of_links`: the total number of links being considered
        - `link_false_pos`: the false-positive rate for links, the
          portion of gene-gene interactions which were included, but
          shouldn't have been
        - `link_false_neg`: the false-negative rate for links, the
          portion of gene-gene interactions which weren't included, but
          should have been
        - `link_prior`: the assumed probability we would select any one
          link

        """
        # Set all parameters, if not set already, and validate them.
        self._set_parameters_at_init(
                link_false_pos=link_false_pos,
                link_false_neg=link_false_neg,
                link_prior=link_prior
        )
        logger.debug(("Initial parameter settings: "
                "link_false_pos={0}, link_false_neg={1}, "
                "link_prior={2}").format(
                    self.link_false_pos,
                    self.link_false_neg,
                    self.link_prior
                )
        )

        self._parameter_selection_cutoffs = None

        self._delta = None



class TermPriorParametersState(RandomTransitionParametersState):
    """Similar to `RandomTransitionParametersState`, but includes an
    extra parameter, the term prior probability.

    Intended for use with `TermsAndLinksState`.

    """
    parameter_names = ('link_false_pos', 'link_false_neg', 'link_prior', 'term_prior')
    # This controls the maximum number of discrete settings we can
    # choose from for the link prior rate
    _size_term_prior_distribution = 20
    _term_prior_max = 0.5

    def __init__(
            self,
            number_of_links,
            number_of_terms,
            link_false_pos=None,
            link_false_neg=None,
            link_prior=None,
            term_prior=None
        ):
        """Create a new instance.

        :Parameters:
        - `number_of_links`: the total number of links being considered
        - `number_of_terms`: the total number of terms being considered
        - `link_false_pos`: the false-positive rate for links, the
          portion of gene-gene interactions which were included, but
          shouldn't have been
        - `link_false_neg`: the false-negative rate for links, the
          portion of gene-gene interactions which weren't included, but
          should have been
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
            logger.log(SUPERDEBUG, ("link_prior_distribution: "
                    "{0}").format(self._link_prior_distribution))

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
            logger.log(SUPERDEBUG, ("term_prior_distribution: "
                    "{0}").format(self._term_prior_distribution))

        # Set all parameters, if not set already, and validate them.
        self._set_parameters_at_init(
                link_false_pos=link_false_pos,
                link_false_neg=link_false_neg,
                link_prior=link_prior,
                term_prior=term_prior
        )
        logger.debug(("Initial parameter settings: "
                "link_false_pos={0}, link_false_neg={1}, "
                "link_prior={2}, term_prior={3}").format(
                    self.link_false_pos,
                    self.link_false_neg,
                    self.link_prior,
                    self.term_prior
                )
        )

        self._parameter_selection_cutoffs = None

        # This variable is used to store the previous state that the
        # current state arrived from. When set, it should be a tuple
        # containing the name of the parameter, and the index in the
        # distribution that it had previous to this state.
        self._delta = None


class FixedTermPriorParametersState(TermPriorParametersState):
    """Similar to `TermPriorParametersState`, but with fixed
    distributions for the term and link priors.

    """
    _link_prior_distribution = (
            FixedDistributionParametersState._link_prior_distribution)
    _term_prior_distribution = _link_prior_distribution
    if SUPERDEBUG_MODE:
        logger.log(SUPERDEBUG, "Term prior distribution: %s" %
                _term_prior_distribution)
    def __init__(
            self,
            number_of_links,
            number_of_terms,
            link_false_pos=None,
            link_false_neg=None,
            link_prior=None,
            term_prior=None
        ):
        """Create a new instance.

        :Parameters:
        - `number_of_links`: the total number of links being considered
        - `number_of_terms`: the total number of terms being considered
        - `link_false_pos`: the false-positive rate for links, the
          portion of gene-gene interactions which were included, but
          shouldn't have been
        - `link_false_neg`: the false-negative rate for links, the
          portion of gene-gene interactions which weren't included, but
          should have been
        - `link_prior`: the assumed probability we would select any one
          link
        - `term_prior`:the assumed probability we would select any one
          term

        """
        # Set all parameters, if not set already, and validate them.
        self._set_parameters_at_init(
                link_false_pos=link_false_pos,
                link_false_neg=link_false_neg,
                link_prior=link_prior,
                term_prior=term_prior
        )
        logger.debug(("Initial parameter settings: "
                "link_false_pos={0}, link_false_neg={1}, "
                "link_prior={2}, term_prior={3}").format(
                    self.link_false_pos,
                    self.link_false_neg,
                    self.link_prior,
                    self.term_prior
                )
        )

        self._parameter_selection_cutoffs = None

        # This variable is used to store the previous state that the
        # current state arrived from. When set, it should be a tuple
        # containing the name of the parameter, and the index in the
        # distribution that it had previous to this state.
        self._delta = None


class TermsParametersState(FixedTermPriorParametersState):
    parameter_names = ('link_false_pos', 'link_false_neg', 'link_prior',
            'term_false_pos', 'term_false_neg', 'term_prior')
    _term_false_pos_distribution = (
            PLNParametersState._link_false_pos_distribution)
    _term_false_pos_min = PLNParametersState._link_false_pos_min
    _term_false_pos_max = PLNParametersState._link_false_pos_max
    _term_false_neg_distribution = (
            PLNParametersState._link_false_pos_distribution)
    _term_false_neg_min = PLNParametersState._link_false_pos_min
    _term_false_neg_max = PLNParametersState._link_false_pos_max

    def __init__(
            self,
            number_of_links,
            number_of_terms,
            link_false_pos=None,
            link_false_neg=None,
            link_prior=None,
            term_false_pos=None,
            term_false_neg=None,
            term_prior=None
        ):
        """Create a new instance.

        :Parameters:
        - `number_of_links`: the total number of links being considered
        - `number_of_terms`: the total number of terms being considered
        - `link_false_pos`: the false-positive rate for links, the
          portion of gene-gene interactions which were included, but
          shouldn't have been
        - `link_false_neg`: the false-negative rate for links, the
          portion of gene-gene interactions which weren't included, but
          should have been
        - `link_prior`: the assumed probability we would select any one
          link
        - `term_false_pos`: the false-positive rate for terms, the
          portion of genes which were included, but shouldn't have been
        - `term_false_neg`: the false-negative rate for terms, the
          portion of genes which weren't included, but should have been
        - `term_prior`: the assumed probability we would select any one
          term
        - `term_prior`:the assumed probability we would select any one
          term

        """
        # Set all parameters, if not set already, and validate them.
        self._set_parameters_at_init(
                link_false_pos=link_false_pos,
                link_false_neg=link_false_neg,
                link_prior=link_prior,
                term_false_pos=term_false_pos,
                term_false_neg=term_false_neg,
                term_prior=term_prior
        )
        logger.debug(("Initial parameter settings: "
                "link_false_pos={0}, link_false_neg={1}, "
                "link_prior={2}, term_false_pos={3}, "
                "term_false_neg={4}, term_prior={5}").format(
                    self.link_false_pos,
                    self.link_false_neg,
                    self.link_prior,
                    self.term_false_pos,
                    self.term_false_neg,
                    self.term_prior,
                )
        )
        self._parameter_selection_cutoffs = None

        # This variable is used to store the previous state that the
        # current state arrived from. When set, it should be a tuple
        # containing the name of the parameter, and the index in the
        # distribution that it had previous to this state.
        self._delta = None


class PLNLinksState(State):
    def __init__(
            self,
            process_links,
            seed_links,
            annotated_interactions,
            active_interactions
        ):
        """Create a new PLNLinksState instance

        :Parameters:
        - `process_links`: the set of all possible links between
          biological processes
        - `seed_links`: the subset of links being considered as
          "selected" initially in the process linkage network
        - `annotated_interactions`: an `AnnotatedInteractionsGraph`
          instance
        - `active_interactions`: a list of indices of interactions that
          are considered "active"

        """
        self.process_links = process_links
        self._annotated_interactions = annotated_interactions
        self._active_interactions = numpy.zeros(
                self._annotated_interactions.calc_num_interactions(),
                bool
        )
        self._active_interactions[numpy.array(active_interactions)] = True
        self._num_active_interactions = len(active_interactions)

        self._num_interactions = self._annotated_interactions.calc_num_coannotated_interactions()
        # _interaction_selection_counts maintains the number of times a
        # gene-gene interaction has been "covered" by the selected
        # process links
        self._interaction_selection_counts = numpy.zeros(
                self._annotated_interactions.calc_num_interactions(),
                int
        )
        # _num_selected_interactions keeps track of how many
        # interactions have been selected
        self._num_selected_interactions = 0
        # _num_selected_active_interactions keeps a cache of how many
        # active interactions are currently selected
        self._num_selected_active_interactions = 0

        # This variable is used to store the previous state that the
        # current state arrived from. When set, it should be a tuple
        # where the first item is a string describing the type of
        # transition, and the second item is the link or pair of links
        # used in the transition. Acceptable strings for first item are:
        # 'link_selection', 'link_unselection', and 'link_swap'. For
        # 'link_selection' and 'link_unselection', the second item is a
        # tuple of (annotation1, annotation2); for 'link_swap', the
        # second item is a tuple of links ((annotation1, annotation2),
        # (annotation3, annotation4))
        #
        # If self._delta is None, it indicates this is an origin state
        # with no ancestor.
        self._delta = None

        # Now get the appropriate tallies for the selected interactions
        self.unselected_links = set(self.process_links)
        self.selected_links = set()
        for annotation1, annotation2 in seed_links:
            self.select_link(annotation1, annotation2)
        # We need to return the delta to None
        self._delta = None


    def report_interactions(self):
        """Have the state report the number of active and overall
        interactions.

        """
        logger.info(("{0} interactions possible to include ({1} "
                "active).").format(
                    self._num_interactions,
                    self._num_active_interactions
                )
        )


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
            # increment the selection count.
            if self._interaction_selection_counts[interaction] == 1:
                self._num_selected_interactions += 1
                # And if this interaction is noted as active, increment
                # the active selected count
                if self._active_interactions[interaction]:
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
        self._delta = ('link_selection', (annotation1, annotation2))


    def _mark_interactions_unselected(self, interactions):
        """Marks interactions as being unselected by coannotating
        links.

        """
        for interaction in interactions:
            self._interaction_selection_counts[interaction] -= 1
            # If we have removed the only link which selected this
            # interaction, decrement the selection count
            if not self._interaction_selection_counts[interaction]:
                self._num_selected_interactions -= 1
                # Further, if it was an active interaction, deduct from
                # the count of selected active interactions
                if self._active_interactions[interaction]:
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
        self._delta = ('link_unselection', (annotation1, annotation2))


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
        self._delta = ('link_swap', (selected_link, unselected_link))


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
        return self._num_selected_interactions


    def calc_num_unselected_interactions(self):
        """Returns the number of interactions covered by no selected
        link.

        """
        return (self._num_interactions -
                self._num_selected_interactions)


    def calc_num_selected_active_interactions(self):
        """Returns the number of active interactions covered by at least
        one selected link.

        """
        return self._num_selected_active_interactions


    def calc_num_unselected_active_interactions(self):
        """Returns the number of active interactions covered by no
        selected link.

        """
        return (self._num_active_interactions -
                self._num_selected_active_interactions)


    def calc_num_selected_inactive_interactions(self):
        """Returns the number of inactive interactions covered by at
        least one selected link.

        """
        return (self.calc_num_selected_interactions() -
                self.calc_num_selected_active_interactions())


    def calc_num_unselected_inactive_interactions(self):
        """Returns the number of inactive interactions covered by no
        selected link.

        """
        return (self.calc_num_unselected_interactions() -
                self.calc_num_unselected_active_interactions())


class ArrayLinksState(PLNLinksState):
    """Similar to `PLNLinksState`, however, uses array data structures
    to track which links are selected and unselected.

    """
    def __init__(
            self,
            annotated_interactions,
            active_interactions,
            seed_links_indices=None
        ):
        """Create a new ArrayLinksState instance

        :Parameters:
        - `annotated_interactions`: an `AnnotatedInteractionsArray`
          instance
        - `active_interactions`: a list of indices of interactions that
          are considered "active"
        - `seed_links_indices`: indices for the subset of links
          being considered as "selected" initially in the process
          linkage network

        """
        self._annotated_interactions = annotated_interactions
        self._num_links = annotated_interactions.calc_num_links()
        self._num_selected_links = 0

        self._num_interactions = self._annotated_interactions.calc_num_coannotated_interactions()
        self._active_interactions = numpy.zeros(
                self._annotated_interactions.calc_num_interactions(),
                bool
        )
        self._active_interactions[numpy.array(active_interactions)] = True
        self._num_active_interactions = len(active_interactions)

        # _interaction_selection_counts maintains the number of times a
        # gene-gene interaction has been "covered" by the selected
        # process links
        self._interaction_selection_counts = numpy.zeros(
                self._annotated_interactions.calc_num_interactions(),
                int
        )
        # _num_selected_interactions keeps track of how many
        # interactions have been selected
        self._num_selected_interactions = 0
        # _num_selected_active_interactions keeps a cache of how many
        # active interactions are currently selected
        self._num_selected_active_interactions = 0

        # _link_selections is a 1-dimensional `numpy.ndarray` of boolean
        # data type, where the value at each index is `True` if the link
        # represented by that index has been selected, or `False` if the
        # link is unselected.
        self.link_selections = numpy.zeros(self._num_links, numpy.bool)
        self._process_seed_links(seed_links_indices)
        self._delta = None
        self._serialization = None


    def _process_seed_links(self, seed_links=None):
        """Adds seed links to the network.

        If no seed links are provided, it will select one legitimate
        link, uniformly at random, to add.

        """
        if seed_links is not None:
            for index in seed_links:
                self.select_link(index)


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
        self._delta = ('link_selection', index)


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
        self._delta = ('link_unselection', index)


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
        self._delta = ('link_swap', (selected_link_index,
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
        new_state._serialization = None
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


    def serialize_state(self):
        """Returns a representation of the state by which it can be
        stored and later re-constituted.

        """
        if self._serialization is None:
            self._serialization = bpn.structures.BoolBitArray(
                    self.link_selections)
        return self._serialization


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
        new_state._serialization = None
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
            active_interactions,
            seed_links_indices=None
        ):
        """Create a new ArrayLinksState instance

        :Parameters:
        - `annotated_interactions`: an `AnnotatedInteractions2dArray`
          instance
        - `active_interactions`: a list of indices of interactions that
          are considered "active"
        - `seed_links_indices`: indices for the subset of links
          being considered as "selected" initially in the process
          linkage network

        """
        self._annotated_interactions = annotated_interactions
        self._num_terms = annotated_interactions.calc_num_terms()
        self._num_selected_terms = 0
        self._num_links = annotated_interactions.calc_num_links()
        self._num_selected_links = 0

        self._num_interactions = self._annotated_interactions.calc_num_coannotated_interactions()
        self._active_interactions = numpy.zeros(
                self._annotated_interactions.calc_num_interactions(),
                bool
        )
        self._active_interactions[numpy.array(active_interactions)] = True
        self._num_active_interactions = len(active_interactions)

        # _interaction_selection_counts maintains the number of times a
        # gene-gene interaction has been "covered" by the selected
        # process links
        self._interaction_selection_counts = numpy.zeros(
                self._annotated_interactions.calc_num_interactions(),
                int
        )
        # _num_selected_interactions keeps track of how many
        # interactions have been selected
        self._num_selected_interactions = 0
        # _num_selected_active_interactions keeps a cache of how many
        # active interactions are currently selected
        self._num_selected_active_interactions = 0

        # link_selections is a 2-dimensional symmetric array of boolean
        # data type, where the value at an index is `True` if the link
        # represented by that index has been selected, or `False` if the
        # link is not selected.
        self.link_selections = bpn.structures.symzeros(self._num_terms,
                bool)

        # _term_links_counts is an array of integers, where a positive
        # value indicates the number of selected links in which that
        # term participates.
        self._term_links_counts = numpy.zeros(self._num_terms, int)

        # term_selections is an array of boolean data type, where the
        # value at an index is `True` if the term represented by that
        # index has been selected, or `False` otherwise.
        self.term_selections = numpy.zeros(self._num_terms, bool)

        self._process_seed_links(seed_links_indices)

        self._delta = None
        self._serialization = None


    def copy(self):
        """Create a copy of this state instance."""
        newcopy = copy.copy(self)
        newcopy._term_links_counts = self._term_links_counts.copy()
        newcopy.term_selections = self.term_selections.copy()
        newcopy.link_selections = self.link_selections.copy()
        newcopy._interaction_selection_counts = (
                self._interaction_selection_counts.copy())
        return newcopy


    def _process_seed_links(self, seed_links=None):
        """Adds seed links to the network.

        If no seed links are provided, it will select one legitimate
        link, uniformly at random, to add.

        """
        if seed_links is not None:
            for index in seed_links:
                # Make sure at least one of the terms is selected before
                # attempting to select the link, so we don't raise an
                # error.
                if not self.term_selections[index[0]]:
                    self.select_term(index[0])
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
            self.select_term(random_term1)
            self.select_link((random_term1, random_term2))


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
        if self.term_selections[term]:
            raise ValueError(
                    "Term {0} has already been selected!".format(term))
        self.term_selections[term] = True
        self._num_selected_terms += 1
        self._delta = ('term_selection', term)


    def unselect_term(self, term):
        """Unmark a term as selected (again)."""
        if not self.term_selections[term]:
            raise ValueError(
                    "Term {0} has already been unselected!".format(term))
        self._num_selected_terms -= 1
        self.term_selections[term] = False
        self._delta = ('term_unselection', term)


    def _select_terms_via_link(self, link):
        for term in link:
            # Increment the link count for each term.
            self._term_links_counts[term] += 1
            # Mark the term as selected.
            if not self.term_selections[term]:
                self.select_term(term)


    def _unselect_terms_via_link(self, link):
        for term in link:
            # Decrement the link count.
            self._term_links_counts[term] -= 1
            # Unselect the term if it's no longer part of any links.
            if not self._term_links_counts[term]:
                self.unselect_term(term)


    def select_link(self, index):
        """Mark a link as selected.

        :Parameters:
        - `index`: the 2-dimensional index of the link to mark as
          selected

        """
        term1_index, term2_index = index
        # Check to make sure that at least one of the two terms has
        # already been selected, since we only permit adding links which
        # have at least one selected term.
        if (not self.term_selections[term1_index]) and (not
                self.term_selections[term2_index]):
            raise ValueError(
                    ("Can not select link {0}; neither term "
                    "selected.").format(index)
            )
        self._select_terms_via_link(index)
        super(TermsAndLinksState, self).select_link(index)


    def unselect_link(self, index):
        """Mark a link as unselected.

        :Parameters:
        - `index`: the 2-dimensional index of the link to mark as
          unselected

        """
        self._unselect_terms_via_link(index)
        super(TermsAndLinksState, self).unselect_link(index)


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
        selected_terms = self.term_selections.nonzero()[0]
        while (term1 == term2) or (interactions is None):
            num_choices_discarded += 1
            # Select a term uniformly at random.
            term1 = random.randrange(self._num_terms)
            # Determine if that term is already in the set of selected
            # terms.
            if self.term_selections[term1]:
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
        new_state._serialization = None
        random_link = new_state._draw_random_valid_link()
        if new_state.link_selections[random_link]:
            # Remove the link if it's already a candidate.
            new_state.unselect_link(random_link)
        else:
            # Add the link if it's not a candidate.
            new_state.select_link(random_link)
        return new_state


class IntraTermsAndLinksState(TermsAndLinksState):
    """Similar to `TermsAndLinksState`, but includes
    intraterm-interactions for every term that participates in one or
    more links.

    """
    def __init__(
            self,
            annotated_interactions,
            active_interactions,
            seed_links_indices=None
        ):
        """Create a new IntraTermsAndLinksState instance

        :Parameters:
        - `annotated_interactions`: an `AnnotatedInteractions2dArray`
          instance
        - `seed_links_indices`: indices for the subset of links
          being considered as "selected" initially in the process
          linkage network
        - `active_interactions`: a list of indices of interactions that
          are considered "active"

        """
        super(IntraTermsAndLinksState, self).__init__(
                annotated_interactions,
                active_interactions,
                seed_links_indices=seed_links_indices
        )
        self._num_interactions = self._annotated_interactions.calc_num_coannotated_and_intraterm_interactions()


    def select_term(self, term):
        """Mark a term as selected (again)."""
        super(IntraTermsAndLinksState, self).select_term(term)
        intraterm_interactions = (
                self._annotated_interactions.get_intraterm_interactions(
                    term)
        )
        if intraterm_interactions:
            self._mark_interactions_selected(intraterm_interactions)


    def unselect_term(self, term):
        """Unmark a term as selected (again)."""
        super(IntraTermsAndLinksState, self).unselect_term(term)
        intraterm_interactions = (
                self._annotated_interactions.get_intraterm_interactions(
                    term)
        )
        if intraterm_interactions:
            self._mark_interactions_unselected(
                    intraterm_interactions)


class IndependentTermsAndLinksState(TermsAndLinksState):
    """Similar to `TermsAndLinksState`, but allowing terms to be
    selected/unselected independent from link selection/unselection.

    """
    def __init__(
            self,
            annotated_interactions,
            active_interactions,
            seed_terms_indices=None,
            seed_links_indices=None
        ):
        """Create a new IndependentTermsAndLinksState instance

        :Parameters:
        - `annotated_interactions`: an `AnnotatedInteractions2dArray`
          instance
        - `active_interactions`: a list of indices of interactions that
          are considered "active"
        - `seed_terms_indices`: indices for the subset of terms being
          considered as "selected" initially
        - `seed_links_indices`: indices for the subset of links
          being considered as "selected" initially

        """
        super(IndependentTermsAndLinksState, self).__init__(
                annotated_interactions,
                active_interactions,
                seed_links_indices=seed_links_indices
        )
        self._process_seed_terms(seed_terms_indices)
        self._delta = None


    def _process_seed_terms(self, seed_terms=None):
        """Adds seed terms to the network."""
        if seed_terms is not None:
            for index in seed_terms:
                self.select_term(index)


    def _process_seed_links(self, seed_links=None):
        """Adds seed links to the network."""
        if seed_links is not None:
            for index in seed_links:
                # Make sure at least one of the terms is selected before
                # attempting to select the link, so we don't raise an
                # error.
                if not self.term_selections[index[0]]:
                    self.select_term(index[0])
                self.select_link(index)


    def _calc_num_terms_based_transitions(self):
        num_selected_terms = self._num_selected_terms
        # Figure out the number of terms which are selected, but
        # participate in no selected links. These are the terms which
        # can be unselected.
        num_selected_terms_with_links = len(
                self._term_links_counts.nonzero()[0])
        num_selected_terms_without_links = (num_selected_terms -
                num_selected_terms_with_links)
        # Next figure out the number of terms which are not selected;
        # these are the terms which can be selected.
        num_unselected_terms = self._num_terms - num_selected_terms
        # The total number of terms-based transitions is the sum of
        # those which can be unselected plus those which can be
        # selected.
        num_terms_based_transitions = (num_selected_terms_without_links
                + num_unselected_terms)
        return num_terms_based_transitions


    def _calc_num_links_based_transitions(self):
        # Here we consider the total number of possible links. Some of
        # those will be selected already, but most will not. The concept
        # is that, regardless of a link's actual selection state, a
        # potential links adds one and only one possible transition:
        # selection if unselected, or unselection if selected.
        num_selected_terms = self._num_selected_terms
        num_unselected_terms = (self._num_terms -
                self._num_selected_terms)
        num_possible_links_between_selected_terms = scipy.comb(
                num_selected_terms, 2, True)
        num_possible_links_from_selected_terms_to_unselected_terms = (
                num_selected_terms * num_unselected_terms)
        num_links_based_transitions = (
                num_possible_links_between_selected_terms +
                num_possible_links_from_selected_terms_to_unselected_terms
        )
        return num_links_based_transitions


    def _calc_transition_ratio(self):
        """Calculates and returns the ratio which determines
        whether to attempt a term-based or link-based transition.

        Returns a floating point value between 0 and 1 representing the
        ratio of selecting a term-based transition versus a links-based
        transition.

        """
        num_terms_based_transitions = (
                self._calc_num_terms_based_transitions())
        num_links_based_transitions = (
                self._calc_num_links_based_transitions())
        total_transitions = (num_terms_based_transitions +
                num_links_based_transitions)
        ratio = float(num_terms_based_transitions) / total_transitions
        return ratio


    def _propose_random_term_transition(self):
        term_index = random.randrange(self._num_terms)
        if not self.term_selections[term_index]:
            return ('term_selection', term_index)
        elif not self._term_links_counts[term_index]:
            return ('term_unselection', term_index)
        else:
            return None


    def _propose_random_link_transition(self):
        term1_index = random.randrange(self._num_terms)
        term2_index = term1_index
        while term2_index == term1_index:
            term2_index = random.randrange(self._num_terms)
        link_index = (term1_index, term2_index)
        # Link can not be considered if neither term is selected.
        if (not self.term_selections[term1_index] and not
                self.term_selections[term2_index]):
            return None
        # Link can not be considered if it doesn't co-annotate any
        # interactions.
        elif (self._annotated_interactions.get_coannotated_interactions(
            link_index) is None):
            return None
        # The link is valid and at least one term is already selected.
        # If the link is selected, propose unselecting it.
        elif self.link_selections[link_index]:
            return ('link_unselection', link_index)
        # Otherwise, propose selecting it.
        else:
            return ('link_selection', link_index)


    def calc_num_neighboring_states(self):
        """Returns the total number of states neighboring this one."""
        num_neighboring_states = (
                self._calc_num_terms_based_transitions() +
                self._calc_num_links_based_transitions()
        )
        return num_neighboring_states


    def create_new_state(self):
        """Creates a new state on the basis of this state instance."""
        logger.debug("Creating a new links state.")
        # First, get an identical copy of this state
        new_state = self.copy()
        new_state._serialization = None
        # Calculate the ratio of terms-based transitions to link-based
        # transitions.
        transition_ratio = self._calc_transition_ratio()
        transition = None
        num_choices_discarded = -1
        while transition is None:
            num_choices_discarded += 1
            transition_coin = random.random()
            if transition_coin <= transition_ratio:
                transition = self._propose_random_term_transition()
            else:
                transition = self._propose_random_link_transition()
        if num_choices_discarded:
            logger.debug("Discarded {0} choices.".format(
                    num_choices_discarded))

        if transition[0] == 'term_selection':
            new_state.select_term(transition[1])
        elif transition[0] == 'term_unselection':
            new_state.unselect_term(transition[1])
        elif transition[0] == 'link_selection':
            new_state.select_link(transition[1])
        else:
            new_state.unselect_link(transition[1])

        return new_state


class IndependentIntraTermsAndLinksState(IndependentTermsAndLinksState,
        IntraTermsAndLinksState):
    """Similar to `IntraTermsAndLinksState`, but allowing terms to be
    selected/unselected independent from link selection/unselection.

    """
    pass


class GenesBasedTermsAndLinksState(IndependentTermsAndLinksState):
    """Similar to `IndependentTermsAndLinksState`, but genes are used in
    place of intraterm-interactions to assess overlap of selected terms.

    """
    def __init__(
            self,
            annotated_interactions,
            active_interactions,
            active_genes,
            seed_terms_indices=None,
            seed_links_indices=None
        ):
        """Create a new GenesBasedTermsAndLinksState instance

        :Parameters:
        - `annotated_interactions`: an `AnnotatedInteractions2dArray`
          instance
        - `active_interactions`: a list of indices of interactions that
          are considered "active"
        - `active_genes`: a list of genes considered "active"
        - `seed_terms_indices`: indices for the subset of terms being
          considered as "selected" initially
        - `seed_links_indices`: indices for the subset of links
          being considered as "selected" initially in the process
          linkage network

        """
        self._num_genes = annotated_interactions.calc_num_genes()
        self._num_active_genes = len(active_genes)
        # Keeps track of the number of genes annotated by one or more
        # selected terms.
        self._num_selected_genes = 0
        # Keeps track of the number of active genes annotated by one or
        # more selected terms.
        self._num_selected_active_genes = 0
        # Stores the number of selected terms by which a gene is
        # annotated.
        self._gene_selection_counts = numpy.zeros(self._num_genes)
        # Use this array to look up whether a gene is active or
        # inactive.
        self._active_genes = numpy.zeros(self._num_genes, bool)
        self._active_genes[numpy.array(active_genes)] = True

        super(GenesBasedTermsAndLinksState, self).__init__(
                annotated_interactions,
                active_interactions,
                seed_terms_indices=seed_terms_indices,
                seed_links_indices=seed_links_indices
        )


    def copy(self):
        """Create a copy of this state instance."""
        newcopy = super(GenesBasedTermsAndLinksState, self).copy()
        newcopy._gene_selection_counts = (
                self._gene_selection_counts.copy())
        return newcopy


    def _mark_genes_selected(self, genes):
        """Marks genes as being selected by a term.

        """
        for gene in genes:
            self._gene_selection_counts[gene] += 1
            # If this is the first time selecting this gene, increment
            # the count.
            if self._gene_selection_counts[gene] == 1:
                self._num_selected_genes += 1
                # Further, if this gene is active, increase the active
                # count.
                if self._active_genes[gene]:
                    self._num_selected_active_genes += 1


    def _mark_genes_unselected(self, genes):
        """Unmarks genes as being selected by a term.

        """
        for gene in genes:
            self._gene_selection_counts[gene] -= 1
            # If we have removed the only term which selected this gene,
            # decrement the selection count.
            if not self._gene_selection_counts[gene]:
                self._num_selected_genes -= 1
                # Further, if this gene was active, decrement that
                # count.
                if self._active_genes[gene]:
                    self._num_selected_active_genes -= 1


    def select_term(self, term):
        """Mark a term as selected (again)."""
        super(GenesBasedTermsAndLinksState, self).select_term(term)
        term_genes = self._annotated_interactions.get_annotated_genes(
                term)
        self._mark_genes_selected(term_genes)


    def unselect_term(self, term):
        """Unmark a term as selected (again)."""
        super(GenesBasedTermsAndLinksState, self).unselect_term(term)
        term_genes = self._annotated_interactions.get_annotated_genes(
                term)
        self._mark_genes_unselected(term_genes)


    def calc_num_selected_genes(self):
        """Returns the number of genes covered by at least one
        selected term.

        """
        return self._num_selected_genes


    def calc_num_unselected_genes(self):
        """Returns the number of genes covered by no selected
        term.

        """
        return self._num_genes - self._num_selected_genes


    def calc_num_selected_active_genes(self):
        """Returns the number of active genes covered by at least
        one selected term.

        """
        return self._num_selected_active_genes


    def calc_num_unselected_active_genes(self):
        """Returns the number of active genes covered by no
        selected term.

        """
        return self._num_active_genes - self._num_selected_active_genes


    def calc_num_selected_inactive_genes(self):
        """Returns the number of inactive genes covered by at
        least one selected term.

        """
        return (self._num_selected_genes -
                self._num_selected_active_genes)


    def calc_num_unselected_inactive_genes(self):
        """Returns the number of inactive genes covered by no
        selected term.

        """
        return (self.calc_num_unselected_genes() -
                self.calc_num_unselected_active_genes())


class PLNOverallState(State):
    def __init__(
            self,
            annotated_interactions,
            active_gene_threshold,
            transition_ratio,
            seed_links=None,
            link_false_pos=None,
            link_false_neg=None,
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
        - `seed_links`: a user-defined seed of links to start as
          selected
        - `link_false_pos`: the false-positive rate; see `PLNParametersState` for
          more information
        - `link_false_neg`: the false-negative rate; see `PLNParametersState` for
          more information
        - `link_prior`: the assumed probability we would select any one
          link; see `PLNParametersState` for more information
        - `parameters_state_class`: the class of the parameters state to
          use [default: `PLNParametersState`]

        """
        process_links = annotated_interactions.get_all_links()
        # We need to convert process links into a set, now.
        process_links = frozenset(process_links)
        # Next, figure out which interactions are active
        logger.info("Determining active interactions.")
        active_interactions = (
                annotated_interactions.get_active_coannotated_interactions(
                        active_gene_threshold)
        )
        self.links_state = PLNLinksState(
                process_links,
                annotated_interactions,
                active_interactions,
                seed_links
        )
        self.links_state.report_interactions()
        self.parameters_state = parameters_state_class(
                len(process_links),
                link_false_pos,
                link_false_neg,
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
        link_false_pos = self.parameters_state.link_false_pos
        log_unselected_probability = \
                (num_unselected_active_interactions * \
                math.log10(link_false_pos)) + \
                (num_unselected_inactive_interactions * \
                math.log10(1 - link_false_pos))

        num_selected_inactive_interactions = \
                self.links_state.calc_num_selected_inactive_interactions()
        num_selected_active_interactions = \
                self.links_state.calc_num_selected_active_interactions()
        link_false_neg = self.parameters_state.link_false_neg
        log_selected_probability = \
                (num_selected_inactive_interactions) * \
                math.log10(link_false_neg) + \
                (num_selected_active_interactions * \
                math.log10(1 - link_false_neg))

        log_prob_observed_given_selected = log_unselected_probability + \
                log_selected_probability
        return log_prob_observed_given_selected


    def calc_log_prob_selected(self):
        """Calculates the log base 10 of the probability that the number
        of links selected would be as large as it is given the link
        prior.

        """
        # TODO: Add some more documentation to this docstring
        num_selected_links = self.links_state.calc_num_selected_links()
        num_unselected_links = (
                self.links_state.calc_num_unselected_links())
        link_prior = self.parameters_state.link_prior
        log_prob_of_selected = (
                (num_selected_links * math.log10(link_prior)) +
                (num_unselected_links * math.log10(1 - link_prior))
        )
        return log_prob_of_selected


    def calc_num_neighboring_states(self):
        """Returns the total number of states neighboring this one."""
        num_neighboring_states = (
                self.links_state.calc_num_neighboring_states() *
                self.parameters_state.calc_num_neighboring_states()
        )
        return num_neighboring_states


    def calc_log_likelihood(self):
        """Returns the log of the likelihood of this current state given
        the observed data.

        """
        # TODO: Add some more documentation to this docstring
        log_prob_obs_given_sel = (
                self.calc_log_prob_observed_given_selected())
        log_prob_sel = self.calc_log_prob_selected()
        log_likelihood = log_prob_obs_given_sel + log_prob_sel
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
            new_state.parameters_state = (
                    self.parameters_state.create_new_state())
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
            seed_links_indices=None,
            link_false_pos=None,
            link_false_neg=None,
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
        - `seed_links_indices`: a user-defined seed of indices to
          links to start as selected
        - `link_false_pos`: the false-positive rate; see `PLNParametersState` for
          more information
        - `link_false_neg`: the false-negative rate; see `PLNParametersState` for
          more information
        - `link_prior`: the assumed probability we would select any one
          link; see `PLNParametersState` for more information
        - `parameters_state_class`: the class of the parameters state to
          use [default: `PLNParametersState`]
        - `links_state_class`: the class of the links state to use
          [default: `ArrayLinksState`]

        """
        num_process_links = annotated_interactions.calc_num_links()
        # Next, figure out which interactions are active
        logger.info("Determining active interactions.")
        active_interactions = (
                annotated_interactions.get_active_coannotated_interactions(
                        active_gene_threshold)
        )
        self.links_state = links_state_class(
                annotated_interactions,
                active_interactions,
                seed_links_indices
        )
        self.links_state.report_interactions()
        self.parameters_state = parameters_state_class(
                num_process_links,
                link_false_pos,
                link_false_neg,
                link_prior
        )
        self.transition_ratio = transition_ratio
        # This is used to track how we arrived at the current state from
        # the previous one. It is `None` for the first state, but for
        # any new created states, it is set to the `_delta` attribute of
        # either the `links_state` or the `parameters_state`, depending
        # on which was used for the transition.
        self._delta = None


    def serialize_state(self):
        """Returns a representation of the state by which it can be
        stored and later re-constituted.

        """
        # TODO: Include the parameters state.
        return self.links_state.serialize_state()


class TermsBasedOverallState(ArrayOverallState):
    """Similar to `ArrayOverallState`, but incorporating annotation
    terms into the likelihood function.

    """
    def __init__(
            self,
            annotated_interactions,
            active_gene_threshold,
            transition_ratio,
            seed_links_indices=None,
            link_false_pos=None,
            link_false_neg=None,
            link_prior=None,
            term_prior=None,
            parameters_state_class=TermPriorParametersState,
            links_state_class=TermsAndLinksState
        ):
        """Create a new instance.

        :Parameters:
        - `annotated_interactions`: an `AnnotatedInteractionsArray`
          instance
        - `active_gene_threshold`: the threshold at or above which a
          gene is considered "active"
        - `transition_ratio`: a `float` indicating the ratio of link
          transitions to parameter transitions
        - `seed_links_indices`: a user-defined seed of indices to
          links to start as selected
        - `link_false_pos`: the false-positive rate; see `PLNParametersState` for
          more information
        - `link_false_neg`: the false-negative rate; see `PLNParametersState` for
          more information
        - `link_prior`: the assumed probability we would select any one
          link; see `PLNParametersState` for more information
        - `term_prior`:the assumed probability we would select any one
          term; see `RandomTransitionParametersState` for more
          information
        - `parameters_state_class`: the class of the parameters state to
          use [default: `TermPriorParametersState`]
        - `links_state_class`: the class of the links state to use
          [default: `TermsAndLinksState`]

        """
        num_process_links = annotated_interactions.calc_num_links()
        num_terms = annotated_interactions.calc_num_terms()

        # Next, figure out which interactions are active
        logger.info("Determining active interactions.")
        # This is super-ugly, but we need to choose the appropriate
        # selection of interactions based off of whether or not we
        # should include intraterm interactions. The easiest way to tell
        # this is to just check if the links state model is supposed to
        # include intraterm interactions by checking if it's subclassed
        # off the intraterms model.
        if issubclass(links_state_class, IntraTermsAndLinksState):
            active_interactions = (
                    annotated_interactions.get_active_coannotated_and_intraterm_interactions(
                            active_gene_threshold)
            )
        else:
            active_interactions = (
                    annotated_interactions.get_active_coannotated_interactions(
                            active_gene_threshold)
            )
        self.links_state = links_state_class(
                annotated_interactions,
                active_interactions,
                seed_links_indices=seed_links_indices
        )
        self.links_state.report_interactions()
        self.parameters_state = parameters_state_class(
                num_process_links,
                num_terms,
                link_false_pos=link_false_pos,
                link_false_neg=link_false_neg,
                link_prior=link_prior,
                term_prior=term_prior
        )
        self.transition_ratio = transition_ratio
        # This is used to track how we arrived at the current state from
        # the previous one. It is `None` for the first state, but for
        # any new created states, it is set to the `_delta` attribute of
        # either the `links_state` or the `parameters_state`, depending
        # on which was used for the transition.
        self._delta = None


    def calc_log_prob_terms_selected(self):
        """Calculates the log base 10 of the probability that the number
        of terms selected would be as large as they are given the term
        prior.

        """
        num_selected_terms = self.links_state.calc_num_selected_terms()
        num_unselected_terms = (
                self.links_state.calc_num_unselected_terms())
        term_prior = self.parameters_state.term_prior
        log_prob_terms_selected = (
                (num_selected_terms * math.log10(term_prior)) +
                (num_unselected_terms * math.log10(1 - term_prior))
        )
        return log_prob_terms_selected


    def calc_log_prob_selected(self):
        """Calculates the log base 10 of the probability that the number
        of terms and links selected would be as large as they are given
        the term and link priors.

        """
        links_log_prob_selected = super(
                TermsBasedOverallState, self).calc_log_prob_selected()
        # We need to include the additional probability of selecting the
        # terms.
        terms_log_prob_selected = self.calc_log_prob_terms_selected()
        log_prob_selected = (terms_log_prob_selected +
                links_log_prob_selected)
        return log_prob_selected


class IndependentTermsBasedOverallState(TermsBasedOverallState):
    """Similar to `ArrayOverallState`, but incorporating annotation
    terms into the likelihood function.

    """
    def __init__(
            self,
            annotated_interactions,
            active_gene_threshold,
            transition_ratio,
            seed_terms_indices=None,
            seed_links_indices=None,
            link_false_pos=None,
            link_false_neg=None,
            link_prior=None,
            term_prior=None,
            parameters_state_class=TermPriorParametersState,
            links_state_class=IndependentTermsAndLinksState
        ):
        """Create a new instance.

        :Parameters:
        - `annotated_interactions`: an `AnnotatedInteractionsArray`
          instance
        - `active_gene_threshold`: the threshold at or above which a
          gene is considered "active"
        - `transition_ratio`: a `float` indicating the ratio of link
          transitions to parameter transitions
        - `seed_links_indices`: a user-defined seed of indices to
          links to start as selected
        - `link_false_pos`: the false-positive rate; see `PLNParametersState` for
          more information
        - `link_false_neg`: the false-negative rate; see `PLNParametersState` for
          more information
        - `link_prior`: the assumed probability we would select any one
          link; see `PLNParametersState` for more information
        - `term_prior`:the assumed probability we would select any one
          term; see `RandomTransitionParametersState` for more
          information
        - `parameters_state_class`: the class of the parameters state to
          use [default: `TermPriorParametersState`]
        - `links_state_class`: the class of the links state to use
          [default: `IndependentTermsAndLinksState`]

        """
        num_process_links = annotated_interactions.calc_num_links()
        num_terms = annotated_interactions.calc_num_terms()

        logger.info("Determining active interactions.")
        # See above comments to explain this hack below.
        if issubclass(links_state_class, IntraTermsAndLinksState):
            active_interactions = (
                    annotated_interactions.get_active_coannotated_and_intraterm_interactions(
                            active_gene_threshold)
            )
        else:
            active_interactions = (
                    annotated_interactions.get_active_coannotated_interactions(
                            active_gene_threshold)
            )
        self.links_state = links_state_class(
                annotated_interactions,
                active_interactions,
                seed_terms_indices=seed_terms_indices,
                seed_links_indices=seed_links_indices
        )
        self.links_state.report_interactions()
        self.parameters_state = parameters_state_class(
                num_process_links,
                num_terms,
                link_false_pos=link_false_pos,
                link_false_neg=link_false_neg,
                link_prior=link_prior,
                term_prior=term_prior
        )
        self.transition_ratio = transition_ratio
        # This is used to track how we arrived at the current state from
        # the previous one. It is `None` for the first state, but for
        # any new created states, it is set to the `_delta` attribute of
        # either the `links_state` or the `parameters_state`, depending
        # on which was used for the transition.
        self._delta = None


class GenesBasedOverallState(TermsBasedOverallState):
    """Similar to `TermsBasedOverallState`, but with genes being the
    measure of overlap between terms.

    """
    def __init__(
            self,
            annotated_interactions,
            active_gene_threshold,
            transition_ratio,
            seed_terms_indices=None,
            seed_links_indices=None,
            link_false_pos=None,
            link_false_neg=None,
            link_prior=None,
            term_false_pos=None,
            term_false_neg=None,
            term_prior=None,
            parameters_state_class=TermsParametersState,
            links_state_class=GenesBasedTermsAndLinksState
        ):
        """Create a new instance.

        :Parameters:
        - `annotated_interactions`: an `AnnotatedInteractionsArray`
          instance
        - `active_gene_threshold`: the threshold at or above which a
          gene is considered "active"
        - `transition_ratio`: a `float` indicating the ratio of link
          transitions to parameter transitions
        - `seed_links_indices`: a user-defined seed of indices to
          links to start as selected
        - `link_false_pos`: the false-positive rate for links, the
          portion of gene-gene interactions which were included, but
          shouldn't have been
        - `link_false_neg`: the false-negative rate for links, the
          portion of gene-gene interactions which weren't included, but
          should have been
        - `link_prior`: the assumed probability we would select any one
          link; see `PLNParametersState` for more information
        - `term_false_pos`: the false-positive rate for terms, the
          portion of genes which were included, but shouldn't have been
        - `term_false_neg`: the false-negative rate for terms, the
          portion of genes which weren't included, but should have been
        - `term_prior`:the assumed probability we would select any one
          term; see `RandomTransitionParametersState` for more
          information
        - `parameters_state_class`: the class of the parameters state to
          use [default: `TermsParametersState`]
        - `links_state_class`: the class of the links state to use
          [default: `GenesBasedTermsAndLinksState`]

        """
        num_process_links = annotated_interactions.calc_num_links()
        num_terms = annotated_interactions.calc_num_terms()

        logger.info("Determining active genes.")
        active_genes = annotated_interactions.get_active_genes(
                active_gene_threshold)
        logger.info("Determining active interactions.")
        active_interactions = (
                annotated_interactions.get_active_coannotated_interactions(
                        active_gene_threshold
                )
        )
        self.links_state = links_state_class(
                annotated_interactions,
                active_interactions,
                active_genes,
                seed_links_indices
        )
        self.links_state.report_interactions()
        self.parameters_state = parameters_state_class(
                num_process_links,
                num_terms,
                link_false_pos=link_false_pos,
                link_false_neg=link_false_neg,
                link_prior=link_prior,
                term_false_pos=term_false_pos,
                term_false_neg=term_false_neg,
                term_prior=term_prior
        )
        self.transition_ratio = transition_ratio
        self._delta = None


    def calc_log_prob_terms_selected(self):
        """Calculates the log base 10 of the probability that the number
        of terms selected would be as large as they are given the term
        prior.

        """
        num_selected_terms = self.links_state.calc_num_selected_terms()
        num_unselected_terms = (
                self.links_state.calc_num_unselected_terms())
        term_prior = self.parameters_state.term_prior
        log_prob_selected = (
                (num_selected_terms * math.log10(term_prior)) +
                (num_unselected_terms * math.log10(1 - term_prior))
        )

        num_unselected_active_genes = (
                self.links_state.calc_num_unselected_active_genes())
        num_unselected_inactive_genes = (
                self.links_state.calc_num_unselected_inactive_genes())
        term_false_pos = self.parameters_state.term_false_pos
        log_unselected_probability = (
                (num_unselected_active_genes *
                    math.log10(term_false_pos)) +
                (num_unselected_inactive_genes *
                    math.log10(1 - term_false_pos))
        )

        num_selected_inactive_genes = (
                self.links_state.calc_num_selected_inactive_genes())
        num_selected_active_genes = (
                self.links_state.calc_num_selected_active_genes())
        term_false_neg = self.parameters_state.term_false_neg
        log_selected_probability = (
                (num_selected_inactive_genes *
                    math.log10(term_false_neg)) +
                (num_selected_active_genes *
                    math.log10(1 - term_false_neg))
        )

        log_prob_observed_given_selected = (log_unselected_probability +
                log_selected_probability)

        log_prob_terms_selected = (log_prob_selected +
                log_prob_observed_given_selected)

        return log_prob_terms_selected

