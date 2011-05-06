#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright (c) 2011 Christopher D. Lasher
#
# This software is released under the MIT License. Please see
# LICENSE.txt for details.


"""State recorders for BPN states."""


import numpy

import logging
logger = logging.getLogger('bpn.mcmcbpn.recorders')

from bpn import structures
from defaults import SUPERDEBUG, SUPERDEBUG_MODE


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


    def record_state_statistics(self, overall_state):
        """Record all the numbers of the current state.

        :Parameters:
        - `overall_state`: an `ArrayOverallState` instance

        """
        state_info = {}
        parameters_state = overall_state.parameters_state
        state_info['link_false_pos'] = parameters_state.link_false_pos
        state_info['link_false_neg'] = parameters_state.link_false_neg
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
        self.record_state_statistics(markov_chain.current_state)


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


class TermsBasedStateRecorder(DetailedArrayStateRecorder):
    """Similar to `PLNStateRecorder`, however, adapted to record the
    state of `TermsBasedOverallState` instances.

    """
    def __init__(self, annotated_interactions, parameter_distributions):
        self._annotated_interactions = annotated_interactions
        self.records_made = 0
        num_terms = self._annotated_interactions.calc_num_terms()
        self.selected_terms_tallies = numpy.zeros(num_terms, int)
        self.selected_links_tallies = structures.symzeros(num_terms,
                int)
        self.parameter_tallies = {}
        for param_name, distribution in parameter_distributions.items():
            distribution_dict = dict.fromkeys(distribution, 0)
            self.parameter_tallies[param_name] = distribution_dict
        self._transitions_data = []
        # We're using this variable to store the information of the
        # overall state.
        self._overall_data = []


    def record_state_statistics(self, overall_state):
        """Record all the numbers of the current state.

        :Parameters:
        - `overall_state`: an `ArrayOverallState` instance

        """
        state_info = {}
        parameters_state = overall_state.parameters_state
        state_info['link_false_pos'] = parameters_state.link_false_pos
        state_info['link_false_neg'] = parameters_state.link_false_neg
        state_info['term_prior'] = parameters_state.term_prior
        state_info['link_prior'] = parameters_state.link_prior
        links_state = overall_state.links_state
        state_info['num_selected_terms'] = (
                links_state.calc_num_selected_terms())
        state_info['num_unselected_terms'] = (
                links_state.calc_num_unselected_terms())
        state_info['num_selected_links'] = (
                links_state.calc_num_selected_links())
        state_info['num_unselected_links'] = (
                links_state.calc_num_unselected_links())
        state_info['num_selected_active_interactions'] = (
                links_state.calc_num_selected_active_interactions())
        state_info['num_selected_inactive_interactions'] = (
                links_state.calc_num_selected_inactive_interactions())
        state_info['num_unselected_active_interactions'] = (
                links_state.calc_num_unselected_active_interactions())
        state_info['num_unselected_inactive_interactions'] = (
                links_state.calc_num_unselected_inactive_interactions())
        self._overall_data.append(state_info)


    def record_links_state(self, links_state):
        """Record the links selected in this state.

        :Parameters:
        - `links_state`: a `PLNLinksState` instance

        """
        super(TermsBasedStateRecorder, self).record_links_state(
                links_state)
        self.selected_terms_tallies += links_state.term_selections


    def record_state(self, markov_chain):
        """Record the features of the current state.

        :Parameters:
        - `markov_chain`: an `ArrayMarkovChain` instance

        """
        super(TermsBasedStateRecorder, self).record_state(
                markov_chain)
        self.record_state_statistics(markov_chain.current_state)


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


    def write_terms_probabilities(self, out_csvfile, buffer_size=100):
        """Output the final probabilities for the links.

        :Parameters:
        - `out_csvfile`: a `csv.DictWriter` instance with these fields:
          ``'term'``, ``'probability'``
        - `buffer_size`: the number of records to write to disk at once

        """
        num_total_steps = float(self.records_made)
        term_probabilities = (self.selected_terms_tallies /
                num_total_steps)
        output_records = []

        # This the indices of any terms which were marked as selected
        # one or more times throughout all recorded steps of the Markov
        # chain.
        selected_terms_indices = self.selected_terms_tallies.nonzero()[0]

        for i, index in enumerate(selected_terms_indices):
            # Get the actual terms from the indices.
            term = self._annotated_interactions.get_term_from_index(
                    index)
            term_probability = term_probabilities[index]
            output_records.append(
                    {
                        'term': term,
                        'probability': term_probability
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


    def write_links_probabilities(self, out_csvfile, buffer_size=100):
        """Output the final probabilities for the links.

        :Parameters:
        - `out_csvfile`: a `csv.DictWriter` instance with these fields:
          `term1`, `term2`, `probability`
        - `buffer_size`: the number of records to write to disk at once

        """
        num_total_steps = float(self.records_made)
        link_probabilities = (self.selected_links_tallies /
                num_total_steps)
        output_records = []

        # This gives the pairwise ``(term1_index, term2_index)`` indices
        # of any links which were marked as selected 1 or more times
        # throughout all recorded steps of the Markov chain.
        selected_links_indices = zip(
                *numpy.triu(self.selected_links_tallies).nonzero())

        for i, index in enumerate(selected_links_indices):
            # Get the actual terms from the indices.
            term1, term2 = self._annotated_interactions.get_termed_link(
                    index)
            link_probability = link_probabilities[index]
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

