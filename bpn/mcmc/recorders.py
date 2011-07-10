#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright (c) 2011 Christopher D. Lasher
#
# This software is released under the MIT License. Please see
# LICENSE.txt for details.


"""State recorders for BPN states."""

import collections

import numpy

import logging
logger = logging.getLogger('bpn.mcmcbpn.recorders')

from bpn import structures
from defaults import (SUPERDEBUG, SUPERDEBUG_MODE, OUTPUT_BUFFER_SIZE,
        TRANSITIONS_BUFFER_SIZE)


# This is a bit of a hack to reduce memory usage of the records. Rather
# than create a new string for every transition detailing the type, this
# will allow just creating a reference to a string.
TRANSITION_TYPES = {
        'link_prior': 'link_prior',
        'link_false_pos': 'link_false_pos',
        'link_false_neg': 'link_false_neg',
        'term_prior': 'term_prior',
        'term_false_pos': 'term_false_pos',
        'term_false_neg': 'term_false_neg',
        'link_selection': 'link_selection',
        'link_unselection': 'link_unselection',
        'link_swap': 'link_swap',
        'term_selection': 'term_selection',
        'term_unselection': 'term_unselection',
}

SELECTION = 'selection'
UNSELECTION = 'unselection'


class OverallStateRecord(object):
    __slots__ = (
            'transition_type',
            'log_transition_ratio',
            'log_state_likelihood',
            'accepted',
            'log_rejection_prob'
    )

    def __init__(
            self,
            transition_type,
            log_transition_ratio,
            log_state_likelihood,
            accepted,
            log_rejection_prob
        ):
        self.transition_type = TRANSITION_TYPES[transition_type]
        self.log_transition_ratio = log_transition_ratio
        self.log_state_likelihood = log_state_likelihood
        self.accepted = accepted
        self.log_rejection_prob = log_rejection_prob


    def to_dict(self):
        """Converts the record to a dictionary for output."""
        d = {
                'transition_type': self.transition_type,
                'log_transition_ratio': self.log_transition_ratio,
                'log_state_likelihood': self.log_state_likelihood,
                'accepted': self.accepted,
                'log_rejection_prob': self.log_rejection_prob,
        }
        return d


class DetailedStateRecord(OverallStateRecord):
    __slots__ = (
            'link_false_pos',
            'link_false_neg',
            'link_prior',
            'num_selected_links',
            'num_unselected_links',
            'num_selected_active_interactions',
            'num_selected_inactive_interactions',
            'num_unselected_active_interactions',
            'num_unselected_inactive_interactions'
    )

    def __init__(
            self,
            transition_type,
            log_transition_ratio,
            log_state_likelihood,
            accepted,
            log_rejection_prob,
            link_false_pos,
            link_false_neg,
            link_prior,
            num_selected_links,
            num_unselected_links,
            num_selected_active_interactions,
            num_selected_inactive_interactions,
            num_unselected_active_interactions,
            num_unselected_inactive_interactions
        ):
        super(DetailedStateRecord, self).__init__(
                transition_type,
                log_transition_ratio,
                log_state_likelihood,
                accepted,
                log_rejection_prob
        )
        self.link_false_pos = link_false_pos
        self.link_false_neg = link_false_neg
        self.link_prior = link_prior
        self.num_selected_links = num_selected_links
        self.num_unselected_links = num_unselected_links
        self.num_selected_active_interactions = (
                num_selected_active_interactions)
        self.num_selected_inactive_interactions = (
                num_selected_inactive_interactions)
        self.num_unselected_active_interactions = (
                num_unselected_active_interactions)
        self.num_unselected_inactive_interactions = (
                num_unselected_inactive_interactions)


    def to_dict(self):
        d = super(DetailedStateRecord, self).to_dict()
        additional = {
                'link_false_pos': self.link_false_pos,
                'link_false_neg': self.link_false_neg,
                'link_prior': self.link_prior,
                'num_selected_links': self.num_selected_links,
                'num_unselected_links': self.num_unselected_links,
                'num_selected_active_interactions': (
                        self.num_selected_active_interactions),
                'num_selected_inactive_interactions': (
                        self.num_selected_inactive_interactions),
                'num_unselected_active_interactions': (
                        self.num_unselected_active_interactions),
                'num_unselected_inactive_interactions': (
                        self.num_unselected_inactive_interactions),
        }
        d.update(additional)
        return d


class TermsStateRecord(DetailedStateRecord):
    __slots__ = (
            'num_selected_terms',
            'num_unselected_terms'
    )

    def __init__(
            self,
            transition_type,
            log_transition_ratio,
            log_state_likelihood,
            accepted,
            log_rejection_prob,
            link_false_pos,
            link_false_neg,
            link_prior,
            num_selected_links,
            num_unselected_links,
            num_selected_active_interactions,
            num_selected_inactive_interactions,
            num_unselected_active_interactions,
            num_unselected_inactive_interactions,
            num_selected_terms,
            num_unselected_terms
        ):
        super(TermsStateRecord, self).__init__(
                transition_type,
                log_transition_ratio,
                log_state_likelihood,
                accepted,
                log_rejection_prob,
                link_false_pos,
                link_false_neg,
                link_prior,
                num_selected_links,
                num_unselected_links,
                num_selected_active_interactions,
                num_selected_inactive_interactions,
                num_unselected_active_interactions,
                num_unselected_inactive_interactions
        )
        self.num_selected_terms = num_selected_terms
        self.num_unselected_terms = num_unselected_terms


    def to_dict(self):
        d = super(TermsStateRecord, self).to_dict()
        additional = {
                'num_selected_terms': self.num_selected_terms,
                'num_unselected_terms': self.num_unselected_terms,
        }
        d.update(additional)
        return d


class IndependentTermsStateRecord(TermsStateRecord):
    __slots__ = ('term_prior',)

    def __init__(
            self,
            transition_type,
            log_transition_ratio,
            log_state_likelihood,
            accepted,
            log_rejection_prob,
            link_false_pos,
            link_false_neg,
            link_prior,
            num_selected_links,
            num_unselected_links,
            num_selected_active_interactions,
            num_selected_inactive_interactions,
            num_unselected_active_interactions,
            num_unselected_inactive_interactions,
            num_selected_terms,
            num_unselected_terms,
            term_prior
        ):
        super(IndependentTermsStateRecord, self).__init__(
                transition_type,
                log_transition_ratio,
                log_state_likelihood,
                accepted,
                log_rejection_prob,
                link_false_pos,
                link_false_neg,
                link_prior,
                num_selected_links,
                num_unselected_links,
                num_selected_active_interactions,
                num_selected_inactive_interactions,
                num_unselected_active_interactions,
                num_unselected_inactive_interactions,
                num_selected_terms,
                num_unselected_terms,
        )
        self.term_prior = term_prior


    def to_dict(self):
        d = super(IndependentTermsStateRecord, self).to_dict()
        d['term_prior'] = self.term_prior
        return d


class GenesBasedStateRecord(IndependentTermsStateRecord):
    __slots__ = (
            'term_false_pos',
            'term_false_neg',
            'num_selected_active_genes',
            'num_selected_inactive_genes',
            'num_unselected_active_genes',
            'num_unselected_inactive_genes'
    )

    def __init__(
            self,
            transition_type,
            log_transition_ratio,
            log_state_likelihood,
            accepted,
            log_rejection_prob,
            link_false_pos,
            link_false_neg,
            link_prior,
            num_selected_links,
            num_unselected_links,
            num_selected_active_interactions,
            num_selected_inactive_interactions,
            num_unselected_active_interactions,
            num_unselected_inactive_interactions,
            num_selected_terms,
            num_unselected_terms,
            term_prior,
            term_false_pos,
            term_false_neg,
            num_selected_active_genes,
            num_selected_inactive_genes,
            num_unselected_active_genes,
            num_unselected_inactive_genes
        ):
        super(GenesBasedStateRecord, self).__init__(
                transition_type,
                log_transition_ratio,
                log_state_likelihood,
                accepted,
                log_rejection_prob,
                link_false_pos,
                link_false_neg,
                link_prior,
                num_selected_links,
                num_unselected_links,
                num_selected_active_interactions,
                num_selected_inactive_interactions,
                num_unselected_active_interactions,
                num_unselected_inactive_interactions,
                num_selected_terms,
                num_unselected_terms,
                term_prior,
        )
        self.term_false_pos = term_false_pos
        self.term_false_neg = term_false_neg
        self.num_selected_active_genes = num_selected_active_genes
        self.num_selected_inactive_genes = num_selected_inactive_genes
        self.num_unselected_active_genes = num_unselected_active_genes
        self.num_unselected_inactive_genes = (
                num_unselected_inactive_genes)


    def to_dict(self):
        d = super(GenesBasedStateRecord, self).to_dict()
        additional = {
                'term_false_pos': self.term_false_pos,
                'term_false_neg': self.term_false_neg,
                'num_selected_active_genes': (
                        self.num_selected_active_genes),
                'num_selected_inactive_genes': (
                        self.num_selected_inactive_genes),
                'num_unselected_active_genes': (
                        self.num_unselected_active_genes),
                'num_unselected_inactive_genes': (
                        self.num_unselected_inactive_genes),
        }
        d.update(additional)
        return d


class StateRecorder(object):
    pass


class PLNStateRecorder(StateRecorder):
    """A class to record the states of a `PLNMarkovChain` instance."""

    def __init__(
            self,
            annotated_interactions,
            parameters_csvwriter,
            links_csvwriter,
            transitions_csvwriter,
            transitions_buffer_size=TRANSITIONS_BUFFER_SIZE
        ):
        """Create a new instance.

        :Parameters:
        - `annotated_interactions`: an `AnnotatedInteractionsGraph`
          instance
        - `parameters_csvwriter`: a `csv.DictWriter` instance for
          outputting parameter results with these fields:
          ``'parameter'``, ``'value'``, ``'probability'``
        - `links_csvwriter`: a `csv.DictWriter` instance for
          outputting links results ``'term1'``, ``'term2'``,
          ``'probability'``
        - `transitions_csvwriter`: a `csv.DictWriter` instance for
          outputting transitions data
        - `transitions_buffer_size`: number of records to maintain
          before outputting transitions information [default: ``10000``]

        """
        self.records_made = 0
        self.selected_links_tallies = dict.fromkeys(
                annotated_interactions.get_all_links(), 0)
        # We'll use nested defaultdicts to track the values for each
        # parameter that we see. In this way, we will not have to know
        # ahead of time what parameters are contained in the parameter
        # state and what values they may take.
        self.parameter_tallies = collections.defaultdict(
                lambda : collections.defaultdict(int))
        self._transitions_data = []
        self.parameters_csvwriter = parameters_csvwriter
        self.links_csvwriter = links_csvwriter
        self.transitions_csvwriter = transitions_csvwriter
        self._transitions_buffer_size = transitions_buffer_size


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
        for param_name in parameters_state.parameter_names:
            param_value = getattr(parameters_state, param_name)
            self.parameter_tallies[param_name][param_value] += 1


    def record_transition(self, markov_chain):
        """Records the information about the previous transition that
        led to this state.

        """
        record = OverallStateRecord(*markov_chain.last_transition_info)
        self._transitions_data.append(record)


    def record_state(self, markov_chain):
        """Record the features of the current state.

        :Parameters:
        - `markov_chain`: a `PLNMarkovChain` instance

        """
        self.record_links_state(markov_chain.current_state.links_state)
        self.record_parameters_state(
                markov_chain.current_state.parameters_state)
        self.record_transition(markov_chain)
        self.records_made += 1
        # Output the transitions data if we've progressed through enough
        # steps.
        if not (self.records_made % self._transitions_buffer_size):
            logger.info("Writing and flushing transitions data.")
            self.write_transition_states()
            self._flush_transitions_data()


    def write_links_probabilities(self, buffer_size=OUTPUT_BUFFER_SIZE):
        """Output the final probabilities for the links.

        :Parameters:
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
                self.links_csvwriter.writerows(output_records)
                # Flush the scores
                output_records = []

        # Flush any remaining records
        if output_records:
            self.links_csvwriter.writerows(output_records)


    def write_parameters_probabilities(self):
        """Output the final probabilities for the parameters."""
        num_total_steps = float(self.records_made)
        output_records = []
        param_names = self.parameter_tallies.keys()
        param_names.sort()
        for param_name in param_names:
            distribution = self.parameter_tallies[param_name]
            param_values = distribution.keys()
            param_values.sort()
            for param_value in param_values:
                value_probability = (distribution[param_value] /
                        num_total_steps)
                output_records.append(
                        {
                            'parameter': param_name,
                            'value': param_value,
                            'probability': value_probability
                        }
                )
        self.parameters_csvwriter.writerows(output_records)


    def _flush_transitions_data(self):
        """Flushes the transitions data that is stored.

        NOTE: Should only be done after outputting transition data
        (e.g., writing to file)

        """
        self._transitions_data = []


    def write_transition_states(self, buffer_size=OUTPUT_BUFFER_SIZE):
        """Writes the transition state information for the Markov chain
        to CSV files.

        :Parameters:
        - `buffer_size`: the number of records to write to disk at once

        """
        output_records = []

        for i, transition_info in enumerate(self._transitions_data):
            record = transition_info.to_dict()
            output_records.append(record)
            # Periodically flush results to disk
            if not ((i + 1) % buffer_size):
                self.transitions_csvwriter.writerows(output_records)
                # Flush the scores
                output_records = []

        # Flush any remaining records
        if output_records:
            self.transitions_csvwriter.writerows(output_records)


class ArrayStateRecorder(PLNStateRecorder):
    """Similar to `PLNStateRecorder`, however, adapted to record the
    state of `ArrayLinksState` instances.

    """
    def __init__(
            self,
            annotated_interactions,
            parameters_csvwriter,
            links_csvwriter,
            transitions_csvwriter,
            transitions_buffer_size=TRANSITIONS_BUFFER_SIZE
        ):
        """Create a new instance.

        :Parameters:
        - `annotated_interactions`: an `AnnotatedInteractionsGraph`
          instance
        - `parameters_csvwriter`: a `csv.DictWriter` instance for
          outputting parameter results with these fields:
          ``'parameter'``, ``'value'``, ``'probability'``
        - `links_csvwriter`: a `csv.DictWriter` instance for
          outputting links results ``'term1'``, ``'term2'``,
          ``'probability'``
        - `transitions_csvwriter`: a `csv.DictWriter` instance for
          outputting transitions data
        - `transitions_buffer_size`: number of records to maintain
          before outputting transitions information [default: ``10000``]

        """
        self.records_made = 0
        self.links = annotated_interactions.get_all_links()
        self.selected_links_tallies = numpy.zeros(len(self.links),
                numpy.int)
        # We'll use nested defaultdicts to track the values for each
        # parameter that we see. In this way, we will not have to know
        # ahead of time what parameters are contained in the parameter
        # state and what values they may take.
        self.parameter_tallies = collections.defaultdict(
                lambda : collections.defaultdict(int))
        self._transitions_data = []
        self.parameters_csvwriter = parameters_csvwriter
        self.links_csvwriter = links_csvwriter
        self.transitions_csvwriter = transitions_csvwriter
        self._transitions_buffer_size = transitions_buffer_size


    def record_links_state(self, links_state):
        """Record the links selected in this state.

        :Parameters:
        - `links_state`: an `ArrayLinksState` instance

        """
        self.selected_links_tallies += links_state.link_selections


    def write_links_probabilities(self, buffer_size=OUTPUT_BUFFER_SIZE):
        """Output the final probabilities for the links.

        :Parameters:
        - `buffer_size`: the number of records to write to disk at once

        """
        num_total_steps = float(self.records_made)
        link_probabilities = (self.selected_links_tallies /
                num_total_steps)
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
                self.links_csvwriter.writerows(output_records)
                # Flush the scores
                output_records = []

        # Flush any remaining records
        if output_records:
            self.links_csvwriter.writerows(output_records)


class DetailedArrayStateRecorder(ArrayStateRecorder):
    """Similar to `ArrayStateRecorder`, only it records more information
    about each step.

    """
    def record_transition(self, markov_chain):
        """Record all the numbers of the current state.

        :Parameters:
        - `markov_chain`: a `PLNMarkovChain` instance

        """
        transition_info = markov_chain.last_transition_info
        overall_state = markov_chain.current_state
        parameters_state = overall_state.parameters_state
        links_state = overall_state.links_state
        record = DetailedStateRecord(
                transition_info[0],
                transition_info[1],
                transition_info[2],
                transition_info[3],
                transition_info[4],
                parameters_state.link_false_pos,
                parameters_state.link_false_neg,
                parameters_state.link_prior,
                links_state.calc_num_selected_links(),
                links_state.calc_num_unselected_links(),
                links_state.calc_num_selected_active_interactions(),
                links_state.calc_num_selected_inactive_interactions(),
                links_state.calc_num_unselected_active_interactions(),
                links_state.calc_num_unselected_inactive_interactions(),
        )
        self._transitions_data.append(record)


class TermsBasedStateRecorder(DetailedArrayStateRecorder):
    """Similar to `PLNStateRecorder`, however, adapted to record the
    state of `TermsBasedOverallState` instances.

    """
    def __init__(
            self,
            annotated_interactions,
            parameters_csvwriter,
            links_csvwriter,
            terms_csvwriter,
            transitions_csvwriter,
            transitions_buffer_size=TRANSITIONS_BUFFER_SIZE
        ):
        """Create a new instance.

        :Parameters:
        - `annotated_interactions`: an `AnnotatedInteractionsGraph`
          instance
        - `parameters_csvwriter`: a `csv.DictWriter` instance for
          outputting parameter results with these fields:
          ``'parameter'``, ``'value'``, ``'probability'``
        - `links_csvwriter`: a `csv.DictWriter` instance for
          outputting links results ``'term1'``, ``'term2'``,
          ``'probability'``
        - `terms_csvwriter`: a `csv.DictWriter` instance for
          outputting terms results with these fields: ``'term'``,
          ``'probability'``
        - `transitions_csvwriter`: a `csv.DictWriter` instance for
          outputting transitions data
        - `transitions_buffer_size`: number of records to maintain
          before outputting transitions information [default: ``10000``]

        """
        self._annotated_interactions = annotated_interactions
        self.records_made = 0
        num_terms = self._annotated_interactions.calc_num_terms()
        self.selected_terms_tallies = numpy.zeros(num_terms, int)
        self.selected_links_tallies = structures.symzeros(num_terms,
                int)
        # We'll use nested defaultdicts to track the values for each
        # parameter that we see. In this way, we will not have to know
        # ahead of time what parameters are contained in the parameter
        # state and what values they may take.
        self.parameter_tallies = collections.defaultdict(
                lambda : collections.defaultdict(int))
        self._transitions_data = []
        self.parameters_csvwriter = parameters_csvwriter
        self.links_csvwriter = links_csvwriter
        self.terms_csvwriter = terms_csvwriter
        self.transitions_csvwriter = transitions_csvwriter
        self._transitions_buffer_size = transitions_buffer_size


    def record_transition(self, markov_chain):
        """Record all the numbers of the current state.

        :Parameters:
        - `markov_chain`: a `PLNMarkovChain` instance

        """
        transition_info = markov_chain.last_transition_info
        overall_state = markov_chain.current_state
        parameters_state = overall_state.parameters_state
        links_state = overall_state.links_state
        record = TermsStateRecord(
                transition_info[0],
                transition_info[1],
                transition_info[2],
                transition_info[3],
                transition_info[4],
                parameters_state.link_false_pos,
                parameters_state.link_false_neg,
                parameters_state.link_prior,
                links_state.calc_num_selected_links(),
                links_state.calc_num_unselected_links(),
                links_state.calc_num_selected_active_interactions(),
                links_state.calc_num_selected_inactive_interactions(),
                links_state.calc_num_unselected_active_interactions(),
                links_state.calc_num_unselected_inactive_interactions(),
                links_state.calc_num_selected_terms(),
                links_state.calc_num_unselected_terms(),
        )
        self._transitions_data.append(record)


    def record_links_state(self, links_state):
        """Record the links selected in this state.

        :Parameters:
        - `links_state`: a `PLNLinksState` instance

        """
        super(TermsBasedStateRecorder, self).record_links_state(
                links_state)
        self.selected_terms_tallies += links_state.term_selections


    def write_terms_probabilities(self, buffer_size=OUTPUT_BUFFER_SIZE):
        """Output the final probabilities for the links.

        :Parameters:
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
                self.terms_csvwriter.writerows(output_records)
                # Flush the scores
                output_records = []

        # Flush any remaining records
        if output_records:
            self.terms_csvwriter.writerows(output_records)


    def write_links_probabilities(self, buffer_size=OUTPUT_BUFFER_SIZE):
        """Output the final probabilities for the links.

        :Parameters:
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
                self.links_csvwriter.writerows(output_records)
                # Flush the scores
                output_records = []

        # Flush any remaining records
        if output_records:
            self.links_csvwriter.writerows(output_records)


class IndependentTermsBasedStateRecorder(TermsBasedStateRecorder):
    """Similar to `PLNStateRecorder`, however, adapted to record the
    state of `IndependentTermsBasedOverallState` instances.

    """
    def record_transition(self, markov_chain):
        """Record all the numbers of the current state.

        :Parameters:
        - `overall_state`: an `ArrayOverallState` instance

        """
        transition_info = markov_chain.last_transition_info
        overall_state = markov_chain.current_state
        parameters_state = overall_state.parameters_state
        links_state = overall_state.links_state
        record = IndependentTermsStateRecord(
                transition_info[0],
                transition_info[1],
                transition_info[2],
                transition_info[3],
                transition_info[4],
                parameters_state.link_false_pos,
                parameters_state.link_false_neg,
                parameters_state.link_prior,
                links_state.calc_num_selected_links(),
                links_state.calc_num_unselected_links(),
                links_state.calc_num_selected_active_interactions(),
                links_state.calc_num_selected_inactive_interactions(),
                links_state.calc_num_unselected_active_interactions(),
                links_state.calc_num_unselected_inactive_interactions(),
                links_state.calc_num_selected_terms(),
                links_state.calc_num_unselected_terms(),
                parameters_state.term_prior,
        )
        self._transitions_data.append(record)


class GenesBasedStateRecorder(TermsBasedStateRecorder):
    """Similar to `PLNStateRecorder`, however, adapted to record the
    state of `GenesBasedOverallState` instances.

    """
    def record_transition(self, markov_chain):
        """Record all the numbers of the current state.

        :Parameters:
        - `overall_state`: an `ArrayOverallState` instance

        """
        transition_info = markov_chain.last_transition_info
        overall_state = markov_chain.current_state
        parameters_state = overall_state.parameters_state
        links_state = overall_state.links_state
        record = GenesBasedStateRecord(
                transition_info[0],
                transition_info[1],
                transition_info[2],
                transition_info[3],
                transition_info[4],
                parameters_state.link_false_pos,
                parameters_state.link_false_neg,
                parameters_state.link_prior,
                links_state.calc_num_selected_links(),
                links_state.calc_num_unselected_links(),
                links_state.calc_num_selected_active_interactions(),
                links_state.calc_num_selected_inactive_interactions(),
                links_state.calc_num_unselected_active_interactions(),
                links_state.calc_num_unselected_inactive_interactions(),
                links_state.calc_num_selected_terms(),
                links_state.calc_num_unselected_terms(),
                parameters_state.term_prior,
                parameters_state.term_false_pos,
                parameters_state.term_false_neg,
                links_state.calc_num_selected_active_genes(),
                links_state.calc_num_selected_inactive_genes(),
                links_state.calc_num_unselected_active_genes(),
                links_state.calc_num_unselected_inactive_genes(),
        )
        self._transitions_data.append(record)

