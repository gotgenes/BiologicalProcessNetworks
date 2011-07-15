#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright (c) 2011 Christopher D. Lasher
#
# This software is released under the MIT License. Please see
# LICENSE.txt for details.


"""Markov chains for BPN."""


import math
import random

import recorders
import states

import logging
logger = logging.getLogger('bpn.mcmcbpn.chains')

from defaults import (NUM_STEPS, BURN_IN, TRANSITION_TYPE_RATIO,
        BROADCAST_PERCENT, SUPERDEBUG, SUPERDEBUG_MODE)


class MarkovChain(object):
    pass


class PLNMarkovChain(MarkovChain):
    """A class representing the Markov chain for process linkage
    networks.

    """
    def __init__(
            self,
            state_recorder,
            burn_in,
            num_steps,
            annotated_interactions,
            active_gene_threshold,
            transition_type_ratio=TRANSITION_TYPE_RATIO,
            seed_links=None,
            link_false_pos=None,
            link_false_neg=None,
            link_prior=None,
            parameters_state_class=states.PLNParametersState
        ):
        """Create a new instance.

        :Parameters:
        - `state_recorder`: a `PLNStateRecorder` instance, used to
          record the steps taken in the chain
        - `burn_in`: the number of steps to take before recording state
          information about the Markov chain (state records are
          discarded until complete)
        - `num_steps`: the number of steps to take in the Markov chain
        - `annotated_interactions`: an `AnnotatedInteractionsGraph`
          instance
        - `active_gene_threshold`: the threshold at or above which a
          gene is considered "active"
        - `transition_type_ratio`: a `float` indicating the ratio of link
          transitions to parameter transitions
        - `seed_links`: a user-defined seed of links to start as
          selected
        - `link_false_pos`: the false-positive rate for links, the
          portion of gene-gene interactions which were included, but
          shouldn't have been
        - `link_false_neg`: the false-negative rate for links, the
          portion of gene-gene interactions which weren't included, but
          should have been
        - `link_prior`: the assumed probability we would pick any one
          link as being active; see `PLNParametersState` for more
          information
        - `parameters_state_class`: the class of the parameters state to
          use [default: `states.PLNParametersState]`

        """
        self.state_recorder = state_recorder
        self.current_state = states.PLNOverallState(
                annotated_interactions,
                active_gene_threshold,
                transition_type_ratio,
                seed_links,
                link_false_pos,
                link_false_neg,
                link_prior,
                parameters_state_class

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
        self.current_step = None


    def calc_log_transition_ratio(self, proposed_state):
        """Calculates the likelihoods of the current state, the proposed
        state, and the transition ratio between them.

        Returns the current state's log-likelihood, the proposed state's
        log-likelihood, and the transition ratio between the current and
        proposed states.

        :Parameters:
        - `proposed_state`: the proposed state of transition

        """
        current_log_likelihood = (
                self.current_state.calc_log_likelihood())
        current_num_neighbors = (
                self.current_state.calc_num_neighboring_states())
        proposed_log_likelihood = (
                proposed_state.calc_log_likelihood())
        proposed_num_neighbors = (
                proposed_state.calc_num_neighboring_states())
        log_transition_ratio = (
                (proposed_log_likelihood -
                    math.log10(current_num_neighbors)) -
                (current_log_likelihood -
                    math.log10(proposed_num_neighbors))
        )
        return (current_log_likelihood, proposed_log_likelihood,
                log_transition_ratio)


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
        (current_log_likelihood, proposed_log_likelihood,
                log_transition_ratio) = self.calc_log_transition_ratio(
                        proposed_state)
        logger.debug("Log of transition ratio: %s" % (
                log_transition_ratio))
        # Remember, a very favorable state has a likelihood close to 1
        # and its log-likelihood close to 0 (i.e., very small negative
        # value); a very unfavorable state has a likelihood close to 0,
        # and a log-likelihood with a very negative value (large
        # magnitude).

        # If the proposed state is more likely than the current state,
        # the proposed state's log-likelihood will be a smaller negative
        # number than the current state's; thus log_transition_ratio
        # will be a positive value. If the proposed state is less likely
        # than the current state, the proposed state's log-likelihood
        # will be a larger negative number than the current state's;
        # thus the log_transition_ratio will be a negative value.
        #
        # If the proposed state is more likely, we want to go ahead and
        # accept it.
        if log_transition_ratio > 0:
            log_rejection_prob = None
            self.current_state = proposed_state
            logger.debug("Accepted proposed state.")
            accepted = True
            log_state_likelihood = proposed_log_likelihood
        else:
            # Otherwise, the proposed state is equally or less likely
            # than the current state, however, we may still choose it,
            # even though this may be unfavorable, to avoid local
            # maxima. We will do this by drawing a number uniformly at
            # random in the range [0, 1], as the probability we will
            # reject the state. If the log of the log-transition ratio
            # is greater than the log of the rejection probability,
            # we'll still accept the less favorable proposed state.
            log_rejection_prob = math.log10(random.random())
            logger.debug("Log rejection probability: %s" % log_rejection_prob)
            if log_transition_ratio > log_rejection_prob:
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
        # In a special case, if this is the very first step of the
        # actual recorded steps (i.e., the first step taken after the
        # burn in period), we want to make sure that it's marked as
        # "accepted" because this is the first time in the records that
        # we will have seen this state (even if it was the last state
        # seen in the burn-in)
        if self.current_step == 0:
            accepted = True
        # NOTE: if changing the order of any of these items, please make
        # sure to update code in recorders.py!!!
        # In the future, consider using a named-tuple instead of a
        # normal tuple for this data structure.
        self.last_transition_info = (
                proposed_transition_type,
                log_transition_ratio,
                log_state_likelihood,
                accepted,
                log_rejection_prob
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
            self.current_step = i
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
            state_recorder,
            burn_in,
            num_steps,
            annotated_interactions,
            active_gene_threshold,
            transition_type_ratio=TRANSITION_TYPE_RATIO,
            seed_links_indices=None,
            link_false_pos=None,
            link_false_neg=None,
            link_prior=None,
            parameters_state_class=states.PLNParametersState,
            links_state_class=states.ArrayLinksState
        ):
        """Create a new instance.

        :Parameters:
        - `state_recorder`: an `ArrayStateRecorder` instance, used to
          record the steps taken in the chain
        - `burn_in`: the number of steps to take before recording state
          information about the Markov chain (state records are
          discarded until complete)
        - `num_steps`: the number of steps to take in the Markov chain
        - `annotated_interactions`: an `AnnotatedInteractionsArray`
          instance
        - `active_gene_threshold`: the threshold at or above which a
          gene is considered "active"
        - `transition_type_ratio`: a `float` indicating the ratio of link
          transitions to parameter transitions
        - `seed_links_indices`: a user-defined seed of indices to
          links to start as selected
        - `link_false_pos`: the false-positive rate for links, the
          portion of gene-gene interactions which were included, but
          shouldn't have been
        - `link_false_neg`: the false-negative rate for links, the
          portion of gene-gene interactions which weren't included, but
          should have been
        - `link_prior`: the assumed probability we would pick any one
          link as being active; see `PLNParametersState` for more
          information
        - `parameters_state_class`: the class of the parameters state to
          use [default: `states.PLNParametersState`]
        - `links_state_class`: the class of the links state to use
          [default: `states.ArrayLinksState`]

        """
        self.state_recorder = state_recorder
        self.current_state = states.ArrayOverallState(
                annotated_interactions,
                active_gene_threshold,
                transition_type_ratio,
                seed_links_indices,
                link_false_pos,
                link_false_neg,
                link_prior,
                parameters_state_class,
                links_state_class
        )
        self.burn_in_steps = burn_in
        self.num_steps = num_steps
        self.burn_in_period = True
        self.last_transition_info = None
        self.current_step = None


class TermsBasedMarkovChain(ArrayMarkovChain):
    """A Markov chain based on states that consider both term and link
    selections.

    """
    def __init__(
            self,
            state_recorder,
            burn_in,
            num_steps,
            annotated_interactions,
            active_gene_threshold,
            transition_type_ratio=TRANSITION_TYPE_RATIO,
            seed_terms_indices=None,
            seed_links_indices=None,
            link_false_pos=None,
            link_false_neg=None,
            link_prior=None,
            term_prior=None,
            parameters_state_class=states.TermPriorParametersState,
            links_state_class=states.TermsAndLinksState
        ):
        """Create a new instance.

        :Parameters:
        - `state_recorder`: an `ArrayStateRecorder` instance, used to
          record the steps taken in the chain
        - `burn_in`: the number of steps to take before recording state
          information about the Markov chain (state records are
          discarded until complete)
        - `num_steps`: the number of steps to take in the Markov chain
        - `annotated_interactions`: an `AnnotatedInteractionsArray`
          instance
        - `active_gene_threshold`: the threshold at or above which a
          gene is considered "active"
        - `transition_type_ratio`: a `float` indicating the ratio of link
          transitions to parameter transitions
        - `seed_links_indices`: a user-defined seed of indices to
          links to start as selected
        - `link_false_pos`: the false-positive rate for links, the
          portion of gene-gene interactions which were included, but
          shouldn't have been
        - `link_false_neg`: the false-negative rate for links, the
          portion of gene-gene interactions which weren't included, but
          should have been
        - `link_prior`: the assumed probability we would pick any one
          link as being active; see `PLNParametersState` for more
          information
        - `term_prior`:the assumed probability we would select any one
          term; see `RandomTransitionParametersState` for more
          information
        - `parameters_state_class`: the class of the parameters state to
          use [default: `states.TermPriorParametersState`]
        - `links_state_class`: the class of the links state to use
          [default: `states.TermsAndLinksState`]

        """
        self.state_recorder = state_recorder
        self.current_state = states.TermsBasedOverallState(
                annotated_interactions,
                active_gene_threshold,
                transition_type_ratio,
                seed_links_indices=seed_links_indices,
                link_false_pos=link_false_pos,
                link_false_neg=link_false_neg,
                link_prior=link_prior,
                term_prior=term_prior,
                parameters_state_class=parameters_state_class,
                links_state_class=links_state_class
        )
        self.burn_in_steps = burn_in
        self.num_steps = num_steps
        self.burn_in_period = True
        self.last_transition_info = None
        self.current_step = None


class IndependentTermsBasedMarkovChain(TermsBasedMarkovChain):
    """A Markov chain based on states that consider both term and link
    selections.

    """
    def __init__(
            self,
            state_recorder,
            burn_in,
            num_steps,
            annotated_interactions,
            active_gene_threshold,
            transition_type_ratio=TRANSITION_TYPE_RATIO,
            seed_terms_indices=None,
            seed_links_indices=None,
            link_false_pos=None,
            link_false_neg=None,
            link_prior=None,
            term_prior=None,
            parameters_state_class=states.TermPriorParametersState,
            links_state_class=states.IndependentIntraTermsAndLinksState
        ):
        """Create a new instance.

        :Parameters:
        - `state_recorder`: a `TermsBasedStateRecorder` instance, used
          to record the steps taken in the chain
        - `burn_in`: the number of steps to take before recording state
          information about the Markov chain (state records are
          discarded until complete)
        - `num_steps`: the number of steps to take in the Markov chain
        - `annotated_interactions`: an `AnnotatedInteractionsArray`
          instance
        - `active_gene_threshold`: the threshold at or above which a
          gene is considered "active"
        - `transition_type_ratio`: a `float` indicating the ratio of link
          transitions to parameter transitions
        - `seed_terms_indices`: indices for the subset of terms being
          considered as "selected" initially
        - `seed_links_indices`: a user-defined seed of indices to
          links to start as selected
        - `link_false_pos`: the false-positive rate for links, the
          portion of gene-gene interactions which were included, but
          shouldn't have been
        - `link_false_neg`: the false-negative rate for links, the
          portion of gene-gene interactions which weren't included, but
          should have been
        - `link_prior`: the assumed probability we would pick any one
          link as being active; see `PLNParametersState` for more
          information
        - `term_prior`:the assumed probability we would select any one
          term; see `RandomTransitionParametersState` for more
          information
        - `parameters_state_class`: the class of the parameters state to
          use [default: `states.TermPriorParametersState`]
        - `links_state_class`: the class of the links state to use
          [default: `states.IndependentIntraTermsAndLinksState`]

        """
        self.state_recorder = state_recorder
        self.current_state = states.IndependentTermsBasedOverallState(
                annotated_interactions,
                active_gene_threshold,
                transition_type_ratio,
                seed_terms_indices=seed_terms_indices,
                seed_links_indices=seed_links_indices,
                link_false_pos=link_false_pos,
                link_false_neg=link_false_neg,
                link_prior=link_prior,
                term_prior=term_prior,
                parameters_state_class=parameters_state_class,
                links_state_class=links_state_class
        )
        self.burn_in_steps = burn_in
        self.num_steps = num_steps
        self.burn_in_period = True
        self.last_transition_info = None
        self.current_step = None


class GenesBasedMarkovChain(IndependentTermsBasedMarkovChain):
    """A Markov chain based on states that consider both term and link
    selections, where overlap is based on genes.

    """
    def __init__(
            self,
            state_recorder,
            burn_in,
            num_steps,
            annotated_interactions,
            active_gene_threshold,
            transition_type_ratio=TRANSITION_TYPE_RATIO,
            seed_terms_indices=None,
            seed_links_indices=None,
            link_false_pos=None,
            link_false_neg=None,
            link_prior=None,
            term_false_pos=None,
            term_false_neg=None,
            term_prior=None,
            parameters_state_class=states.TermsParametersState,
            links_state_class=states.GenesBasedTermsAndLinksState
        ):
        """Create a new instance.

        :Parameters:
        - `state_recorder`: a `GenesBasedStateRecorder` instance, used
          to record the steps taken in the chain
        - `burn_in`: the number of steps to take before recording state
          information about the Markov chain (state records are
          discarded until complete)
        - `num_steps`: the number of steps to take in the Markov chain
        - `annotated_interactions`: an `AnnotatedInteractionsArray`
          instance
        - `active_gene_threshold`: the threshold at or above which a
          gene is considered "active"
        - `transition_type_ratio`: a `float` indicating the ratio of link
          transitions to parameter transitions
        - `seed_terms_indices`: indices for the subset of terms being
          considered as "selected" initially
        - `seed_links_indices`: a user-defined seed of indices to
          links to start as selected
        - `link_false_pos`: the false-positive rate for links, the
          portion of gene-gene interactions which were included, but
          shouldn't have been
        - `link_false_neg`: the false-negative rate for links, the
          portion of gene-gene interactions which weren't included, but
          should have been
        - `link_prior`: the assumed probability we would pick any one
          link as being active; see `PLNParametersState` for more
          information
        - `term_false_pos`: the false-positive rate for terms, the
          portion of genes which were included, but shouldn't have been
        - `term_false_neg`: the false-negative rate for terms, the
          portion of genes which weren't included, but should have been
        - `term_prior`:the assumed probability we would select any one
          term; see `RandomTransitionParametersState` for more
          information
        - `parameters_state_class`: the class of the parameters state to
          use [default: `states.TermsParametersState`]
        - `links_state_class`: the class of the links state to use
          [default: `states.GenesBasedTermsAndLinksState`]

        """
        self.state_recorder = state_recorder
        self.current_state = states.GenesBasedOverallState(
                annotated_interactions,
                active_gene_threshold,
                transition_type_ratio,
                seed_terms_indices=seed_terms_indices,
                seed_links_indices=seed_links_indices,
                link_false_pos=link_false_pos,
                link_false_neg=link_false_neg,
                link_prior=link_prior,
                term_false_pos=term_false_pos,
                term_false_neg=term_false_neg,
                term_prior=term_prior,
                parameters_state_class=parameters_state_class,
                links_state_class=links_state_class
        )
        self.burn_in_steps = burn_in
        self.num_steps = num_steps
        self.burn_in_period = True
        self.last_transition_info = None
        self.current_step = None


