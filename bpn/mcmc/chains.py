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

from defaults import NUM_STEPS, BURN_IN, BROADCAST_PERCENT, \
        SUPERDEBUG, SUPERDEBUG_MODE


class MarkovChain(object):
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
            state_recorder_class=recorders.PLNStateRecorder,
            parameters_state_class=states.PLNParametersState
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
          [default: `recorders.PLNStateRecorder`]
        - `parameters_state_class`: the class of the parameters state to
          use [default: `states.PLNParametersState]`

        """
        self.current_state = states.PLNOverallState(
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
            state_recorder_class=recorders.ArrayStateRecorder,
            parameters_state_class=states.PLNParametersState,
            links_state_class=states.ArrayLinksState
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
          [default: `recorders.ArrayStateRecorder`]
        - `parameters_state_class`: the class of the parameters state to
          use [default: `states.PLNParametersState`]
        - `links_state_class`: the class of the links state to use
          [default: `states.ArrayLinksState`]

        """
        self.current_state = states.ArrayOverallState(
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


class TermsBasedMarkovChain(ArrayMarkovChain):
    """A Markov chain based on states that consider both term and link
    selections.

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
            term_prior=None,
            state_recorder_class=recorders.TermsStateRecorder,
            parameters_state_class=states.TermPriorParametersState,
            links_state_class=states.TermsAndLinksState
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
        - `term_prior`:the assumed probability we would select any one
          term; see `RandomTransitionParametersState` for more
          information
        - `state_recorder_class`: the class of the state recorder to use
          [default: `recorders.TermsStateRecorder`]
        - `parameters_state_class`: the class of the parameters state to
          use [default: `states.TermPriorParametersState`]
        - `links_state_class`: the class of the links state to use
          [default: `states.TermsAndLinksState`]

        """
        self.current_state = states.TermsBasedOverallState(
                annotated_interactions,
                active_gene_threshold,
                transition_ratio,
                selected_links_indices,
                alpha=alpha,
                beta=beta,
                link_prior=link_prior,
                term_prior=term_prior,
                parameters_state_class=parameters_state_class,
                links_state_class=links_state_class
        )
        self.state_recorder = state_recorder_class(
                annotated_interactions,
                self.current_state.parameters_state.get_parameter_distributions()
        )
        self.burn_in_steps = burn_in
        self.num_steps = num_steps
        self.burn_in_period = True
        self.last_transition_info = None


