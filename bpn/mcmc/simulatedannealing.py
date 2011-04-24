#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright (c) 2011 Chris D. Lasher & Phillip Whisenhunt
#
# This software is released under the MIT License. Please see
# LICENSE.txt for details.


"""Simulated Annealing for BPN."""

import math
import random
import states

import logging
logger = logging.getLogger('bpn.sabpn.simulatedannealing')

from defaults import NUM_STEPS


class SimulatedAnnealing(object):
    pass


class PLNSimulatedAnnealing(SimulatedAnnealing):
    """A class representing Simulated Annealing for process linkage
    networks.

    """
    def __init__(
            self,
            annotated_interactions,
            active_gene_threshold,
            transition_ratio,
            num_steps=NUM_STEPS,
            selected_links=None,
            alpha=None,
            beta=None,
            link_prior=None,
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
        - `num_steps`: the number of steps to take anneal
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
        self.num_steps = num_steps
        # Don't hard-code this value. Pass it as a parameter at
        # initialization or put it in defaults.py (or do both). -CDL
        self.temperature = 100000
        self.step_size = 1.0 / self.num_steps


    def next_state(self):
        """Move to the next state in Simulated Annealing.

        This method creates a proposed state for a transition; it then
        assesses the "fitness" of this new state by comparing the
        likelihood of the proposed state to the likelihood of the
        current state as a (log of the) ratio of the two likelihoods.

        If this ratio is greater than 0 or e ^ -deltaE / temperature
        is better then a random value from 0 to 1, then we accept the
        proposed state. Therefore, when the temperature is very
        large we are more likely to accept a worse state. Else we
        don't accept the new state.

        """
        proposed_state = self.current_state.create_new_state()
        proposed_transition_type = proposed_state._delta[0]
        current_log_likelihood = (
                self.current_state.calc_log_likelihood())
        proposed_log_likelihood = (
                proposed_state.calc_log_likelihood())
        log_delta_e = (proposed_log_likelihood -
                current_log_likelihood)

        # Is the new solution better? -PJW
        #
        # Move the math.exp part into log-space. This will crash when
        # the delta_e gets too large. -CDL
        if (log_delta_e > 0) or (
                math.exp(-log_delta_e/self.temperature) > random.random()):
            print "Accepted new state"
            self.current_state = proposed_state
            logger.debug("Accepted proposed state.")
            log_state_likelihood = proposed_log_likelihood
        else:
            print "Reject Random state"
            logger.debug("Rejected proposed state.")
            log_state_likelihood = current_log_likelihood

        logger.debug("Log of state likelihood: %s" % (
                log_state_likelihood))

        # TODO: Phillip, record the state information to the state
        # recorder (refer to the chains.py classes). We will want the
        # transition information later.


    def run(self):
        """Anneal.

        Temperature degrades by a percentage rather then a
        specific step size. This allows the temperature
        to gradually cool off and resembles more of a
        real cooling down. We use 1 - stepsize to allow
        the user to still input the number of steps.
        While the number of steps does not directly correlate
        to how many steps will be annealed, the number of
        states visited is still directly tied to the step
        size.

        """
        # Don't hard-code this value. Pass it as a parameter at
        # initialization or put it in defaults.py (or do both). -CDL
        while self.temperature > 0.1:
            self.next_state()
            self.temperature *= 1 - self.step_size

            # TODO: Log the progress of the Simulated Anealing (percent
            # complete). See the run method in chains.py. To do that,
            # convert this loop from a while-loop to a for-loop (you
            # know how many steps you'll perform).
            #
            # You should then add an assert statement at the end to make
            # sure that you've reached your desired ending temperature
            # at the end of those steps, too. (You should probably not
            # use ``current_temp == desired_final_temp`` because of
            # floating-point error; instead, check that the difference
            # between them is sufficiently small. -CDL


class ArraySimulatedAnnealing(PLNSimulatedAnnealing):
    """Similar to `PLNSimulatedAnnealing`, but using `numpy`
    arrays to track state information.

    """
    def __init__(
            self,
            annotated_interactions,
            active_gene_threshold,
            transition_ratio,
            num_steps=NUM_STEPS,
            selected_links_indices=None,
            alpha=None,
            beta=None,
            link_prior=None,
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
        - `num_steps`: the number of steps to anneal
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
                alpha=alpha,
                beta=beta,
                link_prior=link_prior,
                parameters_state_class=parameters_state_class,
                links_state_class=links_state_class
        )
        self.num_steps = num_steps
        self.last_transition_info = None
        # Don't hard-code this value. Pass it as a parameter at
        # initialization or put it in defaults.py (or do both). -CDL
        self.temperature = 100000
        self.step_size = 1.0 / self.num_steps

