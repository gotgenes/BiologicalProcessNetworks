#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright (c) 2010-2011 Christopher D. Lasher
#
# This software is released under the MIT License. Please see
# LICENSE.txt for details.


"""Contains miscellaneous statistical tests and tools."""

# Configure all the logging stuff
import logging
logger = logging.getLogger('bpn.statstools')
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)

try:
    import fisher
except ImportError:
    logger.warning("fisher not found; some functionality unavailable.")


class MissingRequiredLibrariesError(Exception):
    """Exception raised when required libraries are missing."""
    pass


def calc_sidak_correction(alpha_value, num_total_tests):
    """
    Calculates the Sidak correction for multiple hypothesis testing for
    a given alpha-value.

    See
    http://en.wikipedia.org/wiki/Bonferroni_correction#.C5.A0id.C3.A1k_correction
    for more detail.

    Returns the Sidak corrected p-value upon which one rejects the null
    hypothesis.

    :Parameters:
    - `alpha_value`: the alpha-value desired (e.g., 0.05)
    - `num_total_tests`: the total number of tests for significance
      being made

    """
    if alpha_value < 0 or alpha_value > 1:
        raise ValueError("alpha_value must be between 0 and 1, inclusive.")

    sidak_value = 1 - (1 - alpha_value) ** (1.0 / num_total_tests)
    return sidak_value


def calc_benjamini_hochberg_corrected_value(p_value, index,
        total_num_tests):
    """
    Perform the k-calculation for Benjamini-Hochberg correction.

    See
    http://en.wikipedia.org/wiki/False_discovery_rate#Independent_tests
    for more detail.

    :Parameters:
    - `p_value`: the uncorrected p-value of a test
    - `index`: where in the total list of test values this value is
      [NOTE: this should be one-index based, not zero-index (e.g.,
      the first element is index `1`)]
    - `total_num_tests`: the total number of tests done

    """
    if index > total_num_tests:
        raise ValueError("index is greater than the total number of "
                "tests")

    bh_corrected_value = p_value * total_num_tests / index
    if bh_corrected_value > 1:
        bh_corrected_value = 1.0
    return bh_corrected_value


def calc_benjamini_hochberg_corrections(p_values, num_total_tests):
    """
    Calculates the Benjamini-Hochberg correction for multiple hypothesis
    testing from a list of p-values *sorted in ascending order*.

    See
    http://en.wikipedia.org/wiki/False_discovery_rate#Independent_tests
    for more detail on the theory behind the correction.

    **NOTE:** This is a generator, not a function. It will yield values
    until all calculations have completed.

    :Parameters:
    - `p_values`: a list or iterable of p-values sorted in ascending
      order
    - `num_total_tests`: the total number of tests (p-values)

    """
    prev_bh_value = 0
    for i, p_value in enumerate(p_values):
        bh_value = calc_benjamini_hochberg_corrected_value(
                p_value, i + 1, num_total_tests)
        # One peculiarity of this correction is that even though our
        # uncorrected p-values are in monotonically increasing order,
        # the corrected p-values may actually not wind up being
        # monotonically increasing. That is to say, the corrected
        # p-value at i may be less than the corrected p-value at i-1. In
        # cases like these, the appropriate thing to do is use the
        # larger, previous corrected value.
        if bh_value < prev_bh_value:
            bh_value = prev_bh_value
        prev_bh_value = bh_value
        yield bh_value


def calculate_overlap_scores(
        set1,
        set2,
        universe_size=None
    ):
    """
    Calculates several measures of overlap of two sets.

    Returns a a dictionary of the following structure:
        `{
            'set1_size': ...,
            'set2_size': ...,
            'intersection': ...,
            'intersection_by_set1': ...,
            'intersection_by_set2': ...,
            'union': ...,
            'jaccard': ...,
            'fishers_exact': ...
        }`

    Here, ``'fishers_exact'`` represents the result of Fisher's Exact
    Test, a floating point probability between 0 and 1, inclusive;
    ``'jaccard'`` represents the Jaccard Index, also a floating point
    value between 0 and 1, inclusive. The set sizes are
    self-explanatory.  ``'intersection'`` and ``'union'`` represent the
    size of the intersection and union, respectively, of the two sets.
    ``'intersection_by_set1'`` and ``'intersection_by_set2'`` are the
    values for the size of the intersection divided by the size of the
    first and second set, respectively.

    :Parameters:
    - `set1`: `set` or `frozenset` instance
    - `set2`: `set` or `frozenset` instance
    - `universe_size`: the total number of elements in the
      "universe" of the sets; if provided, will calculate the
      right-tailed Fisher's exact test p-value for the size of the
      intersection of the two sets.

    """
    scores = {
            'set1_size': len(set1),
            'set2_size': len(set2)
    }
    intersection_size = len(set1.intersection(set2))
    union_size = len(set1.union(set2))
    scores['intersection'] = intersection_size
    scores['union'] = union_size
    if set1 or set2:
        jaccard_coefficient = intersection_size / float(union_size)
    else:
        logger.debug('Received two sets with no members.')
        jaccard_coefficient = 0
    scores['jaccard'] = jaccard_coefficient
    try:
        intersection_by_set1 = float(intersection_size) / len(set1)
    except ZeroDivisionError:
        intersection_by_set1 = 0
    try:
        intersection_by_set2 = float(intersection_size) / len(set2)
    except ZeroDivisionError:
        intersection_by_set2 = 0
    scores['intersection_by_set1'] = intersection_by_set1
    scores['intersection_by_set2'] = intersection_by_set2

    if universe_size is not None:
        try:
            fisher
        except NameError:
            raise MissingRequiredLibrariesError("The fisher package "
                    "must be installed to use this function.")
        if universe_size < 1:
            raise ValueError(("universe_size is {0}, should be at "
                    "least 1").format(universe_size))
        num_1_not_2 = len(set1) - intersection_size
        num_2_not_1 = len(set2) - intersection_size
        num_not_1_not_2 = universe_size - (intersection_size + num_1_not_2 +
                num_2_not_1)
        assert intersection_size >= 0
        assert num_1_not_2 >= 0
        assert num_2_not_1 >= 0
        assert num_not_1_not_2 >= 0
        fisher_pvalues = fisher.pvalue(
                intersection_size,
                num_1_not_2,
                num_2_not_1,
                num_not_1_not_2
        )
        scores['fishers_exact'] = fisher_pvalues.right_tail

    return scores

