#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright (c) 2010-2011 Christopher D. Lasher
#
# This software is released under the MIT License. Please see
# LICENSE.txt for details.


"""A program to find contextually linked biological processes.

This method extends that of Biological Process Linkage Networks by
Dotan-Cohen et al. [1] An implementation of the original BPLN is
available from bpln.py.

1. Dotan-Cohen, D., Letovsky, S., Melkman, A.A. & Kasif, S. Biological
Process Linkage Networks. PLoS ONE 4, e5313 (2009).

"""


OUTFILE_FIELDS = (
        'term1',
        'term2',
        'term1_size',
        'neighbors_of_term1',
        'term2_size',
        'intersection',
        'union',
        'jaccard',
        'score',
        'exceedances',
        'p_value'
)

NUM_PERMUTATIONS = 10000

# Estimate p-values after this many iterations
ESTIMATE_INTERVAL = 500

# The number of records of results to buffer before outputting.
RESULTS_BUFFER_SIZE = 10


import collections
import math
import random

from convutils import convutils

import cli

# Configure all the logging stuff
import logging
logger = logging.getLogger('bpn.cbpn')


def translate_gene_ids(
        annotated_genes,
        index_dict,
        randomized_genes_list
    ):
    """Returns a list of the translated gene (product) IDs from a given
    collection of original gene (product) IDs.

    NOTE: If `annotated_genes` is a `set` or `dict`, no order of
    translation is guaranteed.

    :Parameters:
    - `annotated_genes`: an iterable of gene (product) IDs
    - `index_dict`: a dictionary with gene (product) IDs as keys and the
      appropriate index in the randomized list
    - `randomized_genes_list`: a list of gene (product) IDs which have
      been randomly shuffled

    """
    translated_ids = [randomized_genes_list[index_dict[gene_id]] for
            gene_id in annotated_genes]
    return translated_ids


def _calculate_mean_expression_value(interactions_graph):
    """Calculates the mean expression value for the genes in the graph.

    Assumes the expression values are log10-transformed.

    Returns the mean expression value.

    :Parameters:
    - `interactions_graph`: graph containing the gene-gene or gene
      product-gene product interactions

    """
    weights = [value['weight'] for value in
            interactions_graph.node.itervalues()]
    mean_expression_value = sum(weights) / len(weights)
    return mean_expression_value


def calculate_node_z_prime(
        node,
        interactions_graph,
        selected_nodes,
        score_correction=False,
        mean_expression_value=None
    ):
    """Calculates a z'-score for a given node.

    The z'-score is based on the z-scores (weights) of the neighbors of
    the given node, and proportional to the z-score (weight) of the
    given node. Specifically, we find the maximum z-score of all
    neighbors of the given node that are also members of the given set
    of selected nodes, multiply this z-score by the z-score of the given
    node, and return this value as the z'-score for the given node.

    If the given node has no neighbors in the interaction graph, the
    z'-score is defined as zero.

    Returns the z'-score as zero or a positive floating point value.

    For the purposes of Biological Process Linkage Networks,
    `selected_nodes` is a set of nodes annotated by some term.

    :Parameters:
    - `node`: the node for which to compute the z-prime score
    - `interactions_graph`: graph containing the gene-gene or gene
      product-gene product interactions
    - `selected_nodes`: a `set` of nodes fitting some criterion of
      interest (e.g., annotated with a term of interest)
    - `score_correction`: if `True`, perform correction on scores using
      an expected value computed from the mean expression value
      [default: `False`]
    - `mean_expression_value`: the mean expression value for all genes
      in the interaction graph [NOTE: must be provided if
      `score_correction` is `True`]

    """
    if score_correction and (mean_expression_value is None):
        raise ValueError("mean_expression_value may not be none if "
                "score correction is desired.")

    node_neighbors = interactions_graph[node]
    interactions_graph_nodes = interactions_graph.node
    #neighbor_z_scores = [
            #interactions_graph[node][neighbor]['weight'] *
            #interactions_graph_nodes[neighbor]['weight'] for
            #neighbor in node_neighbors if neighbor in selected_nodes
    #]
    neighbor_z_scores = []
    for neighbor in node_neighbors:
        if neighbor in selected_nodes:
            edge_weight = interactions_graph[node][neighbor]['weight']
            neighbor_expr = interactions_graph_nodes[neighbor]['weight']
            contribution = edge_weight * neighbor_expr
            neighbor_z_scores.append(contribution)
    try:
        max_z_score = max(neighbor_z_scores)
    # max() throws a ValueError if its argument has no elements; in this
    # case, we need to set the max_z_score to zero
    except ValueError, e:
        # Check to make certain max() raised this error
        if 'max()' in e.args[0]:
            max_z_score = 0
        else:
            raise e

    z_prime = interactions_graph_nodes[node]['weight'] * max_z_score

    if score_correction:
        logger.debug("Correcting z_prime.")
        edge_weights = [
                interactions_graph[node][neighbor]['weight'] for
                neighbor in node_neighbors if neighbor in selected_nodes
        ]
        try:
            max_edge_weight = max(edge_weights)
        except ValueError, e:
            if 'max()' in e.args[0]:
                max_edge_weight = 0
            else:
                raise e

        if max_edge_weight:
            correction = max_edge_weight * (mean_expression_value ** 2)
            z_prime -= correction

    return z_prime


def calculate_overlap_z_score(
        interactions_graph,
        annotated_i_genes,
        neighbors_of_i,
        annotated_j_genes,
        score_correction=False,
        mean_expression_value=None
    ):
    """Calculate the z-score for neighbors of genes (or products)
    annotated with term i that are annotated with term j (and not also
    with term i).

    The z-score is defined as the sum of the z'-scores for all nodes
    that meet all three following criteria:

    - are annotated with term j
    - neighbor at least one node annotated with term i
    - are not annotated with term i

    :Parameters:
    - `interactions_graph`: graph containing the gene-gene or gene
      product-gene product interactions
    - `annotated_i_genes`: the set of genes annotated with term i
    - `neighbors_of_i`: nodes neighboring those annotated with term i
      and not annotated with term i, themselves
    - `annotated_j_genes`: the set of genes annotated with term j
    - `score_correction`: if `True`, perform correction on scores using
      an expected value computed from the mean expression value
      [default: `False`]
    - `mean_expression_value`: the mean expression value for all genes
      in the interaction graph [NOTE: must be provided if
      `score_correction` is `True`]

    """
    neighbors_annotated_with_j = neighbors_of_i.intersection(
            annotated_j_genes)

    overlap_z_score = 0

    for neighbor in neighbors_annotated_with_j:
        node_z_prime = calculate_node_z_prime(
                neighbor,
                interactions_graph,
                annotated_i_genes,
                score_correction,
                mean_expression_value
        )
        overlap_z_score += node_z_prime

    return overlap_z_score


def calculate_num_exceedances(observed_value, random_distribution):
    """Determines the number of values from a random distribution which
    exceed or equal the observed value.

    :Parameters:
    - `observed_value`: the value that was calculated from the original
      data
    - `random_distribution`: an iterable of values computed from
      randomized data

    """
    num_exceedances = len([score for score in random_distribution
            if score >= observed_value])
    return num_exceedances


def estimate_p_value(num_exceedances, num_permutations):
    """Estimate the p-value for a permutation test.

    The estimated p-value is simply `M / N`, where `M` is the number of
    exceedances, and `M > 10`, and `N` is the number of permutations
    performed to this point (rather than the total number of
    permutations to be performed, or possible). For further details, see
    Knijnenburg et al. [1]

    :Parameters:
    - `num_exceedances`: the number of values from the random
      distribution that exceeded the observed value
    - `num_permutations`: the number of permutations performed prior to
      estimation

    """
    return float(num_exceedances) / num_permutations


def compute_linked_significance(
        interactions_graph,
        annotations_dict,
        annotation_term_i,
        annotation_term_j,
        num_permutations,
        use_estimation=True,
        score_correction=False,
        mean_expression_value=None
    ):
    """

    NOTE: If `use_estimation` is `True`, will periodically assess the
    possible significance of the pair. If the at one of these
    checkpoints the function determines the observed score for the pair
    is very likely insignificant, the full count of permutations will
    not be performed, and instead, an estimated p-value will be
    calculated. Thus, by setting `use_estimation` to `True`, you may
    save large amounts of time by avoiding time-consuming permutations
    for pairs which will have insignificant p-values. This technique and
    the estimation calculation are based on a publication by Knijnenburg
    et al. [1]

    Returns the observed score for the link, the number of scores from
    the random distribution equal to or greater than the observed score,
    and the p-value (a floating point number between zero and one,
    inclusive) for that score.

    1. Knijnenburg, T.A., Wessels, L.F.A., Reinders, M.J.T. &
    Shmulevich, I. Fewer permutations, more accurate P-values.
    Bioinformatics 25, i161-168 (2009).

    :Parameters:
    - `interactions_graph`: graph containing the gene-gene or gene
      product-gene product interactions
    - `annotations_dict`: a dictionary with annotation terms as keys and
      `set`s of genes as values
    - `annotation_term_i`: the first annotation term
    - `annotation_term_j`: the second annotation term for which we want
      to determine if this term "is linked to" term i
    - `num_permutations`: maximum number of permutations to perform
      [NOTE: see `use_estimation`]
    - `use_estimation`: estimate significances for pairs which are
      unlikely to have significant scores [default: `True`] [NOTE: using
      this option will not guarantee that the number of permutations
      specified by `num_permutations` will be performed.]

    """
    # We need to have a list that we can shuffle to get randomization,
    # esesntially permuting the gene labels of the nodes, and also
    # changing gene membership within the sets of genes the annotation
    # terms annotate. By using this list, we also ensure that the true
    # path rule is followed: a gene that is annotated with some GO term
    # will also be guaranteed to be annotated with the GO term's parent
    # terms.
    #
    # Once the list is shuffled, we "translate" a gene ID into the new
    # random gene ID by looking up the original gene ID in a dictionary
    # and get its index in a list. At this index will be the new ID to
    # consider as the node label and the ID of the gene within annotated
    # sets.
    genes_list = interactions_graph.nodes()
    genes_list_index_dict = dict([(k, v) for (v, k) in
        enumerate(genes_list)])

    annotated_i_genes = annotations_dict[annotation_term_i]
    # Get neighbors of genes annotated with term i that are not
    # annotated with term i themselves
    neighbors_of_i = interactions_graph.get_neighbors_of_annotated(
                annotated_i_genes)
    # Find the overlap z-score for the data as it was observed.
    observed_overlap_z_score = calculate_overlap_z_score(
            interactions_graph,
            annotated_i_genes,
            neighbors_of_i,
            annotations_dict[annotation_term_j],
            score_correction,
            mean_expression_value
    )

    num_exceedances = 0
    p_value = None

    for i in xrange(num_permutations):
        # If we're permitted to estimate the p-value, we will check
        # periodically to determine if we need to do the estimation.
        if use_estimation:
            if not i % ESTIMATE_INTERVAL:
                # Look to see how many values exceeded the observed. If
                # it was ten or greater, we can go ahead and estimate
                # the p-value.
                if num_exceedances >= 10:
                    estimated_p_value = estimate_p_value(
                            num_exceedances, i)
                    logger.debug("Estimated p-value of %f for %s, %s" %
                            (estimated_p_value, annotation_term_i,
                                annotation_term_j)
                    )
                    return (observed_overlap_z_score, num_exceedances,
                            estimated_p_value)

        # Shuffle the genes, effectively assigning the annotation terms
        # to new random sets of genes. Note that the true-path rule will
        # still apply when annotation terms are of the Gene Ontology.
        random.shuffle(genes_list)
        # Get all the genes newly annotated with term j
        shuffled_annotated_j_genes = set(translate_gene_ids(
                annotations_dict[annotation_term_j],
                genes_list_index_dict,
                genes_list
        ))
        # Compute the new score for the shuffled genes
        randomized_overlap_z_score = calculate_overlap_z_score(
                interactions_graph,
                annotated_i_genes,
                neighbors_of_i,
                shuffled_annotated_j_genes,
                score_correction,
                mean_expression_value
        )
        if randomized_overlap_z_score >= observed_overlap_z_score:
            num_exceedances += 1

    # The most precise a p-value we can predict is not 0, but 1 / N
    # where N is the number of permutations.
    if not num_exceedances:
        num_adjusted_exceedances = 1
    else:
        num_adjusted_exceedances = num_exceedances
    p_value = float(num_adjusted_exceedances) / num_permutations

    return observed_overlap_z_score, num_exceedances, p_value


def compute_link_statistics(
        interactions_graph,
        annotations_dict,
        annotation_term_i,
        annotation_term_j
    ):
    """Compute basic statistics about the sets of genes involved in a
    link.

    Returns a dictionary with the following key-value pairs:

    - `'term1_size'`: the number of genes contained in the set of
      genes annotated with the first term
    - `'neighbors_of_term1'`: the number of genes contained in
      the set of genes which are neighbors of genes annotated with the
      first term, and not annotated with that term themselves
    - `'term2_size'`: the number of genes contained in the
      set of genes annotated with the second term
    - `'intersection'`: the number of genes annotated with the second term
      that are neighbors of genes annotated with the first term (and not
      annotated with first term, themselves); i.e., the size of the
      intersection of the neighbors with genes annotated with the second
      term
    - `'union'`: the number of genes which are either neighbors of genes
      annotated with the first term or which are annotated with the
      second term
    - `'jaccard'`: the Jaccard index for the neighbors of genes
      annotated by the first term and genes annotated by the second term

    :Parameters:
    - `interactions_graph`: graph containing the gene-gene or gene
      product-gene product interactions
    - `annotations_dict`: a dictionary with annotation terms as keys and
      `set`s of genes as values
    - `annotation_term_i`: the first annotation term
    - `annotation_term_j`: the second annotation term for which we want
      to determine if this term "is linked to" term i

    """
    annotated_i_genes = annotations_dict[annotation_term_i]
    # Get neighbors of genes annotated with term i that are not
    # annotated with term i themselves
    neighbors_of_i = interactions_graph.get_neighbors_of_annotated(
                annotated_i_genes)
    annotated_j_genes = annotations_dict[annotation_term_j]
    intersection_size = len(neighbors_of_i.intersection(
            annotated_j_genes))
    union_size = len(neighbors_of_i.union(annotated_j_genes))
    jaccard = intersection_size / float(union_size)

    stats = {
        'term1_size': len(annotated_i_genes),
        'neighbors_of_term1': len(neighbors_of_i),
        'term2_size': len(annotated_j_genes),
        'intersection': intersection_size,
        'union': union_size,
        'jaccard': jaccard
    }
    return stats


def compute_significance_for_pairs(
        pairs,
        interactions_graph,
        annotations_dict,
        num_permutations,
        use_estimation=True,
        score_correction=False
    ):
    """Compute the significance of whether term i is linked to term j
    for all pairs (i, j) given.

    Yields a a tuple of the pair of terms (i, j) as the first item, and
    a dictionary of their calculated statistics as the second item.

    NOTE: This is a generator; it will yield significance values until
    it has calculated them for all pairs.

    NOTE: See `compute_pair_significance()` for details on
    `use_estimation`.

    :Parameters:
    - `pairs`: an iterable of pairs of annotation terms
    - `interactions_graph`: graph containing the gene-gene or gene
      product-gene product interactions
    - `annotations_dict`: a dictionary with annotation terms as keys and
      `set`s of genes as values
    - `num_permutations`: maximum number of permutations to perform
      [NOTE: see `use_estimation`]
    - `use_estimation`: estimate significances for pairs which are
      unlikely to have significant scores [default: `True`] [NOTE: using
      this option will not guarantee that the number of permutations
      specified by `num_permutations` will be performed.]
    - `score_correction`: if `True`, perform correction on scores using
      an expected value computed from the mean expression value
      [default: `False`]

    """
    if score_correction:
        mean_expression_value = _calculate_mean_expression_value(
                interactions_graph)
    else:
        mean_expression_value = None
    for annotation_term_i, annotation_term_j in pairs:
        logger.debug("Calculating significance for (%s, %s)" % (
                annotation_term_i, annotation_term_j))
        observed_score, num_exceedances, linked_p_value = \
                compute_linked_significance(
                        interactions_graph,
                        annotations_dict,
                        annotation_term_i,
                        annotation_term_j,
                        num_permutations,
                        use_estimation,
                        score_correction,
                        mean_expression_value
        )
        link_statistics = compute_link_statistics(
                interactions_graph,
                annotations_dict,
                annotation_term_i,
                annotation_term_j
        )
        link_statistics['score'] = observed_score
        link_statistics['exceedances'] = num_exceedances
        link_statistics['p_value'] = linked_p_value
        yield (annotation_term_i, annotation_term_j), link_statistics


def compute_significance_for_pairs_edge_swap(
        pairs,
        interactions_graph,
        annotations_dict,
        num_permutations,
        num_edge_swap_events,
        use_estimation=True,
        score_correction=False
    ):
    """Compute the significance of whether term i is linked to term j
    for all pairs (i, j) given by randomly sampling the space of graphs
    of identical degree distributions.

    In this version, a number of graphs are constructed equal to
    `num_permutations` such that each node in the random graph has the
    exact same degree as the node in the original graph, but where the
    edges have been swapped.

    The parameter `num_edge_swap_events` directly controls how many
    "edge swap events" occur to produce each new random graph from the
    original. This parameter is multiplied by the number of edges
    presently in the graph to get the total number of iterations of edge
    swap events (e.g., for a graph with 100 edges, given
    `num_edge_swap_events = 100`, 10,000 edge swap events will be
    performed to generate each random graph.

    Yields a a tuple of the pair of terms (i, j) as the first item, and
    a dictionary of their calculated statistics as the second item.

    NOTE: This is a generator; it will yield significance values until
    it has calculated them for all pairs.

    NOTE: See `compute_pair_significance()` for details on
    `use_estimation`.

    :Parameters:
    - `pairs`: an iterable of pairs of annotation terms
    - `interactions_graph`: graph containing the gene-gene or gene
      product-gene product interactions
    - `annotations_dict`: a dictionary with annotation terms as keys and
      `set`s of genes as values
    - `num_permutations`: maximum number of permutations to perform
      [NOTE: see `use_estimation`]
    - `num_edge_swap_events`: the number of edge swap events desired to
      produce each random graph. [NOTE: this number is multiplied by the
      number of edges in the `interactions_graph` to get the total number
      of edge swap events.]
    - `use_estimation`: estimate significances for pairs which are
      unlikely to have significant scores [default: `True`] [NOTE: using
      this option will not guarantee that the number of permutations
      specified by `num_permutations` will be performed.]

    """
    if not isinstance(pairs, list):
        pairs = list(pairs)
    # A dictionary with terms (i, j) as the keys and their link
    # statistics as values.
    pair_statistics = {}

    # A dictionary with terms (i, j) as the keys, and the number of
    # scores observed to be equal or greater than the original observed
    # score
    #
    # NOTE: By default, int() creates a `0`. We use this trick so that
    # the dictionary returns a value of 0 if an exceedance has not been
    # calculated yet, so we can directly add to it.
    pair_exceedance_counts = collections.defaultdict(int)

    if score_correction:
        mean_expression_value = _calculate_mean_expression_value(
                interactions_graph)
    else:
        mean_expression_value = None

    for annotation_term_i, annotation_term_j in pairs:
        # Compute the original scores
        logger.debug("Calculating the original scores for (%s, %s)" % (
                annotation_term_i, annotation_term_j))

        annotated_i_genes = annotations_dict[annotation_term_i]
        # Get neighbors of genes annotated with term i that are not
        # annotated with term i themselves
        neighbors_of_i = interactions_graph.get_neighbors_of_annotated(
                    annotated_i_genes)
        # Find the overlap z-score for the data as it was observed.
        observed_overlap_z_score = calculate_overlap_z_score(
                interactions_graph,
                annotated_i_genes,
                neighbors_of_i,
                annotations_dict[annotation_term_j],
                score_correction,
                mean_expression_value
        )

        link_statistics = compute_link_statistics(
                interactions_graph,
                annotations_dict,
                annotation_term_i,
                annotation_term_j
        )
        link_statistics['score'] = observed_overlap_z_score

        pair_statistics[(annotation_term_i, annotation_term_j)] = \
                link_statistics

    # Now we must compute the score distributions.
    #
    # Use this variable to track our progress.
    previous_percent_done = 0
    for i in xrange(num_permutations):
        randomized_edges_graph = interactions_graph.randomize_by_edge_swaps(
                num_edge_swap_events)
        for pair in pairs:
            annotation_term_i, annotation_term_j = pair
            observed_overlap_z_score = pair_statistics[pair]['score']
            # If we're permitted to estimate the p-value, we will check
            # periodically to determine if we need to do the estimation.
            if use_estimation:
                if not i % ESTIMATE_INTERVAL:
                    # If it was ten or greater, we can go ahead and
                    # estimate the p-value.
                    num_exceedances = pair_exceedance_counts[pair]
                    if num_exceedances >= 10:
                        estimated_p_value = estimate_p_value(
                                num_exceedances, i)
                        logger.debug("Estimated p-value of %f for %s, %s" %
                                (estimated_p_value, annotation_term_i,
                                    annotation_term_j)
                        )
                        pair_statistics[pair]['p_value'] = \
                                estimated_p_value
                        pair_statistics[pair]['exceedances'] = \
                                num_exceedances
                        # Now remove this from the list so we don't
                        # iterate over it anymore.
                        pairs.remove(pair)
                        del pair_exceedance_counts[pair]

            logger.debug("Calculating random score for (%s, %s)" % (
                    annotation_term_i, annotation_term_j))

            annotated_i_genes = annotations_dict[annotation_term_i]
            # Get neighbors of genes annotated with term i that are not
            # annotated with term i themselves
            neighbors_of_i = \
                    randomized_edges_graph.get_neighbors_of_annotated(
                            annotated_i_genes)
            # Find the overlap z-score for the data as it was observed.
            randomized_overlap_z_score = calculate_overlap_z_score(
                    randomized_edges_graph,
                    annotated_i_genes,
                    neighbors_of_i,
                    annotations_dict[annotation_term_j],
                    score_correction,
                    mean_expression_value
            )
            #pair_distributions[pair].append(randomized_overlap_z_score)
            if randomized_overlap_z_score >= observed_overlap_z_score:
                pair_exceedance_counts[pair] += 1

        # Periodically report how far we've gotten.
        percent_done = int(
                math.floor(100 * (i + 1) / float(num_permutations))
        )
        if percent_done > previous_percent_done:
            previous_percent_done = percent_done
            logger.info("%d iterations of %d (%d%%) performed." %
                    (i + 1, num_permutations, percent_done))

    # For all the un-estimated pairs, finish up by calculating their
    # p-values
    for pair in pairs:
        observed_overlap_z_score = pair_statistics[pair]['score']
        num_exceedances = pair_exceedance_counts[pair]
        # The most precise a p-value we can predict is not 0, but 1 / N
        # where N is the number of permutations.
        if not num_exceedances:
            num_adjusted_exceedances = 1
        else:
            num_adjusted_exceedances = num_exceedances
        p_value = float(num_adjusted_exceedances) / num_permutations
        pair_statistics[pair]['p_value'] = p_value
        pair_statistics[pair]['exceedances'] = num_exceedances

    return pair_statistics


def write_results_to_csv(out_csvfile, results):
    """Writes the significance results to a CSV file.

    :Parameters:
    - `out_csvfile`: a `csv.DictWriter` instance with these fields:
      `neighbors_of_annotation`, `other_annotation`, `p_value`.
    - `results`: a list or iterator of results that are tuples, with the
      first element being a tuple of (`annotation_term_i`,
      `annotation_term_j`), and the second element being the p-value for
      the significance of linkage

    """
    output_dicts = []
    for (term_i, term_j), stats in results:
        output_dict = {
                'term1': term_i,
                'term2': term_j,
        }
        output_dict.update(stats)
        output_dicts.append(output_dict)
    out_csvfile.writerows(output_dicts)


def calculate_and_output_results_resampling(
        outfileh,
        pairs,
        total_pairs,
        interactions_graph,
        annotations_dict,
        num_permutations,
        use_estimation=True,
        score_correction=False
    ):
    """Calculates the significance of a link between each given pair of
    annotation terms using resampling of genes annotated by the second
    term.

    :Parameters:
    - `outfileh`: a file handle to a file for output
    - `pairs`: an iterable of pairs of annotation terms
    - `total_pairs`: the number of total annotation pairs to be
      processed
    - `interactions_graph`: graph containing the gene-gene or gene
      product-gene product interactions
    - `annotations_dict`: a dictionary with annotation terms as keys and
      `set`s of genes as values
    - `num_permutations`: maximum number of permutations to perform
      [NOTE: see `use_estimation`]
    - `use_estimation`: estimate significances for pairs which are
      unlikely to have significant scores [default: `True`] [NOTE: using
      this option will not guarantee that the number of permutations
      specified by `num_permutations` will be performed.]
    - `score_correction`: if `True`, perform correction on scores using
      an expected value computed from the mean expression value
      [default: `False`]

    """
    # Create the output CSV file.
    csv_writer = convutils.make_csv_dict_writer(outfileh, OUTFILE_FIELDS)
    # Set up the test results iterator.
    significance_results = compute_significance_for_pairs(
            pairs,
            interactions_graph,
            annotations_dict,
            num_permutations,
            use_estimation,
            score_correction
    )

    results_for_output = []
    # Output the test results.
    for i, pair_results in enumerate(significance_results):
        results_for_output.append(pair_results)
        # periodically flush results to disk
        if not ((i + 1) % RESULTS_BUFFER_SIZE):
            percent_done = int(
                    math.floor(100 * (i + 1) / float(total_pairs))
            )
            logger.info("%d of %d (%d%%) pairs processed. "
                    "Writing to %s." % (i + 1, total_pairs,
                    percent_done,
                    outfileh.name)
            )

            write_results_to_csv(csv_writer, results_for_output)
            # flush the scores
            results_for_output = []
            outfileh.flush()

    logger.info("All %d pairs processed." % total_pairs)
    if results_for_output:
        logger.info("Writing to %s" % outfileh.name)
        write_results_to_csv(csv_writer, results_for_output)


def calculate_and_output_results_edge_swap(
        outfileh,
        pairs,
        total_pairs,
        interactions_graph,
        annotations_dict,
        num_permutations,
        num_edge_swap_events,
        use_estimation=True,
        score_correction=False
    ):
    """Calculates the significance of a link between each given pair of
    annotation terms using resampling of genes annotated by the second
    term.

    :Parameters:
    - `outfileh`: a file handle to a file for output
    - `pairs`: an iterable of pairs of annotation terms
    - `total_pairs`: the number of total annotation pairs to be
      processed
    - `interactions_graph`: graph containing the gene-gene or gene
      product-gene product interactions
    - `annotations_dict`: a dictionary with annotation terms as keys and
      `set`s of genes as values
    - `num_permutations`: maximum number of permutations to perform
      [NOTE: see `use_estimation`]
    - `num_edge_swap_events`: the number of edge swap events desired to
      produce each random graph. [NOTE: this number is multiplied by the
      number of edges in the `interactions_graph` to get the total number
      of edge swap events.]
    - `use_estimation`: estimate significances for pairs which are
      unlikely to have significant scores [default: `True`] [NOTE: using
      this option will not guarantee that the number of permutations
      specified by `num_permutations` will be performed.]
    - `score_correction`: if `True`, perform correction on scores using
      an expected value computed from the mean expression value
      [default: `False`]

    """
    # Create the output CSV file.
    csv_writer = convutils.make_csv_dict_writer(outfileh, OUTFILE_FIELDS)
    pair_statistics = compute_significance_for_pairs_edge_swap(
            pairs,
            interactions_graph,
            annotations_dict,
            num_permutations,
            num_edge_swap_events,
            use_estimation,
            score_correction
    )
    logger.info("Writing results to %s" % outfileh.name)
    write_results_to_csv(csv_writer, pair_statistics.iteritems())


def main(argv=None):
    cli_parser = cli.ContextualCli()
    input_data = cli_parser.parse_args(argv)

    logger.info("Calculating significance of %d links." %
            input_data.num_links)
    if input_data.estimate:
        logger.info("Estimating p-values when possible.")
    if input_data.score_correction:
        logger.info("Correcting pair scores using mean expression.")

    logger.info("Calculating link significances. Please be patient.")

    if input_data.edge_swaps:
        logger.info("Creating random distribution by edge-swapping.")
        calculate_and_output_results_edge_swap(
                input_data.links_outfile,
                input_data.links,
                input_data.num_links,
                input_data.interactions_graph,
                input_data.annotations_dict,
                input_data.num_permutations,
                input_data.edge_swaps,
                input_data.estimate,
                input_data.score_correction
        )
    else:
        logger.info("Creating random distribution by resampling.")
        calculate_and_output_results_resampling(
                input_data.links_outfile,
                input_data.links,
                input_data.num_links,
                input_data.interactions_graph,
                input_data.annotations_dict,
                input_data.num_permutations,
                input_data.estimate,
                input_data.score_correction
        )

    logger.info("Finished calculating link significances.")


if __name__ == '__main__':
    main()

