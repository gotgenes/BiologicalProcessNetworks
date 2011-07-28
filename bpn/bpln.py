#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright (c) 2010-2011 Christopher D. Lasher
#
# This software is released under the MIT License. Please see
# LICENSE.txt for details.


"""An implementation of Biological Process Linkage Networks [1].

1. Dotan-Cohen, D., Letovsky, S., Melkman, A.A. & Kasif, S. Biological
Process Linkage Networks. PLoS ONE 4, e5313 (2009).

"""


# The number of records of results to buffer before outputting.
RESULTS_BUFFER_SIZE = 100000

OUTFILE_FIELDS = (
        'term1',
        'term2',
        'term1_size',
        'neighbors_of_term1',
        'term2_size',
        'intersection',
        'union',
        'jaccard',
        'intersection_by_neighbors_of_term1',
        'intersection_by_term2',
        'fishers_exact',
)


import math

from convutils import convutils

import cli
import statstools


# Configure all the logging stuff
import logging
logger = logging.getLogger('bpn.bpln')


def calculate_linkage_scores(
        interactions_graph,
        annotations_dict,
        annotation_pairs
    ):
    """Calculate the linkage scores of annotations to each other.

    Yields dictionaries as results. The dictionaries have the following
    key, value pairs:

    - `'set1_name'`: The annotation of the first set
    - `'set2_name'`: The annotation of the second set
    - `'p_value'`: The result of Fisher's Exact Test
    - `'jaccard'`: The Jaccard Index (Jaccard's coefficient) of
      overlap
    - `'set1_size'`: The size of (number of genes/products in) the
          first setbiological_process
    - `'set2_size'`: The size of the second set

    NOTE: This is a generator, not a function. It will continually yield
    values, one result at a time, until it has completed all
    calculations.

    NOTE: The number of genes (or products) in `interactions_graph`
    must equal the number in the union of all sets from
    `annotations_dict`, or the calculations will be incorrect.

    :Parameters:
    - `interactions_graph`: graph containing the gene-gene or gene
      product-gene product interactions
    - `annotations_dict`: a dictionary with annotation terms as keys and
      `set`s of genes as values
    - `annotation_pairs`: an iterable that contains tuples of annotation
      terms to test. If provided, only tests for these pairings will be
      performed. Order of the annotation terms in each tuple is
      important. The test is not symmetrical. For a symmetrical test,
      make sure that the iterable contains both the tuples `('term1',
      'term2')` and `('term2', 'term1')`.

    """
    # Find out the total number of genes (products), which we call the
    # "size of the universe", since it represents all possible genes
    # that may be "picked" to be part of a set.
    # NOTE: This had better be the same as that of the union of all the
    # gene sets in the annotations_dict. This is fastest to just check
    # the interactions_graph
    universe_size = len(interactions_graph)

    # We're going to keep around some variables as cache, to avoid
    # recomputing where we can.
    prev_annotation_i = None
    annotated_i_genes = None
    neighbors = None

    # For each annotation i, get the set of genes with that annotation
    for annotation_i, annotation_j in annotation_pairs:
        # If annotation_i is the same as last time, we can use the
        # cached neighborhood; otherwise, we've got to find this new
        # annotation_i's neigborhood.
        if annotation_i != prev_annotation_i:
            annotated_i_genes = annotations_dict[annotation_i]
            neighbors = interactions_graph.get_neighbors_of_annotated(
                    annotated_i_genes)

        annotated_j_genes = annotations_dict[annotation_j]

        # Find the intersection of genes annotated with j (and not
        # i) with the set of neighbors of i
        pair_scores = statstools.calculate_overlap_scores(
                neighbors,
                annotated_j_genes,
                universe_size
        )
        pair_scores['term1'] = annotation_i
        pair_scores['term2'] = annotation_j
        pair_scores['term1_size'] = len(annotated_i_genes)
        pair_scores['neighbors_of_term1'] = len(neighbors)
        pair_scores['term2_size'] = len(annotated_j_genes)
        pair_scores['intersection_by_neighbors_of_term1'] = (
                pair_scores['intersection_by_set1'])
        pair_scores['intersection_by_term2'] = (
                pair_scores['intersection_by_set2'])
        del pair_scores['set1_size']
        del pair_scores['set2_size']
        del pair_scores['intersection_by_set1']
        del pair_scores['intersection_by_set2']

        prev_annotation_i = annotation_i

        yield pair_scores


def calculate_and_output_scores(
        interactions_graph,
        annotations_dict,
        links,
        num_links,
        links_outfile
    ):
    """Calculate and output the link scores.

    :Parameters:
    - `interactions_graph`: graph containing the gene-gene or gene
      product-gene product interactions
    - `annotations_dict`: a dictionary with annotation terms as keys and
      `set`s of genes as values
    - `links`: pairs of annotation terms of which to calculate link
      scores
    - `num_links`: the number of links contained in `links`
    - `links_outfile`: file for output of link results

    """
    csv_writer = convutils.make_csv_dict_writer(
            links_outfile, OUTFILE_FIELDS)
    overlap_scores = []

    for i, link_scores in enumerate(
            calculate_linkage_scores(
                interactions_graph,
                annotations_dict,
                links
            )
        ):
        overlap_scores.append(link_scores)
        # periodically flush results to disk
        if not ((i + 1) % RESULTS_BUFFER_SIZE):
            percent_done = int(
                    math.floor(100 * (i + 1) / float(num_links))
            )
            logger.info("%d of %d (%d%%) links processed. "
                    "Writing to %s." % (i + 1, num_links,
                    percent_done,
                    links_outfile.name)
            )

            csv_writer.writerows(overlap_scores)
            # flush the scores
            overlap_scores = []

    logger.info("%d of %d (100%%) links processed."
            % (i + 1, num_links))
    logger.info("Writing to %s" % links_outfile.name)
    csv_writer.writerows(overlap_scores)


def main(argv=None):
    cli_parser = cli.BplnCli()
    input_data = cli_parser.parse_args(argv)
    logger.info(("Calculating overlap of %d combinations of gene "
            "sets.\nThis may take a while...") % input_data.num_links)

    # Create the output CSV file
    calculate_and_output_scores(
        input_data.interactions_graph,
        input_data.annotations_dict,
        input_data.links,
        input_data.num_links,
        input_data.links_outfile
    )
    logger.info("Finished calculating overlap scores.")


if __name__ == '__main__':
    main()

