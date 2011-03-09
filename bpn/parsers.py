#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright (c) 2011 Christopher D. Lasher
#
# This software is released under the MIT License. Please see
# LICENSE.txt for details.


"""Parsers for the BPN programs."""

import itertools

from convutils import convstructs, convutils

import structures

import logging
logger = logging.getLogger('bpn.parsers')


class DuplicateIDError(StandardError):
    """Error raised if a gene (product) ID is encountered more than
    once.

    """
    pass


class InvalidFormatError(ValueError):
    """Error raised for an improperly formatted file."""
    pass


class GmtFormatError(InvalidFormatError):
    """Error raised for an improperly formatted GMT file."""
    pass


def parse_interactions_file_to_graph(interactions_fileh):
    """Parse a CSV interactions file to a graph.

    The interactions file should have two columns with headings
    "interactor1" and "interactor2". If it contains an additional column
    with header "weight", values in that column will be used as the
    weight or "confidence" in the interaction. The file may have
    additional columns, which will be ignored.

    Returns a graph with genes/gene products as nodes and interactions
    as (weighted) edges.

    :Parameters:
    - `interactions_fileh`: a CSV file with a header line as the first
      line

    """
    interactions_graph = structures.EdgeSwapGraph()
    csv_reader = convutils.make_csv_reader(interactions_fileh)
    for entry in csv_reader:
        node1 = entry['interactor1']
        node2 = entry['interactor2']
        if 'weight' in entry:
            weight = float(entry['weight'])
            interactions_graph.add_edge(node1, node2, weight=weight)
        else:
            interactions_graph.add_edge(node1, node2, weight=1)

    return interactions_graph


def parse_annotations_to_dict(annotations_fileh):
    """Parse a CSV annotations file to a dictionary.

    The annotations file should have a column titled "gene_id" which has
    the gene/gene product ID, and a column titled "term" which contains
    the name or ID of a term by which the gene/product is annotated. The
    file may have additional columns, which will be ignored.

    Returns a `convstructs.TwoWaySetDict` instance with annotation as
    strings and `set`s of genes as values.

    :Parameters:
    - `annotations_fileh`: a CSV file with a header line as the first
      line

    """
    annotations_dict = convstructs.TwoWaySetDict()
    csv_reader = convutils.make_csv_reader(annotations_fileh)
    for entry in csv_reader:
        gene = entry['gene_id']
        term = entry['term']
        if term in annotations_dict:
            annotations_dict.add_item(term, gene)
        else:
            annotations_dict[term] = set([gene])

    return annotations_dict


def parse_gmt_to_dict(gmt_fileh):
    """Parse a GMT-format annotations file to a dictionary.

    The Gene Matrix Transposed (GMT) format specification can be found
    on the MSigDB and GSEA website at
    http://www.broadinstitute.org/cancer/software/gsea/wiki/index.php/Data_formats#GMT:_Gene_Matrix_Transposed_file_format_.28.2A.gmt.29

    The GMT format is a tab-separated (TSV) file. The first column
    contains the annotation term. The second column contains the
    description of the term. All following columns contain gene IDs for
    genes which that term annotates.

    Returns a `convstructs.TwoWaySetDict` instance with annotation as
    strings and `set`s of genes as values.

    :Parameters:
    - `gmt_fileh`: a GMT-format file with the annotation term as the
      first column, description as the second column, and genes
      annotated by the term in all following columns

    """
    annotations_dict = convstructs.TwoWaySetDict()
    for i, line in enumerate(gmt_fileh):
        line = line.strip()
        try:
            term, description, genes = line.split('\t', 2)
            # Skip lines where the term annotates no genes; happens
            # occasionally.
            if not genes:
                continue
            # genes is still a whole string; split it up.
            genes = genes.split('\t')
            if term in annotations_dict:
                annotations_dict.update(genes)
            else:
                annotations_dict[term] = set(genes)
        except ValueError as e:
            logger.critical("Error parsing {0} at line {1}!".format(
                    gmt_fileh.name, i+1))
            raise GmtFormatError(e)

    return annotations_dict


def parse_expression_file(expression_fileh):
    """Parse a CSV expression file.

    Returns a dictionary with gene (product) identifiers as keys and
    expression values as values.

    :Parameters:
    - `expression_fileh`: a CSV file of gene (or gene product)
      expression values. The file should have a column titled "id" which
      has the gene (or gene product) ID, and a column titled
      "expression" which gives a value for the expression level, or
      difference in expression levels.

    """
    csv_file = convutils.make_csv_reader(expression_fileh)
    expressions_dict = {}
    warned_of_multiple_values = False
    for i, entry in enumerate(csv_file):
        expression_value = float(entry['expression'])
        if entry['id'] in expressions_dict:
            # We've already seen an entry for this ID
            if not warned_of_multiple_values:
                logger.warning("WARNING! Multiple expression values "
                        "detected for at least one gene; continuing "
                        "anyway.")
                warned_of_multiple_values = True
            #msg = ("Warning: on line %d: %s has already been seen; "
                    #"continuing anyway" % (i + 1, entry['id']))
            #logger.warning(msg)
            #msg = "Error on line %d: %s has already been seen" % (
                    #i + 1, entry['id'])
            #raise DuplicateIDError(msg)
            if expression_value > expressions_dict[entry['id']]:
                expressions_dict[entry['id']] = expression_value
        else:
            expressions_dict[entry['id']] = expression_value
    return expressions_dict


def parse_selected_links_file(selected_links_fileh):
    """Parse a CSV pairs file to an iterator of links.

    The file should have no header and only two columns, where the
    annotation in the first column needs to be tested if it is "linked
    to" the annotation in the second column.

    NOTE: This is a generator; it will yield links until the file is
    completely consumed.

    :Parameters:
    - `selected_links_fileh`: a CSV file of two columns and no headers
      with annotations in the columns

    """
    csv_reader = convutils.make_csv_reader(selected_links_fileh, False)
    for i, link in enumerate(csv_reader):
        assert len(link) == 2, ("Line %d has fewer or greater than "
                "two annotation entries." % i + 1)
        yield tuple(link)


def parse_selected_terms_file(selected_terms_fileh):
    """Parse a CSV terms file to an iterator of annotation terms.

    NOTE: This is a generator; it will yield terms until the file is
    completely consumed.

    The file should have no header and only one column, with one
    term term per row.

    :Parameters:
    - `selected_terms_fileh`: a CSV file with one column and no
      headers

    """
    for line in selected_terms_fileh:
        yield line.strip()

