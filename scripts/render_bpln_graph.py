#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright (c) 2011 Christopher D. Lasher
#
# This software is released under the MIT License. Please see
# LICENSE.txt for details.

"""Renders BPN results to graphs of annotations based on the
significances of the links between them using Graphviz.

"""

import bisect
import itertools
import optparse
import os.path
import sys

from convutils import convutils
import pygraphviz

from bpn import parsers, qvaluetocolor

# Configure all the logging stuff
import logging
logger = logging.getLogger('render_bpn_graph')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)

CUTOFFS = (0.001, 0.01, 0.05, 0.1, 0.2)
# Do 1 - percent_value when you bisect_left on this.
PERCENT_CUTOFFS = (0.01, 0.05, 0.1, 0.2, 0.3)

BLUES = (
        (8, 69, 148),       # <= 0.001
        (33, 113, 181),     # <= 0.01
        (66, 146, 198),     # <= 0.05
        (107, 174, 214),    # <= 0.1
        (158, 202, 225),    # <= 0.2
        (198, 219, 239),    # > 0.2
)
BLUES_HEX_CODES = [qvaluetocolor.rgb_vals_to_hex(rgb_ints) for rgb_ints
        in BLUES]

# The Graphviz program used for layout (positioning).
LAYOUT_PROGRAM = 'fdp'

get_sign = lambda x: '-' if x.startswith('-') else '+'

def make_cli_parser():
    """Creates the command line interface."""

    usage = """\
python %prog [OPTIONS] EDGES_FILE1 [EDGES_FILE2 ...]

ARGUMENTS:
    EDGES_FILE: a tab-delimited file of pairs of gene sets followed by
        the significance of their overlap as a p-value.
"""
    cli_parser = optparse.OptionParser(usage)
    cli_parser.add_option('--probability', action='store_true',
            help=("Indicates link significances are probabilities "
                "rather than p-values. [NOTE: Please adjust "
                "the significance threshold using --significance "
                "when using this option.]")
    )
    cli_parser.add_option('-s', '--significance', type='float',
            default=0.05,
            help="significance threshold [DEFAULT: %default]"
    )
    cli_parser.add_option('--annotation1-col', default='annotation1',
            help=("Title of column containing name of the first "
                "annotation [DEFAULT: %default]"
            )
    )
    cli_parser.add_option('--annotation2-col', default='annotation2',
            help=("Title of column containing name of the second "
                "annotation [DEFAULT: %default]"
            )
    )
    cli_parser.add_option('--significance-col', default='p_value',
            help=("Title of column containing pair significance"
                " [DEFAULT: %default]"
            )
    )
    cli_parser.add_option('--annotation1-size-col',
            default='annotation1_size',
            help=("Title of column containing size of the set of "
                "genes annotated by the first annotation [DEFAULT: "
                "%default]"
            )
    )
    cli_parser.add_option('--annotation1-neighbors-size-col',
            default='annotation1_neighbors_size',
            help=("Title of column containing size of the set of "
                "genes neighboring those annotated by the first "
                "annotation [DEFAULT: %default]"
            )
    )
    cli_parser.add_option('--annotation2-size-col',
            default='annotation2_size',
            help=("Title of column containing size of the set of "
                "genes annotated by the second annotation [DEFAULT: "
                "%default]"
            )
    )
    cli_parser.add_option('-g', '--gene-set-significances',
            help="A CSV file containing gene set significances. "
            "The file should contain a header row, which will be "
            "ignored. The first column should contain the name of "
            "each gene set; each column following should correspond "
            "with that gene set's significance in each file, in "
            "order with respect to the input order on the command line."
    )
    cli_parser.add_option('--selected-annotations',
            help=("A file containing annotation terms to test "
                "linkage to each other. The file should contain one "
                "term per line. Selecting this option restricts the "
                "output to interactions between these terms."
            )
    )
    cli_parser.add_option('-p', '--png', action='store_true',
            help="Render a PNG image for each graph"
    )
    cli_parser.add_option('--pdf', action='store_true',
            help="Render a PDF image for each graph"
    )
    cli_parser.add_option('--svg', action='store_true',
            help="Render an SVG image for each graph"
    )
    cli_parser.add_option('-d', '--no-dot', action='store_true',
            help="Do not write DOT files"
    )
    cli_parser.add_option('--print-significances', action='store_true',
            help="Print the significance of each edge as an edge label"
    )
    cli_parser.add_option(
            '--print-annotation-stats',
            action='store_true',
            help="Print the statistics about each gene set"
    )
    return cli_parser


def results_to_edges_and_stats(
        csv_fileh,
        significance_cutoff,
        annotation1_column_title,
        annotation2_column_title,
        significance_column_title,
        annotation1_size_column_title,
        annotation1_neighbors_size_column_title,
        annotation2_size_column_title,
        selected_annotations,
        less_than=True
    ):
    """Reads results from a CSV file and converts them into edges.

    Returns a dictionary with annotation pairs as the keys and their
    p-values as values.

    :Parameters:
    - `csv_fileh`: a CSV file with column headers
    - `significance_cutoff`: a value to use as the threshold for
      including an edge
    - `annotation1_column_title`: title of the column containing the
      name of the first annotation
    - `annotation2_column_title`: title of the column containing the
      name of the second annotation
    - `significance_column_title`: title of the column containing
      significance values (e.g., `'pvalue'`)
    - `annotation1_size_column_title`: title of the column containing
      the size of the set of genes annotated by the first annotation
    - `annotation1_neighbors_size_column_title`: title of the column
      containing the size of the set of genes neighboring those
      annotated by the first annotation
    - `annotation2_size_column_title`: title of the column containing
      the size of the set of genes annotated by the second annotation
    - `selected_annotations`: a `set` of annotations to which
      interactions should be restricted
    - `less_than`: whether the significance should be less than or
      equal to the `significance_cutoff` to be considered significant,
      or be greater than or equal to the `significance_cutoff`
      [default: `True`]

    """
    csv_reader = convutils.make_csv_reader(csv_fileh)
    edges = {}
    annotation_stats = {}
    for entry in csv_reader:
        significance = float(entry[significance_column_title])
        if less_than:
            significant = significance <= significance_cutoff
        else:
            significant = significance >= significance_cutoff
        if significant:
            annotation1 = entry[annotation1_column_title]
            annotation2 = entry[annotation2_column_title]
            if selected_annotations is not None:
                # If we are restricting the annotations to include, skip
                # the rest of this if either of the annotation terms
                # aren't in the set of selected terms.
                if (not annotation1 in selected_annotations) or (not
                        annotation2 in selected_annotations):
                    continue
            pair_key = (annotation1, annotation2)
            # TODO: Commented out for mcmcbpn hack, fix
            #annotation1_size = entry[annotation1_size_column_title]
            #annotation1_neighbors_size = \
                    #entry[annotation1_neighbors_size_column_title]
            #annotation2_size = entry[annotation2_size_column_title]
            ## We should probably check if these stats have already been
            ## put in the annotations_stats dictionary, but I'm lazy and
            ## they *should* be consistent.
            #if annotation1 not in annotation_stats:
                #annotation_stats[annotation1] = {
                        #'size': int(annotation1_size),
                        #'neighbors': int(annotation1_neighbors_size)
                #}
            #else:
                #annotation_stats[annotation1]['size'] = \
                        #int(annotation1_size)
                #annotation_stats[annotation1]['neighbors'] = \
                        #int(annotation1_neighbors_size)
            #if annotation2 not in annotation_stats:
                #annotation_stats[annotation2] = {
                    #'size': annotation2_size
                #}
            #else:
                #annotation_stats[annotation2]['size'] = \
                        #int(annotation2_size)
            ## Keep only the edge statistics in the dictionary
            #del entry[annotation1_column_title]
            #del entry[annotation2_column_title]
            #del entry[annotation1_size_column_title]
            #del entry[annotation1_neighbors_size_column_title]
            #del entry[annotation2_size_column_title]
            edges[pair_key] = entry
    return edges, annotation_stats


def get_all_edges_from_files(
        edge_file_names,
        significance_cutoff,
        annotation1_column_title,
        annotation2_column_title,
        significance_column_title,
        annotation1_size_column_title,
        annotation1_neighbors_size_column_title,
        annotation2_size_column_title,
        selected_annotations=None,
        less_than=True
    ):
    """Reads the edge files and returns a dictionary with pairs of
    annotation terms as keys and their statistics as dictionaries as
    values.

    :Parameters:
    - `edge_file_names`: names of edge files
    - `significance_cutoff`: a value to use as the threshold for
      including an edge
    - `annotation1_column_title`: title of the column containing the
      name of the first annotation
    - `annotation2_column_title`: title of the column containing the
      name of the second annotation
    - `significance_column_title`: title of the column containing
      significance values (e.g., `'pvalue'`)
    - `annotation1_size_column_title`: title of the column containing
      the size of the set of genes annotated by the first annotation
    - `annotation1_neighbors_size_column_title`: title of the column
      containing the size of the set of genes neighboring those
      annotated by the first annotation
    - `annotation2_size_column_title`: title of the column containing
      the size of the set of genes annotated by the second annotation
    - `selected_annotations`: a `set` of annotations to which
      interactions should be restricted
    - `less_than`: whether the significance should be less than or
      equal to the `significance_cutoff` to be considered significant,
      or be greater than or equal to the `significance_cutoff`
      [default: `True`]

    """
    all_edges = {}
    all_annotation_stats = {}
    for edge_file_name in edge_file_names:
        edge_file = open(edge_file_name, 'rb')
        logger.info("Reading edge info from %s." % edge_file_name)
        edges, annotation_stats = results_to_edges_and_stats(
                edge_file,
                significance_cutoff,
                annotation1_column_title,
                annotation2_column_title,
                significance_column_title,
                annotation1_size_column_title,
                annotation1_neighbors_size_column_title,
                annotation2_size_column_title,
                selected_annotations,
                less_than=less_than
        )
        all_edges[edge_file_name] = edges
        all_annotation_stats.update(annotation_stats)
        edge_file.close()

    return all_edges, all_annotation_stats


def build_universe_graph_from_edges(
        all_edges,
        all_annotation_stats=None,
        print_annotation_stats=False
    ):
    """Constructs a Graphviz graph of the universe of all edges.

    :Parameters:
    - `all_edges`: a dictionary with arbitrary strings as keys and edge
      dictionaries as values. Each edge dictionary should have a tuple
      of annotation terms as a key and significance as a value.
    - `all_annotation_stats`: a dictionary containing the sizes of
      annotations and their neighborhoods
    - `print_annotation_stats`: if `True`, statistics about each
      annotation term will be printed in its node label [default:
      `False`]

    """
    #universe_graph = pygraphviz.AGraph(directed=True)
    universe_graph = pygraphviz.AGraph()
    #universe_graph.graph_attr['K'] = '5'
    universe_graph.graph_attr['overlap'] = 'scale'
    universe_graph.graph_attr['splines'] = 'true'
    universe_graph.edge_attr['style'] = 'invis'
    #universe_graph.edge_attr['len'] = '4'
    #universe_graph.edge_attr['weight'] = '20'
    universe_graph.node_attr['shape'] = 'box'
    #universe_graph.node_attr['fillcolor'] = 'white'
    for edges in all_edges.values():
        universe_graph.add_edges_from(edges.keys())
    if all_annotation_stats is not None:
        for annotation, stats in all_annotation_stats.iteritems():
            node = universe_graph.get_node(annotation)
            node.attr['label'] = '%s' % annotation
            if print_annotation_stats:
                node.attr['label'] += '\\nsize: %d\\nneighb.: %d' % (
                    stats['size'], stats['neighbors'])
    return universe_graph


def significance_to_blue(significance, probability=False):
    """Converts a significance to a color in the ColorBrewer 2 Blues
    spectrum.

    See http://colorbrewer2.org/ for more information on the color
    scheme.

    :Parameters:
    - `significance`: a floating point value representing the
      significance of a connection
    - `probability`: indicates link significances are probabilities
      [default: `False`]

    """
    if significance > 1 or significance < 0:
        raise ValueError("significance should be between 0 and 1, "
                "inclusive")
    if probability:
        significance = 1 - significance
        index = bisect.bisect_left(PERCENT_CUTOFFS, significance)
    else:
        index = bisect.bisect_left(CUTOFFS, significance)
    return BLUES_HEX_CODES[index]


def parse_set_significances(significances_fileh):
    """Parses the set significances from a CSV file.

    Returns a dictionary with set names as keys and their significances
    as a list of floats.

    :Parameters:
    - `significances_fileh`: A CSV file containing gene set
      significances. The file should contain a header row, which will
      be ignored. The first column should contain the name of each gene
      set; each column following should correspond with that gene set's
      significance in each file, in order with respect to the input
      order on the command line.

    """
    csv_reader = convutils.make_csv_reader(significances_fileh,
            headers=False)
    csv_reader.next()
    set_significances = {}
    for entry in csv_reader:
        significances = [(get_sign(value), abs(float(value))) for value
                in entry[1:]]
        set_significances[entry[0]] = significances
    return set_significances


def make_specific_graph(
        universe_graph,
        desired_edges,
        significance_column_title,
        file_index,
        set_significances=None,
        print_significances=False,
        probability=False
    ):
    """Makes a specific graph in which the desired edges and their
    incident nodes are visible.

    :Parameters:
    - `universe_graph`: the graph that contains the universe of
      interactions
    - `desired_edges`: an edge dictionary with pairs of annotations as
      keys and their significances as values
    - `significance_column_title`: the title of the column containing
      the significance value of the pair (e.g., 'p_value')
    - `file_index`: the 0-indexed index of the graph, corresponding to the
      original file on the command line
    - `set_significances`: if provided, should be a dictionary with set
      names as keys and significances as lists of floats
    - `print_significances`: if `True`, edges will be labeled with their
      significances [default: `False`]
    - `probability`: indicates link significances are probabilities
      [default: `False`]

    """
    graph = universe_graph.copy()
    if set_significances:
        for g_node in graph.nodes():
            g_node.attr['style'] = 'filled'
            significance_sign, significance_value = set_significances[g_node][file_index]
            rdylgn_color = qvaluetocolor.value_to_rdylgn(
                    significance_value, significance_sign)
            g_node.attr['fillcolor'] = rdylgn_color
            if rdylgn_color in (
                    qvaluetocolor.DOWNREGULATED_HEX_CODES[:2] +
                    qvaluetocolor.UPREGULATED_HEX_CODES[:2]
                ):
                g_node.attr['fontcolor'] = 'white'
    for edge, values in desired_edges.items():
        g_edge = graph.get_edge(*edge)
        g_edge.attr['style'] = 'solid'
        g_edge.attr['penwidth'] = '5.0'
        # TODO: Commented out for mcmcbpn hack, fix
        #g_edge.attr['arrowsize'] = '2.0'
        #g_edge.attr['arrowtype'] = 'normal'
        if print_significances:
            significance = float(values[significance_column_title])
            g_edge.attr['label'] = '%.3f' % significance
            g_edge.attr['color'] = significance_to_blue(significance,
                    probability)
        else:
            if significance_column_title in values:
                significance = float(values[significance_column_title])
                g_edge.attr['color'] = significance_to_blue(
                        significance, probability)
            else:
                g_edge.attr['color'] = significance_to_blue(0)
    return graph


def main(argv):
    cli_parser = make_cli_parser()
    opts, args = cli_parser.parse_args(argv)
    if len(args) < 1:
        cli_parser.error("Please provide at least one edges file.")

    if opts.selected_annotations:
        selected_annotations_file = open(opts.selected_annotations)
        selected_annotations = set(
                parsers.parse_selected_terms_file(
                    selected_annotations_file)
        )
    else:
        selected_annotations = None

    all_edges, all_annotation_stats = get_all_edges_from_files(
            args,
            opts.significance,
            opts.annotation1_col,
            opts.annotation2_col,
            opts.significance_col,
            opts.annotation1_size_col,
            opts.annotation1_neighbors_size_col,
            opts.annotation2_size_col,
            selected_annotations,
            less_than=(not opts.probability)
    )
    universe_graph = build_universe_graph_from_edges(all_edges,
            all_annotation_stats)

    if opts.gene_set_significances:
        set_significances_file = open(opts.gene_set_significances, 'rb')
        set_significances = parse_set_significances(
                set_significances_file)
    else:
        set_significances = None

    for i, infile_name in enumerate(args):
        desired_edges = all_edges[infile_name]
        graph = make_specific_graph(
                universe_graph,
                desired_edges,
                opts.significance_col,
                i,
                set_significances,
                opts.print_significances,
                probability=(opts.probability)
        )
        outfile_base, ext = os.path.splitext(
                os.path.basename(infile_name))
        dot_outfile_name = outfile_base + '.dot'
        if not opts.no_dot:
            logger.info("Writing DOT file to %s." % dot_outfile_name)
            graph.write(dot_outfile_name)
        if opts.png:
            png_outfile_name = outfile_base + '.png'
            logger.info("Rendering PNG image to %s." % png_outfile_name)
            graph.draw(png_outfile_name, prog=LAYOUT_PROGRAM)
        if opts.pdf:
            pdf_outfile_name = outfile_base + '.pdf'
            logger.info("Rendering PDF image to %s." % pdf_outfile_name)
            graph.draw(pdf_outfile_name, prog=LAYOUT_PROGRAM)
        if opts.svg:
            svg_outfile_name = outfile_base + '.svg'
            logger.info("Rendering SVG image to %s." % svg_outfile_name)
            graph.draw(svg_outfile_name, prog=LAYOUT_PROGRAM)


if __name__ == '__main__':
    main(sys.argv[1:])
