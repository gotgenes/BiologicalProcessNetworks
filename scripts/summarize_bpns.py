#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright (c) 2011 Christopher D. Lasher
#
# This software is released under the MIT License. Please see
# LICENSE.txt for details.


"""Calculates the overlap between significant links."""


import collections
import datetime
import itertools
import logging
import math
import optparse
import os.path
import sys

import bpn.cli
import bpn.structures
import bpn.mcmc.defaults
from bpn import statstools
from convutils import convutils, convstructs
import networkx
import numpy
import matplotlib
# Uncomment if running on computer without X windows
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

logger = logging.getLogger()
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
#stream_handler.setLevel(logging.INFO)
logger.addHandler(stream_handler)
formatter = logging.Formatter('%(message)s')
stream_handler.setFormatter(formatter)


# Output fields for interactions' link tallies
INTERACTION_TALLIES_FIELDNAMES = [
        'links_per_interaction',
        'all',
        'all_percent',
        'all_percent_explained'
]

# Additional fields if expression data is provided
INTERACTION_TALLIES_EXPR_FIELDNAMES = (
        INTERACTION_TALLIES_FIELDNAMES + [
            'active',
            'active_percent',
            'active_percent_explained',
            'inactive',
            'inactive_percent',
            'inactive_percent_explained'
        ]
)

# Output fields for summaries of interactions' link tallies
INTERACTION_TALLIES_SUMMARY_FIELDNAMES = [
        'links_per_interaction',
        'all_percent_mean',
        'all_percent_stddev',
        'all_percent_explained_mean',
        'all_percent_explained_stddev',
]
# Additional fields if expression data is provided
INTERACTION_TALLIES_SUMMARY_EXPR_FIELDNAMES = (
        INTERACTION_TALLIES_SUMMARY_FIELDNAMES + [
            'active_percent_mean',
            'active_percent_stddev',
            'active_percent_explained_mean',
            'active_percent_explained_stddev',
            'inactive_percent_mean',
            'inactive_percent_stddev',
            'inactive_percent_explained_mean',
            'inactive_percent_explained_stddev',
        ]
)

# Output fields for term gene overlap statistics within a single file
TERM_GENES_FIELDNAMES = [
        'term1',
        'term2',
        'term1_genes',
        'term2_genes',
        'intersection',
        'union',
        'jaccard',
        'fishers_exact'
]
# Additional fields if expression data is provided
TERM_GENES_EXPR_FIELDNAMES = TERM_GENES_FIELDNAMES + [
        'term1_active_genes',
        'term2_active_genes',
        'active_intersection',
        'active_union',
        'active_jaccard',
        'active_fishers_exact',
        'term1_inactive_genes',
        'term2_inactive_genes',
        'inactive_intersection',
        'inactive_union',
        'inactive_jaccard',
        'inactive_fishers_exact',
]

# Output fields for term interactions overlap statistics within a single
# file
TERM_INTERACTIONS_FIELDNAMES = [
        'term1',
        'term2',
        'term1_intraterm_interactions',
        'term2_intraterm_interactions',
        'intersection',
        'union',
        'jaccard',
        'fishers_exact'
]
# Additional fields if expression data is provided
TERM_INTERACTIONS_EXPR_FIELDNAMES = TERM_INTERACTIONS_FIELDNAMES + [
        'term1_active_intraterm_interactions',
        'term2_active_intraterm_interactions',
        'active_intersection',
        'active_union',
        'active_jaccard',
        'active_fishers_exact',
        'term1_inactive_intraterm_interactions',
        'term2_inactive_intraterm_interactions',
        'inactive_intersection',
        'inactive_union',
        'inactive_jaccard',
        'inactive_fishers_exact',
]

# Output fields for link overlap statistics within a single file
INTRALINKS_FIELDNAMES = [
        'link1',
        'link2',
        'link1_interactions',
        'link2_interactions',
        'intersection',
        'union',
        'jaccard'
]
# Additional fields if expression data is provided
INTRALINKS_EXPR_FIELDNAMES = INTRALINKS_FIELDNAMES + [
        'link1_active_interactions',
        'link2_active_interactions',
        'active_intersection',
        'active_union',
        'active_jaccard',
        'link1_inactive_interactions',
        'link2_inactive_interactions',
        'inactive_intersection',
        'inactive_union',
        'inactive_jaccard'
]

# Output fields for comparison of overlap in interactions of two graphs
INTERFILE_INTERACTIONS_OVERLAP_FIELDNAMES = [
        'file1',
        'file2',
        'interactions1',
        'interactions2',
        'intersection',
        'union',
        'jaccard',
        'intersection_by_set1',
        'intersection_by_set2',
]
# Additional fields if expression data is provided
INTERFILE_INTERACTIONS_OVERLAP_EXPR_FIELDNAMES = (
        INTERFILE_INTERACTIONS_OVERLAP_FIELDNAMES + [
            'active_interactions1',
            'active_interactions2',
            'active_intersection',
            'active_union',
            'active_jaccard',
            'active_intersection_by_set1',
            'active_intersection_by_set2',
            'inactive_interactions1',
            'inactive_interactions2',
            'inactive_intersection',
            'inactive_union',
            'inactive_jaccard',
            'inactive_intersection_by_set1',
            'inactive_intersection_by_set2',
        ]
)

# Output fields for mean summaries
MEANS_AND_STDDEVS_FIELDNAMES = [
        'bin',
        'mean',
        'stddev'
]
# Additional fields if expression data is provided
MEANS_AND_STDDEVS_EXPR_FIELDNAMES = MEANS_AND_STDDEVS_FIELDNAMES + [
        'active_mean',
        'active_stddev',
        'inactive_mean',
        'inactive_stddev'
]

# Output fields for summary statistics of BPNs
BPN_STATISTICS_FIELDNAMES = [
        'file',
        'terms',
        'links',
        'interactions'
]
# Additional fields if expression data is provided
BPN_STATISTICS_EXPR_FIELDNAMES = BPN_STATISTICS_FIELDNAMES + [
        'active_interactions',
        'inactive_interactions'
]

BPN_STATISTICS_SUMMARY_FIELDNAMES = list(
        itertools.chain.from_iterable(
            ('{0}_mean'.format(x), '{0}_stddev'.format(x)) for x
            in BPN_STATISTICS_FIELDNAMES[1:]
        )
)
BPN_STATISTICS_SUMMARY_EXPR_FIELDNAMES = list(
        itertools.chain.from_iterable(
            ('{0}_mean'.format(x), '{0}_stddev'.format(x)) for x
            in BPN_STATISTICS_EXPR_FIELDNAMES[1:]
        )
)

NAME_OVERLAPS_FIELDNAMES = [
        'file1',
        'file2',
        'links1',
        'links2',
        'links_intersection',
        'links_union',
        'links_jaccard',
        'links_intersection_by_1',
        'links_intersection_by_2',
        'terms1',
        'terms2',
        'terms_intersection',
        'terms_union',
        'terms_jaccard',
        'terms_intersection_by_1',
        'terms_intersection_by_2',
]

NAME_OVERLAPS_KEY_MAP = {
        'links': {
                'set1_size': 'links1',
                'set2_size': 'links2',
                'intersection': 'links_intersection',
                'union': 'links_union',
                'jaccard': 'links_jaccard',
                'intersection_by_set1': 'links_intersection_by_1',
                'intersection_by_set2': 'links_intersection_by_2',
        },
        'terms': {
                'set1_size': 'terms1',
                'set2_size': 'terms2',
                'intersection': 'terms_intersection',
                'union': 'terms_union',
                'jaccard': 'terms_jaccard',
                'intersection_by_set1': 'terms_intersection_by_1',
                'intersection_by_set2': 'terms_intersection_by_2',
        },
}

NAME_OVERLAPS_SUMMARY_FIELDNAMES = [
        'links_jaccard_mean',
        'links_jaccard_stddev',
        'terms_jaccard_mean',
        'terms_jaccard_stddev'
]

OVERLAP_CLASS_TO_TITLE = {
        'genes': 'Genes-based overlap\nof processes',
        'links': 'Overlap between links based on\ncross-annotated interactions',
        'interaction_link_tallies': 'Explanatory links per interaction',
        'interactions': ('Overlap between\nprocesses based on\nintraterm interactions ')
}

OVERLAP_CLASS_TO_YAXIS = {
        'genes': 'Percent of process pairs',
        'links': 'Percent of link pairs',
        'interaction_link_tallies': 'Links per interaction',
        'interactions': 'Percent of link pairs'
}

OVERLAP_CLASS_TO_YAXIS_MAX = {
        'genes': 'Percent of processes',
        'links': 'Percent of links',
        'interaction_link_tallies': 'Links per interaction',
        'interactions': 'Percent of link pairs'
}

PERCENTAGE_FORMATTER = FuncFormatter(lambda x, y: '{0:1.0f}%'.format(
        100 * x))

ALL_BAR_COLOR = '#A98FBE'
ACTIVE_BAR_COLOR = '#FABFBB'
INACTIVE_BAR_COLOR = '#8AA0DA'


class LinksStatsInputData(object):
    def __init__(
            self,
            interactions_graph,
            annotations_dict,
            annotations_stats,
            links_files,
            terms_files,
            output_dir,
            extensions,
            activity_threshold,
            interactions_greater,
            links_cutoff,
            links_greater,
            intraterms,
            num_bins
        ):
        """Create a new instance.

        :Parameters:
        - `interactions_graph`: graph containing the gene-gene or gene
          product-gene product interactions
        - `annotations_dict`: a dictionary with annotation terms as keys
          and `set`s of genes as values
        - `annotations_stats`: a dictionary containing statistics about
          the annotations
        - `links_file`: files containing links and their significances
        - `output_dir`: directory for output files
        - `activity_threshold`: expression threshold for considering an
          interaction active (both genes must meet this threshold)
        - `interactions_greater`: if `True`, interaction considered
          active if genes' activities are greater than or equal to
          activity threshold, else if `False`, if genes' activities are
          less than or equal to the threshold
        - `links_cutoff`: cutoff at which a link is significant
        - `links_greater`: whether a link should be considered
          significant if it is greater (`True`) or less (`False`) than
          the cutoff
        - `num_bins`: number of bins for histograms

        """
        # TODO: Update docstring
        self.interactions_graph = interactions_graph
        self.annotations_dict = annotations_dict
        self.annotations_stats = annotations_stats
        self.links_files = links_files
        self.terms_files = terms_files
        self.output_dir = output_dir
        self.extensions = extensions
        self.activity_threshold = activity_threshold
        self.interactions_greater = interactions_greater
        self.links_cutoff = links_cutoff
        self.links_greater = links_greater
        self.intraterms = intraterms
        self.num_bins = num_bins


class LinksStatsArgParser(bpn.cli.ExpressionBasedArgParser):
    """Command line parser for statistics."""

    _prog_name = 'linkstats'


    def __init__(self):
        super(LinksStatsArgParser, self).__init__()
        self.timestamp = datetime.datetime.now().strftime(
                '%Y-%m-%d')
        # Re-run this so we get tho output directory correct; sort of
        # hackish, but it doesn't much time.
        self.make_cli_parser()

    def make_cli_parser(self):
        super(LinksStatsArgParser, self).make_cli_parser()
        usage = """\
python %prog [OPTIONS] INTERACTIONS_FILE ANNOTATIONS_FILE LINKS_FILE1 [LINKS_FILE2 ... LINKS_FILEN]
    INTERACTIONS_FILE: a CSV file containing interactions. The file
        should have two columns with headings "interactor1" and
        "interactor2". It may have an optional column with the heading
        "weight", whose values will be used as the weight or confidence
        of the interaction. The file may have additional columns, which
        will be ignored.
    ANNOTATIONS_FILE: a file containing annotations. The annotations
        file may be in one of two formats:
        - GMT format: if the file ends with the extension ".gmt", it is
          automatically parsed as a GMT-format file. The file is a
          tab-separated (TSV) format with no headers. The first column
          contains the annotation term. The second column contains a
          description. All following columns contain gene IDs for genes
          annotated by that term. Full GMT format specification is
          available from the MSigDB and GSEA website.
        - Two-column format: The file should have a column titled
          "gene_id" which has the gene/gene product ID, and a column
          titled "term" which contains the term with which the
          gene/product is annotated. The file may have additional
          columns, which will be ignored.
    LINKS_FILE: a tab-delimited file of pairs of gene sets
        followed by the significance of the link between them. The first
        row should be a header with 'process1', 'process2', and
        'probability'. The user may provide multiple links files.\
"""
        self.cli_parser.set_usage(usage)
        self.cli_parser.remove_option('--links-outfile')
        logfile_option = self.cli_parser.get_option('--logfile')
        logfile_option.help = ("name of logfile.")
        logfile_option.default = None
        self.cli_parser.add_option('-o', '--output-dir',
                default='bpn_summaries-{0}'.format(self.timestamp),
                help=("name of output directory [DEFAULT: "
                    "link_overlaps-TIMESTAMP"
                )
        )
        self.cli_parser.add_option('--pdf', action='append_const',
                dest='extensions', const='pdf',
                help="output plots in PDF format")
        self.cli_parser.add_option('--png', action='append_const',
                dest='extensions', const='png',
                help=("output plots in PNG format [default if none"
                    "specified]")
        )
        self.cli_parser.add_option('--svg', action='append_const',
                dest='extensions', const='svg',
                help="output plots in SVG format")
        self.cli_parser.add_option('-l', '--links-cutoff', type='float',
                default=0.75,
                help=("cutoff for which link is significant [DEFAULT "
                    "%default]")
        )
        self.cli_parser.add_option('--links-lt', dest='links_greater',
                action='store_false', default=True,
                help=("link is significant if less than or equal "
                    "to threshold [DEFAULT: greater than or equal to]"
                )
        )
        self.cli_parser.add_option('-e', '--expression-file',
                help=("a CSV file of gene (or gene product) "
                    "expression values. The file should have a column "
                    "titled \"id\" which has the gene (or gene "
                    "product) ID, and a column titled \"expression\" "
                    "which gives a value for the expression level, or "
                    "difference in expression levels."
                )
        )
        self.cli_parser.add_option('--terms-files',
                help=("a file listing terms results files, one file "
                    "per line, each file corresponding to the "
                    "respective links file as listed in order in "
                    "the arguments; the word 'None' can be used to "
                    "indicate no terms results file should be "
                    "associated with the respective links file."
                )
        )
        self.cli_parser.add_option('-a', '--activity-threshold',
                type='float', default=-math.log10(0.05),
                help=("expression threshold for considering an "
                    "interaction active (both genes must meet this "
                    "threshold) [DEFAULT: %default=-log10(0.05)] "
                    "[NOTE: this option is ignored if "
                    "--expresions-file is not provided.]"
                )
        )
        self.cli_parser.add_option('--interactions-lt',
                dest='interactions_greater', action='store_false',
                default=True,
                help=("interaction is significant if less than or "
                    "equal to threshold [DEFAULT: greater than or "
                    "equal to] [NOTE: this option is ignored if "
                    "--expresions-file is not provided.]"
                )
        )
        self.cli_parser.add_option('--intraterms', action='store_true',
                help=("Consider intra-term interactions."))
        self.cli_parser.add_option('--bins', type='int', default=10,
                help=("number of bins for histograms [DEFAULT: "
                        "%default]")
        )


    def check_num_arguments(self):
        """Verifies that the number of arguments given is correct."""
        if len(self.args) < 3:
            self.cli_parser.error(
                    "Please provide paths to an interactions file, "
                    "an annotations file, and at least one links "
                    "results file."
            )


    def _post_process_opts_and_args(self):
        if self.opts.expression_file is None:
            self.opts.activity_threshold = None
            self.opts.interactions_greater = None
        # Remove trailing slashes or backslashes from the output
        # directory.
        self.opts.output_dir = self.opts.output_dir.rstrip('/\\')
        if not self.opts.extensions:
            self.opts.extensions = ['png']


class LinksStatsCli(bpn.cli.ContextualCli):
    """Command line interface for linksstats."""
    def __init__(self):
        self.cli_parser = LinksStatsArgParser()


    def _begin_logging(self):
        if self.opts.logfile:
            file_handler = logging.FileHandler(self.opts.logfile)
            formatter = logging.Formatter(
                    '%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            file_handler.setLevel(logger.level)
            logger.addHandler(file_handler)


    def _construct_links_of_interest(self):
        # Overridden to do nothing since it's not applicable.
        pass


    def _process_input_files(self):
        # This is dirty, because we're mixing super() with explicit
        # superclass method calls, and we're also fooling with self.args
        # but I couldn't think of a better way to make it work.
        #
        # If the user gave an expression file, process it along with the
        # interactions and annotations.
        if self.opts.expression_file:
            self.args.insert(2, self.opts.expression_file)
            bpn.cli.ContextualCli._process_input_files(self)
            self.links_files = [open(arg, 'rb') for arg in
                    self.args[3:]]
        # Otherwise, just process the interactions and annotations file.
        else:
            bpn.cli.BplnCli._process_input_files(self)
            self.links_files = [open(arg, 'rb') for arg in
                    self.args[2:]]
        if self.opts.terms_files:
            self.terms_files = []
            terms_list_file = open(self.opts.terms_files)
            for terms_file_name in terms_list_file:
                terms_file_name = terms_file_name.strip()
                if terms_file_name == 'None':
                    self.terms_files.append(None)
                else:
                    logging.info(("Including significant terms from "
                        "{0}").format(terms_file_name))
                    terms_file = open(terms_file_name)
                    self.terms_files.append(terms_file)
            terms_list_file.close()
        else:
            self.terms_files = itertools.repeat(None)


    def _open_output_files(self):
        output_dir = self.opts.output_dir
        # See if the output directory already exists.
        if os.path.exists(output_dir):
            if not os.path.isdir(output_dir):
                msg = "{0} exists but is not a directory".format(
                        output_dir)
                logger.critical(msg)
                raise ValueError(msg)
        else:
            # There's no output directory; make it.
            os.mkdir(output_dir)
        self.output_dir = output_dir


    def _construct_data_struct(self):
        data = LinksStatsInputData(
                interactions_graph=self.interactions_graph,
                annotations_dict=self.annotations_dict,
                annotations_stats=self.annotations_stats,
                links_files=self.links_files,
                terms_files=self.terms_files,
                output_dir=self.output_dir,
                extensions=self.opts.extensions,
                activity_threshold=self.opts.activity_threshold,
                interactions_greater=self.opts.interactions_greater,
                links_cutoff=self.opts.links_cutoff,
                links_greater=self.opts.links_greater,
                intraterms=self.opts.intraterms,
                num_bins=self.opts.bins
        )
        return data


def make_process_links_graph(
        links_csv_reader,
        cutoff,
        terms_csv_reader=None,
        greater=True,
        term1_col=bpn.mcmc.defaults.LINKS_FIELDNAMES[0],
        term2_col=bpn.mcmc.defaults.LINKS_FIELDNAMES[1],
        significance_col=bpn.mcmc.defaults.LINKS_FIELDNAMES[2],
        term_name_col=bpn.mcmc.defaults.TERMS_FIELDNAMES[0]
    ):
    """Returns the significant links as a graph.

    :Parameters:
    - `links_csv_reader`: a `csv.DictReader` instance for links results
    - `cutoff`: a floating point `cutoff` for significance
    - `links_csv_reader`: a `csv.DictReader` instance for terms results
    - `greater`: if `True`, indicates significant links will be greater
      than or equal too the cutoff, else, indicates they are less than
      or equal to the `cutoff` [default: `True`]
    - `term1_col`: title of the column containing the name of the first
      term
    - `term2_col`: title of the column containing the name of the second
      term
    - `significance_col`: title of the column containing the
      significance of the link
    - `term_name_col`: title of the column containing the name of the
      term in the terms results file

    """
    process_graph = networkx.Graph()
    for entry in links_csv_reader:
        significance = float(entry[significance_col])
        if greater:
            is_significant = significance >= cutoff
        else:
            is_significant = significance <= cutoff
        if is_significant:
            process_graph.add_edge(
                entry[term1_col], entry[term2_col], weight=significance)
    if terms_csv_reader is not None:
        for entry in terms_csv_reader:
            significance = float(entry[significance_col])
            if greater:
                is_significant = significance >= cutoff
            else:
                is_significant = significance <= cutoff
            if is_significant:
                process_graph.add_node(
                    entry[term_name_col], weight=significance)
    return process_graph


def get_link_interactions(
        link,
        annotated_interactions,
        intraterms=False
    ):
    """Returns the interactions for a link.

    :parameters:
    - `link`: a pair of linked annotation terms
    - `annotated_interactions`: a
      `bpn.structures.annotatedinteractionsgraph` instance
    - `intraterms`: include intraterm links if `True`

    """
    link_interactions = (
            annotated_interactions.get_coannotated_interactions(
                    link[0], link[1])
    )
    if intraterms:
        link_interactions.update(
                annotated_interactions.get_intraterm_interactions(
                    link[0])
        )
        link_interactions.update(
                annotated_interactions.get_intraterm_interactions(
                    link[1])
        )
    return link_interactions


def calc_term_gene_overlaps(
        annotated_interactions,
        term1,
        term2,
        active_genes=None
    ):
    """Calculates the overlap of two terms based on the genes they
    annotate.

    """
    term1_genes = annotated_interactions.get_annotated_genes(term1)
    term2_genes = annotated_interactions.get_annotated_genes(term2)
    num_total_genes = annotated_interactions.calc_num_genes()
    term_overlaps = statstools.calculate_overlap_scores(term1_genes,
            term2_genes, num_total_genes)
    term_overlaps = {'all': term_overlaps}

    if active_genes:
        term1_active_genes = active_genes.intersection(term1_genes)
        term2_active_genes = active_genes.intersection(term2_genes)
        num_active_genes = len(active_genes)
        active_overlaps = statstools.calculate_overlap_scores(
                term1_active_genes,
                term2_active_genes,
                num_active_genes
        )
        term_overlaps['active'] = active_overlaps

        term1_inactive_genes = term1_genes.difference(
                term1_active_genes)
        term2_inactive_genes = term2_genes.difference(
                term2_active_genes)
        num_inactive_genes = num_total_genes - num_active_genes
        inactive_overlaps = statstools.calculate_overlap_scores(
                term1_inactive_genes,
                term2_inactive_genes,
                num_inactive_genes
        )
        term_overlaps['inactive'] = inactive_overlaps

    return term_overlaps


def calc_term_interactions_overlaps(
        annotated_interactions,
        term1,
        term2,
        active_interactions=None
    ):
    """Calculates the overlap of two terms based on their intra-term
    interactions.

    """
    term1_interactions = (
            annotated_interactions.get_intraterm_interactions(term1))
    term2_interactions = (
            annotated_interactions.get_intraterm_interactions(term2))
    num_total_interactions = (
            annotated_interactions.calc_num_interactions())
    intraterm_overlaps = statstools.calculate_overlap_scores(
            term1_interactions,
            term2_interactions,
            num_total_interactions
    )
    intraterm_overlaps = {'all': intraterm_overlaps}

    if active_interactions:
        term1_active_interactions = active_interactions.intersection(
                term1_interactions)
        term2_active_interactions = active_interactions.intersection(
                term2_interactions)
        num_active_interactions = len(active_interactions)
        active_overlaps = statstools.calculate_overlap_scores(
                term1_active_interactions,
                term2_active_interactions,
                num_active_interactions
        )
        intraterm_overlaps['active'] = active_overlaps

        term1_inactive_interactions = term1_interactions.difference(
                term1_active_interactions)
        term2_inactive_interactions = term2_interactions.difference(
                term2_active_interactions)
        num_inactive_interactions = (num_total_interactions -
                num_active_interactions)
        inactive_overlaps = statstools.calculate_overlap_scores(
                term1_inactive_interactions,
                term2_inactive_interactions,
                num_inactive_interactions
        )
        intraterm_overlaps['inactive'] = inactive_overlaps

    return intraterm_overlaps


def calc_interaction_link_tallies(
        links,
        annotated_interactions,
        active_interactions=None
    ):
    """Calculates the number of selected links in which each interaction
    participates.

    - `links`: the `set` of significant links
    - `annotated_interactions`: a
      `bpn.structures.AnnotatedInteractionsGraph` instance
    - `active_interactions`: a set of interactions noted as "active"

    """
    # This is a little ugly, but I have to directly access the
    # cross_annotations dictionary and convert it into a two-way set
    # dictionary; as it is, the AnnotatedInteractionsGraph class does
    # not have methods yet to map from interactions to cross_annotations.
    cross_annotations_two_way_dict = convstructs.TwoWaySetDict(
            annotated_interactions._coannotations_to_interactions)
    # We need three tallies, one for overall, one for active, one for
    # inactive.
    overall_tallies = collections.defaultdict(int)
    active_tallies = collections.defaultdict(int)
    inactive_tallies = collections.defaultdict(int)

    # Now get the amount for each interaction.
    for interaction, cross_annotations in (
            cross_annotations_two_way_dict.reverse_items()):
        selected_cross_annotations = [
                cross_annotation for cross_annotation in cross_annotations
                if cross_annotation in links
        ]
        num_selected_cross_annotations = len(selected_cross_annotations)
        overall_tallies[num_selected_cross_annotations] += 1
        if active_interactions:
            if interaction in active_interactions:
                active_tallies[num_selected_cross_annotations] += 1
            else:
                inactive_tallies[num_selected_cross_annotations] += 1

    links_per_interaction = {'all': overall_tallies}
    if active_interactions:
        links_per_interaction['active'] = active_tallies
        links_per_interaction['inactive'] = inactive_tallies

    # Now go back and figure out the percentages for each tally.
    num_total_interactions = {'all':
            annotated_interactions.calc_num_interactions()}
    if active_interactions:
        num_total_interactions['active'] = len(active_interactions)
        num_total_interactions['inactive'] = (
                num_total_interactions['all'] -
                num_total_interactions['active']
        )
    # Get the percents relative to the number of total explained
    # interactions and put those totals in total_explained.
    total_explained = collections.defaultdict(int)
    # Also compute the percents relative to all interactions, regardless
    # of whether they are explained, in percent_tallies.
    percent_tallies = {}
    for category, tallies in links_per_interaction.items():
        category_percents = {}
        for tally, num_interactions in tallies.items():
            if tally > 0:
                total_explained[category] += num_interactions
            category_percents[tally] = (float(num_interactions) /
                    num_total_interactions[category])
        key_name = '{0}_percent'.format(category)
        percent_tallies[key_name] = category_percents

    # Now go back and calculate the explained percents.
    total_explained = dict(total_explained)
    explained_percent_tallies = {}
    for category, tallies in links_per_interaction.items():
        category_explained_percents = {}
        for tally, num_interactions in tallies.items():
            if tally == 0:
                category_explained_percents[tally] = 0
            else:
                try:
                    category_explained_percents[tally] = (
                            float(num_interactions) /
                            total_explained[category]
                    )
                except ZeroDivisionError:
                    category_explained_percents[tally] = 0
        key_name = '{0}_percent_explained'.format(category)
        explained_percent_tallies[key_name] = category_explained_percents

    links_per_interaction.update(percent_tallies)
    links_per_interaction.update(explained_percent_tallies)

    return links_per_interaction


def write_interaction_link_tallies(csvwriter, links_per_interaction):
    outrecords = []
    tallies = links_per_interaction['all'].keys()
    tallies.sort()
    if 'active' in links_per_interaction:
        fields = INTERACTION_TALLIES_EXPR_FIELDNAMES[1:]
    else:
        fields = INTERACTION_TALLIES_FIELDNAMES[1:]
    for tally_value in tallies:
        record = dict((
            (field, links_per_interaction[field].get(tally_value, 0))
            for field in fields)
        )
        record['links_per_interaction'] = tally_value
        outrecords.append(record)
    csvwriter.writerows(outrecords)


def calc_interaction_links_tallies_means_and_stddevs(
        interaction_link_tallies_per_file):
    tallies = interaction_link_tallies_per_file[0]['all'].keys()
    tallies.sort()
    fields = ['all_percent', 'all_percent_explained']
    if 'active' in interaction_link_tallies_per_file[tallies[0]]:
        fields.extend([
                'active_percent',
                'active_percent_explained',
                'inactive_percent',
                'inactive_percent_explained'
        ])
    collected_tallies = collections.defaultdict(
            lambda: collections.defaultdict(list))
    # For every combination of tally value (e.g., ``1`` link per
    # interaction) and category (e.g., ``'active_percent'``), get the
    # value of that combination across every single file. Store this
    # keyed by the tally first, then by the category second.
    for interaction_link_tallies in interaction_link_tallies_per_file:
        for tally_value in tallies:
            for field in fields:
                value = (
                        interaction_link_tallies[field].get(
                            tally_value, 0)
                )
                collected_tallies[tally_value][field].append(value)
    tallies_means_and_stddevs = {}
    for tally_value, tally_data in collected_tallies.items():
        tally_means_and_stddevs = {}
        for category, category_data in tally_data.items():
            category_data = numpy.array(category_data)
            tally_means_and_stddevs[category] = (
                    category_data.mean(),
                    category_data.std()
            )
        tallies_means_and_stddevs[tally_value] = tally_means_and_stddevs

    return tallies_means_and_stddevs


def write_interaction_links_tallies_means_and_stddevs(csvwriter,
        tallies_means_and_stddevs):
    outrecords = []
    tallies = tallies_means_and_stddevs.keys()
    tallies.sort()
    for tally in tallies:
        record = {'links_per_interaction': tally}
        for category, (mean_value, stddev_value) in (
                tallies_means_and_stddevs[tally].items()):
            record['{0}_mean'.format(category)] = mean_value
            record['{0}_stddev'.format(category)] = stddev_value
        outrecords.append(record)
    csvwriter.writerows(outrecords)


def write_within_link_gene_overlaps(csvwriter, overlaps):
    outrecords = []
    for link in sorted(overlaps.keys()):
        gene_overlaps = overlaps[link]
        outrecord = {
                'term1': link[0],
                'term2': link[1],
                'term1_genes': gene_overlaps['all']['set1_size'],
                'term2_genes': gene_overlaps['all']['set2_size'],
                'intersection': gene_overlaps['all']['intersection'],
                'union': gene_overlaps['all']['union'],
                'jaccard': gene_overlaps['all']['jaccard'],
                'fishers_exact': gene_overlaps['all']['fishers_exact']
        }
        if 'active' in overlaps[link]:
            outrecord.update({
                'term1_active_genes': gene_overlaps['active']['set1_size'],
                'term2_active_genes': gene_overlaps['active']['set2_size'],
                'active_intersection': gene_overlaps['active']['intersection'],
                'active_union': gene_overlaps['active']['union'],
                'active_jaccard': gene_overlaps['active']['jaccard'],
                'active_fishers_exact': gene_overlaps['active']['fishers_exact'],
                'term1_inactive_genes': gene_overlaps['inactive']['set1_size'],
                'term2_inactive_genes': gene_overlaps['inactive']['set2_size'],
                'inactive_intersection': gene_overlaps['inactive']['intersection'],
                'inactive_union': gene_overlaps['inactive']['union'],
                'inactive_jaccard': gene_overlaps['inactive']['jaccard'],
                'inactive_fishers_exact': gene_overlaps['inactive']['fishers_exact']
            })
        outrecords.append(outrecord)

    csvwriter.writerows(outrecords)


def write_within_link_interactions_overlaps(csvwriter, overlaps):
    outrecords = []
    for link in sorted(overlaps.keys()):
        intraterm_interaction_overlaps = overlaps[link]
        outrecord = {
                'term1': link[0],
                'term2': link[1],
                'term1_intraterm_interactions': intraterm_interaction_overlaps['all']['set1_size'],
                'term2_intraterm_interactions': intraterm_interaction_overlaps['all']['set2_size'],
                'intersection': intraterm_interaction_overlaps['all']['intersection'],
                'union': intraterm_interaction_overlaps['all']['union'],
                'jaccard': intraterm_interaction_overlaps['all']['jaccard'],
                'fishers_exact': intraterm_interaction_overlaps['all']['fishers_exact']
        }
        if 'active' in overlaps[link]:
            outrecord.update({
                'term1_active_intraterm_interactions': intraterm_interaction_overlaps['active']['set1_size'],
                'term2_active_intraterm_interactions': intraterm_interaction_overlaps['active']['set2_size'],
                'active_intersection': intraterm_interaction_overlaps['active']['intersection'],
                'active_union': intraterm_interaction_overlaps['active']['union'],
                'active_jaccard': intraterm_interaction_overlaps['active']['jaccard'],
                'active_fishers_exact': intraterm_interaction_overlaps['active']['fishers_exact'],
                'term1_inactive_intraterm_interactions': intraterm_interaction_overlaps['inactive']['set1_size'],
                'term2_inactive_intraterm_interactions': intraterm_interaction_overlaps['inactive']['set2_size'],
                'inactive_intersection': intraterm_interaction_overlaps['inactive']['intersection'],
                'inactive_union': intraterm_interaction_overlaps['inactive']['union'],
                'inactive_jaccard': intraterm_interaction_overlaps['inactive']['jaccard'],
                'inactive_fishers_exact': intraterm_interaction_overlaps['inactive']['fishers_exact']
            })
        outrecords.append(outrecord)

    csvwriter.writerows(outrecords)


def calc_overlap_between_links(
        link1,
        link2,
        annotated_interactions,
        active_interactions=None,
        intraterms=False
    ):
    """calculate the overlap of two links.

    :parameters:
    - `link1`: a pair of linked annotation terms
    - `link2`: a pair of linked annotation terms
    - `annotated_interactions`: a
      `bpn.structures.annotatedinteractionsgraph` instance
    - `intraterms`: include intraterm links if `True`

    """
    link1_interactions = get_link_interactions(link1,
            annotated_interactions, intraterms)
    num_link1_interactions = len(link1_interactions)
    link2_interactions = get_link_interactions(link2,
            annotated_interactions, intraterms)
    num_link2_interactions = len(link2_interactions)

    interactions_intersection = link1_interactions.intersection(
            link2_interactions)
    num_interactions_intersection = len(interactions_intersection)
    interactions_union = link1_interactions.union(link2_interactions)
    num_interactions_union = len(interactions_union)
    interactions_jaccard = float(num_interactions_intersection) / \
            num_interactions_union
    all_interactions_statistics = {
            'link1': num_link1_interactions,
            'link2': num_link2_interactions,
            'intersection': num_interactions_intersection,
            'union': num_interactions_union,
            'jaccard': interactions_jaccard
    }
    links_overlap_statistics = {'all': all_interactions_statistics}

    if active_interactions:
        link1_active_interactions = link1_interactions.intersection(
                active_interactions)
        num_link1_active_interactions = len(link1_active_interactions)
        link2_active_interactions = link2_interactions.intersection(
                active_interactions)
        num_link2_active_interactions = len(link2_active_interactions)
        active_interactions_intersection = \
                link1_active_interactions.intersection(
                        link2_active_interactions)
        num_active_interactions_intersection = len(
                active_interactions_intersection)
        active_interactions_union = link1_active_interactions.union(
                link2_active_interactions)
        num_active_interactions_union = len(active_interactions_union)
        try:
            active_interactions_jaccard = (
                    float(num_active_interactions_intersection) /
                    num_active_interactions_union)
        except ZeroDivisionError:
            active_interactions_jaccard = 0
        active_interactions_statistics = {
                'link1': num_link1_active_interactions,
                'link2': num_link2_active_interactions,
                'intersection': num_active_interactions_intersection,
                'union': num_active_interactions_union,
                'jaccard': active_interactions_jaccard
        }
        links_overlap_statistics['active'] = \
                active_interactions_statistics

        inactive_interactions_statistics = {}
        for key, value in all_interactions_statistics.items():
            active_value = active_interactions_statistics[key]
            inactive_interactions_statistics[key] = value - active_value
        # the jaccard calculation is incorrect from the above; compute
        # it correctly here.
        try:
            inactive_interactions_statistics['jaccard'] = (float(
                    inactive_interactions_statistics['intersection']) /
                    inactive_interactions_statistics['union'])
        except ZeroDivisionError:
            inactive_interactions_statistics['jaccard'] = 0
        links_overlap_statistics['inactive'] = \
                inactive_interactions_statistics

    return links_overlap_statistics


def calc_within_file_overlaps(
        links,
        annotated_interactions,
        cutoff=None,
        greater=True,
        intraterms=False
    ):
    """Calculate overlap in links interactions.

    :parameters:
    - `links`: the significant links
    - `annotated_interactions`: a
      `bpn.structures.AnnotatedInteractionsGraph` instance
    - `cutoff`: if provided, the value at which an interaction's genes
      are considered active [default: ``None``]
    - `greater`: interaction is active if greater than or equal to
      cutoff if `True`, else, less than or equal to
    - `intraterms`: include intraterm links if `True`

    """
    # the link names in the edges are returned in random order, for some
    # reason; make sure in each edge, the link is described such that
    # term1 < term2.
    if cutoff is not None:
        active_genes = annotated_interactions.get_active_genes(cutoff,
                greater)
        active_interactions = (
                annotated_interactions.get_active_interactions(cutoff,
                        greater))
    else:
        active_genes = None
        active_interactions = None
    term_genes_statistics = {}
    term_interactions_statistics = {}
    # Calculate the overlap between all pairs of terms in the BPN.
    terms = list(set(itertools.chain.from_iterable(links)))
    terms.sort()
    for pair in itertools.combinations(terms, 2):
        term_genes_overlap = calc_term_gene_overlaps(
                annotated_interactions,
                pair[0],
                pair[1],
                active_genes
        )
        term_genes_statistics[pair] = term_genes_overlap
        if intraterms:
            term_interactions_overlap = calc_term_interactions_overlaps(
                    annotated_interactions,
                    pair[0],
                    pair[1],
                    active_interactions
            )
            term_interactions_statistics[pair] = term_interactions_overlap
    links_statistics = {}
    for link1, link2 in itertools.combinations(links, 2):
        links_statistics[(link1, link2)] = calc_overlap_between_links(
                link1,
                link2,
                annotated_interactions,
                active_interactions,
                intraterms
        )

    interaction_link_tallies = calc_interaction_link_tallies(
            links,
            annotated_interactions,
            active_interactions
    )

    all_statistics = {
            'genes':  term_genes_statistics,
            'links': links_statistics,
            'interaction_link_tallies': interaction_link_tallies
    }
    if intraterms:
        all_statistics['interactions'] = term_interactions_statistics
    return all_statistics


def write_file_links_statistics(csvwriter, links_statistics):
    """output the overlap statistics of a single collection of links."""
    outrecords = []
    for links, stats in links_statistics.items():
        outrecord = {
                'link1': '--'.join(links[0]),
                'link2': '--'.join(links[1]),
                'link1_interactions': stats['all']['link1'],
                'link2_interactions': stats['all']['link2'],
                'intersection': stats['all']['intersection'],
                'union': stats['all']['union'],
                'jaccard': stats['all']['jaccard']
        }
        if 'active' in stats:
            outrecord.update({
                'link1_active_interactions': stats['active']['link1'],
                'link2_active_interactions': stats['active']['link2'],
                'active_intersection': stats['active']['intersection'],
                'active_union': stats['active']['union'],
                'active_jaccard': stats['active']['jaccard'],
                'link1_inactive_interactions': stats['inactive']['link1'],
                'link2_inactive_interactions': stats['inactive']['link2'],
                'inactive_intersection': stats['inactive']['intersection'],
                'inactive_union': stats['inactive']['union'],
                'inactive_jaccard': stats['inactive']['jaccard'],
            })
        outrecords.append(outrecord)
    csvwriter.writerows(outrecords)


def summarize_jaccards(stats):
    summary = {
            'all': [],
            'active': [],
            'inactive': []
    }
    for link_stats in stats.values():
        summary['all'].append(link_stats['all']['jaccard'])
        if 'active' in link_stats:
            summary['active'].append(link_stats['active']['jaccard'])
            summary['inactive'].append(
                    link_stats['inactive']['jaccard'])
    if not summary['active']:
        del summary['active']
        del summary['inactive']
    return summary


def max_jaccards(stats):
    link_stats_dict = collections.defaultdict(lambda:
            collections.defaultdict(float))
    for links, overlap_stats in stats.items():
        all_jaccard = overlap_stats['all']['jaccard']
        if 'active' in overlap_stats:
            active_jaccard = overlap_stats['active']['jaccard']
            inactive_jaccard = overlap_stats['inactive']['jaccard']
        for link in links:
            if link_stats_dict[link]['all'] < all_jaccard:
                link_stats_dict[link]['all'] = all_jaccard
            if 'active' in overlap_stats:
                if link_stats_dict[link]['active'] < active_jaccard:
                    link_stats_dict[link]['active'] = active_jaccard
                if link_stats_dict[link]['inactive'] < inactive_jaccard:
                    link_stats_dict[link]['inactive'] = inactive_jaccard
    links_max_jaccards = collections.defaultdict(list)
    for link, jaccards in link_stats_dict.items():
        links_max_jaccards['all'].append(jaccards['all'])
        if 'active' in jaccards:
            links_max_jaccards['active'].append(jaccards['active'])
            links_max_jaccards['inactive'].append(jaccards['inactive'])
    return links_max_jaccards


def plot_histogram(
        stats,
        outfile_names,
        num_bins=10,
        title="Overlap",
        y_axis_label="Percent of Links",
    ):
    """Plots a histogram from the overlap statistics.

    Returns a tuple. The first item is a dictionary with the categories
    (e.g., ``'all'``, ``'active'``, ``'inactive'``) as the keys the
    histogram percents; the second item is the bins for each category as
    the values.

    :Parameters:
    - `stats`: a dictionary of overlap statistics
    - `outfile_name`: name of the file to write
    - `num_bins`: number of bins for the histogram (default: 10)
    - `title`: the title of the overall graph (default: Overlap)
    - `y_axis_label`: the label of the y axis (default: Percent of
      Links)

    """
    fig = plt.figure(figsize=(4,3))
    step = 1.0 / num_bins
    bins = numpy.arange(0, 1 + step, step)
    xvals = bins[:-1]
    all_counts = numpy.histogram(stats['all'], bins)[0]
    all_percents = all_counts.astype(float) / all_counts.sum()
    # Determine whether the data contains information on the active
    # interactions, or is just overall data.
    num_subplots = 3 if 'active' in stats else 1
    all_axes = fig.add_subplot(num_subplots, 1, 1, xlim=(-0.5, 1.5),
            ylim=(0, 1),)
            #xlabel='Jaccard Index', ylabel='Percent of Links')
    if num_subplots > 1:
        fig.set_size_inches(4, 8)
        fig.subplots_adjust(
                #bottom=0.1,
                left=0.22,
                right=0.94,
                top=0.88,
                hspace=0.3
        )
        all_axes.set_title('All')
    else:
        all_axes.set_xlabel('Jaccard Index')
        all_axes.set_ylabel(y_axis_label)
        fig.subplots_adjust(
                bottom=0.15,
                left=0.22,
                right=0.94,
                top=0.80,
                hspace=0.3
        )
    fig.suptitle(title, fontsize=15)
    all_axes.bar(
            xvals,
            all_percents,
            width=step,
            color=ALL_BAR_COLOR,
    )
    all_axes.yaxis.set_major_formatter(PERCENTAGE_FORMATTER)
    all_axes.set_title('All')
    class_percents = {'all': all_percents}
    if 'active' in stats:
        active_counts = numpy.histogram(stats['active'], bins)[0]
        active_percents = active_counts.astype(float) / active_counts.sum()
        active_axes = fig.add_subplot(3, 1, 2, sharex=all_axes,
                sharey=all_axes,
            ylabel=y_axis_label)
        active_axes.bar(
                xvals,
                active_percents,
                width=step,
                color=ACTIVE_BAR_COLOR,
        )
        active_axes.yaxis.set_major_formatter(PERCENTAGE_FORMATTER)
        active_axes.set_title('Perturbed')
        class_percents['active'] = active_percents

        inactive_counts = numpy.histogram(stats['inactive'], bins)[0]
        inactive_percents = (inactive_counts.astype(float) /
                inactive_counts.sum())
        inactive_axes = fig.add_subplot(3, 1, 3, sharex=all_axes,
                sharey=all_axes,
                xlabel='Jaccard Index')
        inactive_axes.bar(
                xvals,
                inactive_percents,
                width=step,
                color=INACTIVE_BAR_COLOR,
        )
        inactive_axes.yaxis.set_major_formatter(PERCENTAGE_FORMATTER)
        inactive_axes.set_title('Unperturbed')
        class_percents['inactive'] = inactive_percents
    else:
        all_axes.set_title('')
        all_axes.set_xlabel('Jaccard Index')
        all_axes.set_ylabel(y_axis_label)

    all_axes.set_xlim(-0.05, 1.05)
    all_axes.set_ylim(0, 1)
    for outfile_name in outfile_names:
        fig.savefig(outfile_name)
    return class_percents, bins


def percents_to_avg_and_stddev(all_percents):
    """Returns the column-wise averages for the percents (average per
    bin) and their standard deviations

    :Parameters:
    - `all_percents`: nested list of percents, one list containing
      percents per bin for a single file

    """
    all_percents = numpy.array(all_percents, dtype='float64')
    averages = all_percents.mean(0)
    stddevs = all_percents.std(0)
    return averages, stddevs


def plot_summary_histogram(
        means_and_stddevs,
        outfile_names,
        bins,
        title="Overlap",
        y_axis_label="Percent of Links"
    ):
    """Plots a summary histogram from the overlap statistics.

    :Parameters:
    - `stats`: a dictionary of overlap statistics
    - `outfile_name`: name of the file to write
    - `num_bins`: number of bins for the histogram (default: 10)
    - `title`: the title of the overall graph (default: Overlap)
    - `y_axis_label`: the label of the y axis (default: Percent of
      Links)

    """
    fig = plt.figure(figsize=(4,3))
    step = bins[1] - bins[0]
    xvals = bins[:-1]
    # Determine whether the data contains information on the active
    # interactions, or is just overall data.
    num_subplots = 3 if 'active' in means_and_stddevs else 1
    all_axes = fig.add_subplot(num_subplots, 1, 1, xlim=(-0.5, 1.5),
            ylim=(0, 1),)
    if num_subplots > 1:
        fig.set_size_inches(4, 8)
        fig.subplots_adjust(
                #bottom=0.1,
                left=0.22,
                right=0.94,
                top=0.88,
                hspace=0.3
        )
        all_axes.set_title('All')
    else:
        all_axes.set_xlabel('Jaccard Index')
        all_axes.set_ylabel(y_axis_label)
        fig.subplots_adjust(
                bottom=0.15,
                left=0.22,
                right=0.94,
                top=0.80,
                hspace=0.3
        )
    fig.suptitle(title, fontsize=15)
    all_means, all_stddevs = means_and_stddevs['all']
    all_axes.bar(
            xvals,
            all_means,
            yerr=all_stddevs,
            width=step,
            ecolor='k',
            color=ALL_BAR_COLOR,
    )
    all_axes.yaxis.set_major_formatter(PERCENTAGE_FORMATTER)
    if 'active' in means_and_stddevs:
        active_means, active_stddevs = means_and_stddevs['active']
        active_axes = fig.add_subplot(num_subplots, 1, 2, sharex=all_axes,
                sharey=all_axes,
                ylabel=y_axis_label
        )
        active_axes.bar(
                xvals,
                active_means,
                yerr=active_stddevs,
                width=step,
                ecolor='k',
                color=ACTIVE_BAR_COLOR,
        )
        active_axes.yaxis.set_major_formatter(PERCENTAGE_FORMATTER)
        active_axes.set_title('Perturbed')

        inactive_means, inactive_stddevs = means_and_stddevs['inactive']
        inactive_axes = fig.add_subplot(num_subplots, 1, 3, sharex=all_axes,
                sharey=all_axes,
                xlabel='Jaccard Index'
        )
        inactive_axes.bar(
                xvals,
                inactive_means,
                yerr=inactive_stddevs,
                width=step,
                ecolor='k',
                color=INACTIVE_BAR_COLOR,
        )
        inactive_axes.yaxis.set_major_formatter(PERCENTAGE_FORMATTER)
        inactive_axes.set_title('Unperturbed')

    all_axes.set_xlim(-0.05, 1.05)
    all_axes.set_ylim(0, 1)
    for outfile_name in outfile_names:
        fig.savefig(outfile_name)


def calc_all_within_file_overlaps(
        input_data,
        annotated_interactions,
        significant_links
    ):
    within_file_overlaps = {}
    for i in range(len(input_data.links_files)):
        links_file_name = input_data.links_files[i].name
        logger.info("Calculating within file overlap statistics for "
                "{0}.".format(links_file_name))
        file_overlaps = calc_within_file_overlaps(
            significant_links[i],
            annotated_interactions,
            input_data.activity_threshold,
            input_data.interactions_greater,
            input_data.intraterms
        )
        within_file_overlaps[links_file_name] = file_overlaps
    return within_file_overlaps


def make_file_overlaps_output_paths(
        links_file_name,
        output_dir,
        links_cutoff,
        activity_threshold=None,
        summary=False
    ):
    infile_base, ext = os.path.splitext(os.path.basename(
            links_file_name))
    if summary:
        file_output_dir = output_dir
    else:
        file_output_dir = os.path.join(output_dir, infile_base)
    output_base = os.path.join(file_output_dir, infile_base)
    output_paths = {}
    output_paths['genes'] = '{0}-genes_overlaps-{1:.3g}.tsv'.format(
            output_base,
            links_cutoff
    )
    output_paths['interactions'] = (
            '{0}-intraterm_interaction_overlaps-{1:.3g}.tsv'.format(
                output_base,
                links_cutoff
            )
    )
    output_paths['links'] = '{0}-links_overlaps-{1:.3g}.tsv'.format(
            output_base,
            links_cutoff
    )
    output_paths['interaction_link_tallies'] = (
            '{0}-interaction_tallies-{1:.3g}.tsv'.format(
                output_base,
                links_cutoff
            )
    )
    if activity_threshold is not None:
        for outfile_type, outfile_path in output_paths.items():
            output_paths[outfile_type] = (
                    convutils.append_to_file_base_name(
                        outfile_path,
                        '-{0:.3g}'.format(activity_threshold)
                    )
            )
    output_paths['directory'] = file_output_dir
    return output_paths


def write_all_within_file_statistics(input_data, all_overlap_data):
    for links_file_name, file_statistics in all_overlap_data.items():
        output_paths = make_file_overlaps_output_paths(
                links_file_name,
                input_data.output_dir,
                input_data.links_cutoff,
                input_data.activity_threshold
        )
        if not os.path.isdir(output_paths['directory']):
            os.mkdir(output_paths['directory'])
        # Output the genes overlaps
        genes_outfile_name = output_paths['genes']
        genes_outfile = open(genes_outfile_name, 'wb')
        logger.info("Writing within-file term genes overlaps to {0}.".format(
                genes_outfile_name))
        genes_outfile = open(genes_outfile_name, 'wb')
        if input_data.activity_threshold is None:
            genes_writer = convutils.make_csv_dict_writer(genes_outfile,
                    TERM_GENES_FIELDNAMES)
        else:
            genes_writer = convutils.make_csv_dict_writer(genes_outfile,
                    TERM_GENES_EXPR_FIELDNAMES)
        write_within_link_gene_overlaps(genes_writer,
                file_statistics['genes'])

        # Output the interactions overlaps
        if 'interactions' in file_statistics:
            interactions_outfile_name = output_paths['interactions']
            logger.info("Writing within-file term interactions overlaps to {0}.".format(
                    interactions_outfile_name))
            interactions_outfile = open(interactions_outfile_name, 'wb')
            if input_data.activity_threshold is None:
                interactions_writer = convutils.make_csv_dict_writer(interactions_outfile,
                        TERM_INTERACTIONS_FIELDNAMES)
            else:
                interactions_writer = convutils.make_csv_dict_writer(interactions_outfile,
                        TERM_INTERACTIONS_EXPR_FIELDNAMES)
            write_within_link_interactions_overlaps(
                    interactions_writer,
                    file_statistics['interactions']
            )

        # Output the links overlaps
        links_outfile_name = output_paths['links']
        logger.info("Writing within-file links overlaps to {0}.".format(
                links_outfile_name))
        links_outfile = open(links_outfile_name, 'wb')
        if input_data.activity_threshold is None:
            links_writer = convutils.make_csv_dict_writer(links_outfile,
                    INTRALINKS_FIELDNAMES)
        else:
            links_writer = convutils.make_csv_dict_writer(links_outfile,
                    INTRALINKS_EXPR_FIELDNAMES)
        write_file_links_statistics(links_writer,
                file_statistics['links'])
        links_outfile.close()

        # Output the tallies of links per interaction
        tallies_outfile_name = output_paths['interaction_link_tallies']
        logger.info("Writing interaction tallies to {0}.".format(
                tallies_outfile_name))
        tallies_outfile = open(tallies_outfile_name, 'wb')
        if input_data.activity_threshold is None:
            tallies_writer = convutils.make_csv_dict_writer(
                    tallies_outfile, INTERACTION_TALLIES_FIELDNAMES)
        else:
            tallies_writer = convutils.make_csv_dict_writer(
                    tallies_outfile,
                    INTERACTION_TALLIES_EXPR_FIELDNAMES
            )
        write_interaction_link_tallies(tallies_writer,
                file_statistics['interaction_link_tallies'])
        tallies_outfile.close()


def plot_within_overlaps_histograms(input_data, all_overlap_data):
    percents_by_file = {}
    max_percents_by_file = {}
    bins = None
    for links_file_name, overlap_data in all_overlap_data.items():
        output_paths = make_file_overlaps_output_paths(
                links_file_name,
                input_data.output_dir,
                input_data.links_cutoff,
                input_data.activity_threshold
        )
        file_percents = {}
        file_max_percents = {}
        for overlap_type, type_overlaps in overlap_data.items():
            if overlap_type == 'interaction_link_tallies':
                continue
            summarized_type_overlaps = summarize_jaccards(
                    type_overlaps)
            outfile_base, outfile_ext = os.path.splitext(
                    output_paths[overlap_type])

            histfile_names = ['{0}.{1}'.format(outfile_base, extension)
                    for extension in input_data.extensions]
            #logger.info("Drawing Jaccard histogram to {0}.".format(
                    #histfile_name))
            percents, bins = plot_histogram(
                    summarized_type_overlaps,
                    histfile_names,
                    input_data.num_bins,
                    OVERLAP_CLASS_TO_TITLE[overlap_type],
                    OVERLAP_CLASS_TO_YAXIS[overlap_type]
            )
            file_percents[overlap_type] = percents

            max_type_overlaps = max_jaccards(type_overlaps)
            max_histfile_names = ['{0}-max.{1}'.format(outfile_base,
                    extension) for extension in input_data.extensions]
            #logger.info("Drawing max Jaccard histogram to {0}.".format(
                    #max_histfile_name))
            max_percents, bins = plot_histogram(
                    max_type_overlaps,
                    max_histfile_names,
                    input_data.num_bins,
                    OVERLAP_CLASS_TO_TITLE[overlap_type],
                    OVERLAP_CLASS_TO_YAXIS_MAX[overlap_type]
            )
            file_max_percents[overlap_type] = max_percents
        percents_by_file[links_file_name] = file_percents
        max_percents_by_file[links_file_name] = file_max_percents

    assert bins is not None
    return percents_by_file, max_percents_by_file, bins


def build_summary_dict(percents_by_file, activity=False):
    # Set up the dictionary that's going to contain all the statistics.
    # The keys of the dictionary should be the the categories of data
    # that we recorded, e.g., 'genes' for genes overlaps; the values
    # should themselves be dictionaries, with the keys being 'all',
    # 'active', and 'inactive', or just 'all', if there's no expression
    # data.
    overlap_data_keys = percents_by_file[percents_by_file.keys()[0]].keys()
    if activity:
        ldict = lambda: {
                'all': [],
                'active': [],
                'inactive': [],
        }
    else:
        ldict = lambda: {'all': []}
    overlap_summaries_data = dict((key, ldict()) for key in
            overlap_data_keys)

    # First, gather up all the relevant data.
    for links_file_name, file_data in percents_by_file.items():
        for overlap_type, overlap_data in file_data.items():
            for activity_class, measurement in overlap_data.items():
                overlap_summaries_data[overlap_type][activity_class].append(
                        measurement)
    return overlap_summaries_data


def file_percents_to_means_and_stddevs(
        percents_by_file,
        activity=False
    ):
    file_summaries = build_summary_dict(percents_by_file, activity)
    file_means_and_stddevs = {}
    for overlap_type, type_summaries in file_summaries.items():
        if overlap_type == 'interaction_link_tallies':
            continue
        all_percents = type_summaries['all']
        all_means, all_stddevs = percents_to_avg_and_stddev(
                all_percents)
        type_means_and_stddevs = {'all': (all_means, all_stddevs)}
        if 'active' in type_summaries:
            active_percents = type_summaries['active']
            active_means, active_stddevs = (
                    percents_to_avg_and_stddev(active_percents))
            type_means_and_stddevs['active'] = (active_means,
                    active_stddevs)
            inactive_percents = type_summaries['inactive']
            inactive_means, inactive_stddevs = (
                    percents_to_avg_and_stddev(inactive_percents))
            type_means_and_stddevs['inactive'] = (inactive_means,
                    inactive_stddevs)
        file_means_and_stddevs[overlap_type] = type_means_and_stddevs

    return file_means_and_stddevs


def write_means_and_stddevs(
        input_data,
        file_means_and_stddevs,
        bins,
        summary_name,
    ):
    summary_output_paths = make_file_overlaps_output_paths(
            summary_name,
            input_data.output_dir,
            input_data.links_cutoff,
            input_data.activity_threshold,
            summary=True
    )
    for overlap_type, overlap_means_and_stddevs in file_means_and_stddevs.items():
        outfile_base, outfile_ext = os.path.splitext(
                summary_output_paths[overlap_type])
        summary_outfile_name = outfile_base + '.tsv'
        logger.info("Writing {0} summaries to {1}.".format(
                overlap_type, summary_outfile_name))
        summary_outfile = open(summary_outfile_name, 'wb')
        if input_data.activity_threshold is None:
            summary_writer = convutils.make_csv_dict_writer(
                    summary_outfile, MEANS_AND_STDDEVS_FIELDNAMES)
        else:
            summary_writer = convutils.make_csv_dict_writer(
                    summary_outfile, MEANS_AND_STDDEVS_EXPR_FIELDNAMES)
        means_and_stddevs = {'all': overlap_means_and_stddevs['all']}
        fields = ['bin', 'mean', 'stddev']
        if input_data.activity_threshold is not None:
            means_and_stddevs.update({
                    'active': overlap_means_and_stddevs['active'],
                    'inactive': overlap_means_and_stddevs['inactive'],
            })
            fields.extend(('active_mean', 'active_stddev',
                    'inactive_mean', 'inactive_stddev'))

        out_records = []
        for i in range(len(means_and_stddevs['all'][0])):
            the_bin = bins[i + 1]
            mean = means_and_stddevs['all'][0][i]
            stddev = means_and_stddevs['all'][1][i]
            record = {
                    'bin': the_bin,
                    'mean': mean,
                    'stddev': stddev
            }
            if 'active' in means_and_stddevs:
                active_mean = means_and_stddevs['active'][0][i]
                active_stddev = means_and_stddevs['active'][1][i]
                inactive_mean = means_and_stddevs['inactive'][0][i]
                inactive_stddev = means_and_stddevs['inactive'][1][i]
                record.update({
                        'active_mean': active_mean,
                        'active_stddev': active_stddev,
                        'inactive_mean': inactive_mean,
                        'inactive_stddev': inactive_stddev,
                })
            out_records.append(record)

        summary_writer.writerows(out_records)


def plot_within_overlaps_summary_histograms(
        input_data,
        file_means_and_stddevs,
        bins,
        summary_name,
    ):
    summary_output_paths = make_file_overlaps_output_paths(
            summary_name,
            input_data.output_dir,
            input_data.links_cutoff,
            input_data.activity_threshold,
            summary=True
    )
    for overlap_type, overlap_means_and_stddevs in file_means_and_stddevs.items():
        outfile_base, outfile_ext = os.path.splitext(
                summary_output_paths[overlap_type])
        histfile_names = ['{0}.{1}'.format(outfile_base, extension)
                for extension in input_data.extensions]
        #logger.info("Drawing summary histogram to {0}.".format(
                #histfile_name))
        logger.info("Drawing summary histograms")
        # HACK HACK HACK
        if '-max' in summary_name:
            y_axis_title = OVERLAP_CLASS_TO_YAXIS_MAX[overlap_type]
        else:
            y_axis_title = OVERLAP_CLASS_TO_YAXIS[overlap_type]
        plot_summary_histogram(
                overlap_means_and_stddevs,
                histfile_names,
                bins,
                OVERLAP_CLASS_TO_TITLE[overlap_type],
                y_axis_title
        )


def get_significant_induced_subgraph(
        interactions_graph,
        annotated_interactions,
        links,
        intraterms=False,
        terms=None
    ):
    """Returns the interactions induced by the links.

    :Parameters:
    - `interactions_graph`: graph containing the gene-gene or gene
      product-gene product interactions
    - `annotated_interactions`: a
      `bpn.structures.AnnotatedInteractionsGraph` instance
    - `links`: the significant links
    - `intraterms`: include intraterm links if `True`
    - `terms`: the significant terms

    """
    selected_interactions = set()
    for link in links:
        link_interactions = get_link_interactions(
                link,
                annotated_interactions,
                intraterms
        )
        selected_interactions.update(link_interactions)
    if intraterms and (terms is not None):
        for term in terms:
            try:
                terms_based_interactions = (
                        annotated_interactions.get_intraterm_interactions(
                            term)
                )
                selected_interactions.update(terms_based_interactions)
            except KeyError:
                # Some terms just won't have any intraterm interactions
                continue
    return selected_interactions


def get_activity_induced_subgraphs(interactions,
        active_interactions):
    """Returns the interactions induced by active and inactive
    interactions.

    :Parameters:
    - `interactions`: a `set` of interactions
    - `active_interactions`: a set of interactions noted as "active"

    """
    active_subgraph = interactions.intersection(active_interactions)
    inactive_subgraph = interactions.difference(active_interactions)
    return active_subgraph, inactive_subgraph


def calc_overlap_of_graphs(edges1, edges2):
    """Calculates statistics on the overlap of two graphs."""
    stats = statstools.calculate_overlap_scores(edges1, edges2)
    stats['interactions1'] = stats['set1_size']
    stats['interactions2'] = stats['set2_size']
    del stats['set1_size']
    del stats['set2_size']
    return stats


def calc_interaction_overlaps(
        bpn_1_interactions,
        bpn_2_interactions,
        annotated_interactions,
        activity_cutoff=None,
        greater=True
    ):
    """Calculate the overlap in two interaction graphs.

    Returns a dictionary with key ``'all'`` with a dictionary as a
    value. The dictionary contains keys and values of the following ::

        {
            'interactions1': ...,   # number of interactions in graph1
            'interactions2': ...,   # number of interactions in graph2
            'intersection': ...,    # number of interactions in both
            'union': ...,           # number of interactions in either
            'jaccard': ...,         # Jaccard index of overlap
            'intersection_by_set1', # overlap of intersection to graph1
            'intersection_by_set2'  # overlap of intersection to graph2
        }

    If ``active`` is ``True``, the returned dictionary contains two
    additional keys, ``'active'``, and ``'inactive'``, whose values are
    dictionaries similarly structured to that of ``'all'``.

    :Parameters:
    - `bpn_1_interactions`: graph containing the gene-gene or gene
      product-gene product interactions
    - `bpn_2_interactions`: similar to ``bpn_1_interactions``
    - `annotated_interactions`: a
      `bpn.structures.AnnotatedInteractionsGraph` instance
    - `activity_cutoff`: a numerical threshold value for determining
      whether a gene is active or not; if provided, will calculate
      overlaps for "active" and "inactive" portions of the networks
    - `greater`: if `True`, indicates significant links will be greater
      than or equal too the cutoff, else, indicates they are less than
      or equal to the `cutoff` [default: `True`]

    """
    interaction_overlaps = {}
    interaction_overlaps['all'] = calc_overlap_of_graphs(
            bpn_1_interactions, bpn_2_interactions)
    if activity_cutoff is not None:
        active_interactions = (
                annotated_interactions.get_active_interactions(
                    activity_cutoff, greater))
        active_subgraph1, inactive_subgraph1 = (
                get_activity_induced_subgraphs(bpn_1_interactions,
                    active_interactions))
        active_subgraph2, inactive_subgraph2 = (
                get_activity_induced_subgraphs(bpn_2_interactions,
                    active_interactions))
        interaction_overlaps['active'] = calc_overlap_of_graphs(
                active_subgraph1, active_subgraph2)
        interaction_overlaps['inactive'] = calc_overlap_of_graphs(
                inactive_subgraph1, inactive_subgraph2)
    return interaction_overlaps


def calc_all_between_file_overlaps(
        input_data,
        annotated_interactions,
        significant_links,
        interactions_graph,
        intraterms = None,
        significant_terms = None
    ):
    all_overlap_statistics = {}
    # For every combination of files, do the comparison.
    fcombos = list(itertools.combinations(range(len(significant_links)),
        2))
    # TODO: HACK HACK HACK HACK
    # If there's only one file, we still perform this, because we use
    # the output later to get the basic information for each file. FIX
    # THIS LATER!!!!
    if not fcombos:
        fcombos = [(0, 0)]
    for i, j in fcombos:
        links_files_names = tuple([input_data.links_files[x].name for x
                in i, j])
        logger.info("Comparing interaction overlaps between {0} and "
                "{1}.".format(*links_files_names))
        significant_links_i = significant_links[i]
        significant_links_j = significant_links[j]
        if significant_terms is not None:
            significant_terms_i = significant_terms[i]
            significant_terms_j = significant_terms[j]
        else:
            significant_terms_i = significant_terms_j = None
        subgraph_i = get_significant_induced_subgraph(
                    interactions_graph,
                    annotated_interactions,
                    significant_links_i,
                    intraterms,
                    significant_terms_i
        )
        subgraph_j = get_significant_induced_subgraph(
                    interactions_graph,
                    annotated_interactions,
                    significant_links_j,
                    intraterms,
                    significant_terms_j
        )
        overlap_statistics = calc_interaction_overlaps(
                subgraph_i,
                subgraph_j,
                annotated_interactions,
                input_data.activity_threshold,
                input_data.interactions_greater
        )
        all_overlap_statistics[links_files_names] = overlap_statistics

    return all_overlap_statistics


def write_interactions_overlap_statistics(csvwriter,
        interactions_overlap_statistics):
    """Output the overlap statistics between two interaction graphs."""
    outrecords = []
    all_filenames = sorted(interactions_overlap_statistics.keys())
    for filenames in all_filenames:
        stats = interactions_overlap_statistics[filenames]
        outrecord = {
                'file1': filenames[0],
                'file2': filenames[1],
                'interactions1': stats['all']['interactions1'],
                'interactions2': stats['all']['interactions2'],
                'intersection': stats['all']['intersection'],
                'union': stats['all']['union'],
                'jaccard': stats['all']['jaccard'],
                'intersection_by_set1': stats['all']['intersection_by_set1'],
                'intersection_by_set2': stats['all']['intersection_by_set2'],
        }
        if 'active' in stats:
            outrecord.update({
                'active_interactions1': stats['active']['interactions1'],
                'active_interactions2': stats['active']['interactions2'],
                'active_intersection': stats['active']['intersection'],
                'active_union': stats['active']['union'],
                'active_jaccard': stats['active']['jaccard'],
                'active_intersection_by_set1': stats['active']['intersection_by_set1'],
                'active_intersection_by_set2': stats['active']['intersection_by_set2'],
                'inactive_interactions1': stats['inactive']['interactions1'],
                'inactive_interactions2': stats['inactive']['interactions2'],
                'inactive_intersection': stats['inactive']['intersection'],
                'inactive_union': stats['inactive']['union'],
                'inactive_jaccard': stats['inactive']['jaccard'],
                'inactive_intersection_by_set1': stats['inactive']['intersection_by_set1'],
                'inactive_intersection_by_set2': stats['inactive']['intersection_by_set2'],
            })
        outrecords.append(outrecord)
    csvwriter.writerows(outrecords)


def summarize_btwn_jaccards(file_names, btwn_stats):
    num_files = len(file_names)
    summary = {
            'all': bpn.structures.symzeros(num_files),
            'active': bpn.structures.symzeros(num_files),
            'inactive': bpn.structures.symzeros(num_files)
    }
    activity = False
    # We use indices into the file names so that we get order the files
    # in the heatmap in accordance to how they were entered on the
    # command line, rather than sorting the names in the keys. This way
    # the user will know which square indicates which two files were
    # being compared.
    for (i, j) in itertools.combinations(range(num_files), 2):
        overlap_stats = btwn_stats[file_names[i], file_names[j]]
        summary['all'][i,j] = overlap_stats['all']['jaccard']
        if 'active' in overlap_stats:
            activity = True
            summary['active'][i,j] = overlap_stats['active']['jaccard']
            summary['inactive'][i,j] = overlap_stats['inactive']['jaccard']
    if not activity:
        del summary['active']
        del summary['inactive']
    # Set all the self-comparisons to 1
    for s in summary.values():
        for i in range(num_files):
            s[i,i] = 1
    return summary


def format_heatmaps_axes(axes, tick_labels):
    axes.set_xticklabels(tick_labels)
    # This hides the tick marks
    for t in axes.xaxis.get_ticklines():
        t.set_visible(False)
    axes.set_yticklabels(tick_labels)
    for t in axes.yaxis.get_ticklines():
        t.set_visible(False)


def plot_all_btwn_overlaps_heatmaps(
        outfile_names,
        links_files_names,
        btwn_overlap_statistics
    ):
    jaccard_summaries = summarize_btwn_jaccards(links_files_names,
            btwn_overlap_statistics)

    num_files = len(links_files_names)
    tick_labels = ['{0}'.format(i) for i in range(0, num_files + 1)]
    fig = plt.figure(figsize=(4.5, 11.5))
    fig.suptitle('Overlap of explained interactions\nbetween BPNs',
            fontsize=16)
    num_subplots = 3 if 'active' in jaccard_summaries else 1

    all_axes = fig.add_subplot(num_subplots, 1, 1)
    all_im = all_axes.matshow(
            jaccard_summaries['all'],
            cmap=plt.cm.Purples,
            vmin=0,
            vmax=1
    )
    all_cb = fig.colorbar(all_im)
    all_cb.set_label('Jaccard Index')
    all_axes.set_title('All')
    format_heatmaps_axes(all_axes, tick_labels)

    if 'active' in jaccard_summaries:
        active_axes = fig.add_subplot(num_subplots, 1, 2)
        active_im = active_axes.matshow(
                jaccard_summaries['active'],
                cmap=plt.cm.Reds,
                vmin=0,
                vmax=1
        )
        active_cb = fig.colorbar(active_im)
        active_cb.set_label('Jaccard Index')
        active_axes.set_title('Perturbed')
        format_heatmaps_axes(active_axes, tick_labels)

        inactive_axes = fig.add_subplot(num_subplots, 1, 3)
        inactive_im = inactive_axes.matshow(
                jaccard_summaries['inactive'],
                cmap=plt.cm.Blues,
                vmin=0,
                vmax=1
        )
        inactive_cb = fig.colorbar(inactive_im)
        inactive_cb.set_label('Jaccard Index')
        inactive_axes.set_title('Unperturbed')
        format_heatmaps_axes(inactive_axes, tick_labels)

    for outfile_name in outfile_names:
        logger.info("Drawing BPN heatmaps to {0}.".format(outfile_name))
        fig.savefig(outfile_name)


def collect_bpn_statistics(
        file_names,
        btwn_bpns_comparison_data,
        all_significant_links,
        all_significant_terms
    ):
    all_bpn_statistics = {}
    for compared_files, comparison_statistics in (
            btwn_bpns_comparison_data.items()):
        for i, file_name in enumerate(compared_files):
            if file_name in all_bpn_statistics:
                continue
            file_stats = {}
            intkey = 'interactions{0}'.format(i + 1)
            j = file_names.index(file_name)
            file_stats['links'] = len(all_significant_links[j])
            file_stats['terms'] = len(all_significant_terms[j])
            file_stats['interactions'] = comparison_statistics['all'][
                    intkey]
            if 'active' in comparison_statistics:
                file_stats['active_interactions'] = (
                        comparison_statistics['active'][intkey])
                file_stats['inactive_interactions'] = (
                        comparison_statistics['inactive'][intkey])
            all_bpn_statistics[file_name] = file_stats

    return all_bpn_statistics


def write_bpn_stats(csv_writer, file_names, all_bpn_statistics):
    outrecords = []
    for file_name in file_names:
        record = {'file': file_name}
        for field, value in all_bpn_statistics[file_name].items():
            record[field] = value
        outrecords.append(record)
    csv_writer.writerows(outrecords)


def calc_bpn_stats_means_and_stddevs(all_bpn_statistics):
    collected_fields = collections.defaultdict(list)
    for bpn_stats in all_bpn_statistics.values():
        for field, value in bpn_stats.items():
            collected_fields[field].append(value)
    means_and_stddevs = {}
    for field, values in collected_fields.items():
        a = numpy.array(values, dtype='float64')
        mean = a.mean()
        stddev = a.std()
        means_and_stddevs[field] = (mean, stddev)
    return means_and_stddevs


def write_bpn_stats_means_and_stddevs(csv_writer,
        bpn_means_and_stddevs):
    out_record = {}
    for field, (mean, stddev) in bpn_means_and_stddevs.items():
        out_record['{0}_mean'.format(field)] = mean
        out_record['{0}_stddev'.format(field)] = stddev
    csv_writer.writerow(out_record)


def calc_all_name_overlap_statistics(
        links_files_names,
        links_per_file,
        terms_per_file
    ):
    all_name_overlap_statistics = {}
    # For every combination of files, do the comparison.
    fcombos = itertools.combinations(range(len(links_files_names)), 2)
    for (i, j) in fcombos:
        files = tuple([links_files_names[x] for x
                in (i, j)])
        logger.info("Comparing name overlaps between {0} and "
                "{1}.".format(*links_files_names))
        links_i = links_per_file[i]
        links_j = links_per_file[j]
        link_name_overlaps = statstools.calculate_overlap_scores(
                links_i, links_j)
        terms_i = terms_per_file[i]
        terms_j = terms_per_file[j]
        term_name_overlaps = statstools.calculate_overlap_scores(
                terms_i, terms_j)
        all_name_overlap_statistics[files] = {
                'links': link_name_overlaps,
                'terms': term_name_overlaps
        }
    return all_name_overlap_statistics


def write_name_overlap_statistics(csv_writer, links_files_names,
        all_name_overlap_statistics):
    outrecords = []
    fcombos = itertools.combinations(range(len(links_files_names)), 2)
    for (i, j) in fcombos:
        file1, file2 = tuple([links_files_names[x] for x
                in (i, j)])
        overlap_data = all_name_overlap_statistics[(file1, file2)]
        record = {'file1': file1, 'file2': file2}
        for name_type, name_data in overlap_data.items():
            for key, value in name_data.items():
                record_key = NAME_OVERLAPS_KEY_MAP[name_type][key]
                record[record_key] = value
        outrecords.append(record)
    csv_writer.writerows(outrecords)


def calc_name_overlap_means_and_stddevs(all_name_overlap_statistics):
    links_jaccards = []
    terms_jaccards = []
    for overlap_data in all_name_overlap_statistics.values():
        links_jaccards.append(overlap_data['links']['jaccard'])
        terms_jaccards.append(overlap_data['terms']['jaccard'])
    links_jaccards_array = numpy.array(links_jaccards)
    links_jaccards_mean = links_jaccards_array.mean()
    links_jaccards_stddev = links_jaccards_array.std()
    terms_jaccards_array = numpy.array(terms_jaccards)
    terms_jaccards_mean = terms_jaccards_array.mean()
    terms_jaccards_stddev = terms_jaccards_array.std()
    means_and_stddevs = {
            'links_jaccard_mean': links_jaccards_mean,
            'links_jaccard_stddev': links_jaccards_stddev,
            'terms_jaccard_mean': terms_jaccards_mean,
            'terms_jaccard_stddev': terms_jaccards_stddev,
    }
    return means_and_stddevs


def write_name_overlap_summaries(csv_writer,
        name_overlap_means_and_stddevs):
    csv_writer.writerow(name_overlap_means_and_stddevs)


def main(argv=None):
    cli_parser = LinksStatsCli()
    input_data = cli_parser.parse_args(argv)
    links_files_names = [f.name for f in input_data.links_files]

    logger.info("Constructing links graph.")
    # This is going to cache the significant links per file.
    significant_links = []
    # This is going to cache the significant terms per file.
    significant_terms = []
    for links_file, terms_file in zip(input_data.links_files,
            input_data.terms_files):
        links_reader = convutils.make_csv_reader(links_file)
        if terms_file is not None:
            terms_reader = convutils.make_csv_reader(terms_file)
        else:
            terms_reader = None
        links_graph = make_process_links_graph(
                links_reader,
                input_data.links_cutoff,
                terms_reader,
                input_data.links_greater
        )
        # The link names in the edges are returned in random order, for
        # some reason; make sure in each edge, the link is described
        # such that term1 < term2.
        these_significant_links = set([tuple(sorted(e)) for e in
                links_graph.edges()])
        these_significant_terms = set(links_graph.nodes())
        logger.info("{0} links and {1} terms from {2}.".format(
                len(these_significant_links),
                len(these_significant_terms),
                links_file.name)
        )
        significant_links.append(these_significant_links)
        significant_terms.append(these_significant_terms)

    logger.info("Constructing supporting data structures; this may "
            "take a while...")
    annotated_interactions = bpn.structures.AnnotatedInteractionsGraph(
            input_data.interactions_graph,
            input_data.annotations_dict
    )

    # Now, we go through each file's set of links and calculate
    # statistics for overlap within its own links network.
    within_overlap_statistics = calc_all_within_file_overlaps(
            input_data, annotated_interactions, significant_links)
    write_all_within_file_statistics(input_data,
            within_overlap_statistics)
    percents_by_file, max_percents_by_file, bins = (
            plot_within_overlaps_histograms(input_data,
                within_overlap_statistics)
    )
    summary_means_and_stddevs = file_percents_to_means_and_stddevs(
        percents_by_file,
        (input_data.activity_threshold is not None)
    )
    max_summary_means_and_stddevs = file_percents_to_means_and_stddevs(
        max_percents_by_file,
        (input_data.activity_threshold is not None)
    )
    write_means_and_stddevs(
            input_data,
            summary_means_and_stddevs,
            bins,
            'summary'
    )
    write_means_and_stddevs(
            input_data,
            max_summary_means_and_stddevs,
            bins,
            'summary-max'
    )
    plot_within_overlaps_summary_histograms(
            input_data,
            summary_means_and_stddevs,
            bins,
            'summary'
    )
    plot_within_overlaps_summary_histograms(
            input_data,
            max_summary_means_and_stddevs,
            bins,
            'summary-max'
    )

    # TODO: HACK HACK HACK! See if we can fold this in to the other
    # summarizing functions later, or at least fold the summary results
    # into other summary results.
    # Summarize the links-per-interaction tallies.
    links_per_interaction_per_file = [
            stats['interaction_link_tallies'] for stats in
            within_overlap_statistics.values()
    ]
    links_per_interaction_means_and_stddevs = (
            calc_interaction_links_tallies_means_and_stddevs(
                links_per_interaction_per_file)
    )
    # Hack to get output path.
    link_tallies_summary_file_name = make_file_overlaps_output_paths(
            'summary',
            input_data.output_dir,
            input_data.links_cutoff,
            input_data.activity_threshold,
            summary=True)['interaction_link_tallies']
    logger.info(("Writing interaction link tallies summary to "
            "{0}.").format(link_tallies_summary_file_name))
    link_tallies_summary_file = open(link_tallies_summary_file_name,
            'wb')
    if input_data.activity_threshold is None:
        link_tallies_summary_csvwriter = convutils.make_csv_dict_writer(
                link_tallies_summary_file,
                INTERACTION_TALLIES_SUMMARY_FIELDNAMES
        )
    else:
        link_tallies_summary_csvwriter = convutils.make_csv_dict_writer(
                link_tallies_summary_file,
                INTERACTION_TALLIES_SUMMARY_EXPR_FIELDNAMES
        )
    write_interaction_links_tallies_means_and_stddevs(
            link_tallies_summary_csvwriter,
            links_per_interaction_means_and_stddevs
    )

    # Now we go through each pair of files and calculate the overlap
    # between their links networks.
    btwn_overlap_statistics = calc_all_between_file_overlaps(
            input_data,
            annotated_interactions,
            significant_links,
            input_data.interactions_graph,
            input_data.intraterms,
            significant_terms
    )

    heatmap_file_names = [os.path.join(input_data.output_dir,
            'bpn_interactions_comparisons.{0}'.format(extension)) for
            extension in input_data.extensions]
    plot_all_btwn_overlaps_heatmaps(
            heatmap_file_names,
            links_files_names,
            btwn_overlap_statistics
    )

    btwn_overlaps_outfile_name = os.path.join(input_data.output_dir,
            'bpn_interactions_comparisons.tsv')
    btwn_overlaps_outfile = open(btwn_overlaps_outfile_name, 'wb')
    logger.info("Writing comparison to {0}.".format(
            btwn_overlaps_outfile_name))
    if input_data.activity_threshold is None:
        btwn_overlaps_out_writer = convutils.make_csv_dict_writer(
                btwn_overlaps_outfile,
                INTERFILE_INTERACTIONS_OVERLAP_FIELDNAMES
        )
    else:
        btwn_overlaps_out_writer = convutils.make_csv_dict_writer(
                btwn_overlaps_outfile,
                INTERFILE_INTERACTIONS_OVERLAP_EXPR_FIELDNAMES
        )

    write_interactions_overlap_statistics(btwn_overlaps_out_writer,
            btwn_overlap_statistics)
    btwn_overlaps_outfile.close()

    # Calculate the overlap of the BPNs purely by the names of the
    # links.
    if len(input_data.links_files) > 1:
        bpn_name_overlap_statistics = calc_all_name_overlap_statistics(
                links_files_names,
                significant_links,
                significant_terms
        )
        name_overlaps_file_name = os.path.join(input_data.output_dir,
                'bpn_name_overlaps.tsv')
        logger.info("Writing name overlaps to {0}.".format(
                name_overlaps_file_name))
        name_overlaps_outfile = open(name_overlaps_file_name, 'wb')
        name_overlaps_out_writer = convutils.make_csv_dict_writer(
                name_overlaps_outfile, NAME_OVERLAPS_FIELDNAMES)
        write_name_overlap_statistics(name_overlaps_out_writer,
                links_files_names, bpn_name_overlap_statistics)

        bpn_name_overlap_means_and_stddevs = (
                calc_name_overlap_means_and_stddevs(
                    bpn_name_overlap_statistics)
        )
        name_overlaps_summaries_file_name = os.path.join(
                input_data.output_dir, 'bpn_name_overlaps_summary.tsv')
        logger.info("Writing name overlaps summaries to {0}.".format(
                name_overlaps_summaries_file_name))
        name_overlap_summaries_file = open(
                name_overlaps_summaries_file_name, 'wb')
        name_overlap_sumaries_out_writer = convutils.make_csv_dict_writer(
                name_overlap_summaries_file,
                NAME_OVERLAPS_SUMMARY_FIELDNAMES
        )
        write_name_overlap_summaries(name_overlap_sumaries_out_writer,
                bpn_name_overlap_means_and_stddevs)

    # Now get the basic statistics on each BPN: number of links,
    # terms, interactions, etc.
    all_bpn_statistics = collect_bpn_statistics(
            links_files_names,
            btwn_overlap_statistics,
            significant_links,
            significant_terms
    )
    bpn_stats_file_name = os.path.join(input_data.output_dir,
            'bpn_statistics.tsv')
    logger.info("Writing statistics for all BPNs to {0}".format(
            bpn_stats_file_name))
    bpn_stats_file = open(bpn_stats_file_name, 'wb')
    if input_data.activity_threshold is None:
        bpn_stats_out_writer = convutils.make_csv_dict_writer(
                bpn_stats_file, BPN_STATISTICS_FIELDNAMES)
    else:
        bpn_stats_out_writer = convutils.make_csv_dict_writer(
                bpn_stats_file, BPN_STATISTICS_EXPR_FIELDNAMES)
    write_bpn_stats(bpn_stats_out_writer, links_files_names,
            all_bpn_statistics)
    bpn_stats_file.close()

    bpn_stats_means_and_stddevs = calc_bpn_stats_means_and_stddevs(
            all_bpn_statistics)
    bpn_stats_summary_file_name = os.path.join(input_data.output_dir,
            'bpn_statistics_summary.tsv')
    logger.info(("Writing summary of statistics for all BPNs to "
            "{0}.").format(bpn_stats_summary_file_name))
    bpn_stats_summary_file = open(bpn_stats_summary_file_name, 'wb')
    if input_data.activity_threshold is None:
        bpn_stats_summary_out_writer = convutils.make_csv_dict_writer(
                bpn_stats_summary_file,
                BPN_STATISTICS_SUMMARY_FIELDNAMES
        )
    else:
        bpn_stats_summary_out_writer = convutils.make_csv_dict_writer(
                bpn_stats_summary_file,
                BPN_STATISTICS_SUMMARY_EXPR_FIELDNAMES
        )
    write_bpn_stats_means_and_stddevs(bpn_stats_summary_out_writer,
            bpn_stats_means_and_stddevs)
    bpn_stats_summary_file.close()


if __name__ == '__main__':
    main()

