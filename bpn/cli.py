#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright (c) 2011 Christopher D. Lasher
#
# This software is released under the MIT License. Please see
# LICENSE.txt for details.


"""Command line interfaces to the BPLN programs."""


LINKS_OUTFILE='links_results.tsv'


import bz2
import datetime
import itertools
import os
import sys

import conflictsparse
from convutils import convutils

import cbpn
import logconf
import mcmc.defaults
import parsers
import structures

import logging
logger = logging.getLogger('bpn.cli')


class BaseArgParser(object):
    """Command line parser base for BPLN programs."""
    # Set this to `None` so we can tell if we've already declared what
    # program we are; used later in creating the log file.
    _prog_name = None

    def __init__(self):
        """Create a new instance."""
        self.timestamp = datetime.datetime.now().strftime(
                '%Y-%m-%d-%H%M%S%f')
        self.logfile_template = '{0}-{{0}}.log'.format(self._prog_name)
        self.default_logfile_name = self.logfile_template.format(
                self.timestamp)
        self.make_cli_parser()


    def make_cli_parser(self):
        """Create the command line interface parser."""
        usage = """\
python %prog [OPTIONS] INTERACTIONS_FILE ANNOTATIONS_FILE

ARGUMENTS:
    INTERACTIONS_FILE: a CSV file containing interactions. The file
        should have two columns with headings "interactor1" and
        "interactor2". The file may have additional columns, which will
        be ignored.
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
          columns, which will be ignored.\
"""
        self.cli_parser = conflictsparse.ConflictsOptionParser(usage)
        self.cli_parser.add_option('--links-outfile',
                default=LINKS_OUTFILE,
                help=("the file to which the links results should "
                    "be written [default: %default]")
        )
        self.cli_parser.add_option('--logfile',
                help=("the file to which information for the run will "
                    "be logged [default: {0}]".format(
                        self.logfile_template.format('TIMESTAMP'))
                )
        )


    def check_num_arguments(self):
        """Verifies that the number of arguments given is correct."""
        if len(self.args) != 2:
            self.cli_parser.error("Please provide paths to an "
                    "interactions file and an annotations file.")


    def are_readable_files(self, fnames):
        """Verifies the user has read-permissions to the files."""
        for fname in fnames:
            if not os.access(fname, os.R_OK):
                self.cli_parser.error("%s doesn't exist or you do "
                        "not have read permissions to it." % fname)


    def check_arguments(self):
        """Verifies the arguments given are compatible with the
        program.

        """
        self.check_num_arguments()
        self.are_readable_files(self.args)


    def _post_process_opts_and_args(self):
        """Process the options and arguments after parsing.

        A convenience method, meant to act as a hook for subclasses that
        need to manipulate the options and arguments before returning
        them.
        """
        if not self.opts.logfile:
            self.opts.logfile = self.default_logfile_name


    def parse_args(self, argv=None):
        """Parse the command line, verify compatibility, and return them
        as arguments and options.

        :Parameters:
        - `argv`: the command line arguments [OPTIONAL]

        """
        self.opts, self.args = self.cli_parser.parse_args(argv)
        self.check_arguments()
        self._post_process_opts_and_args()
        return self.opts, self.args


class BplnArgParser(BaseArgParser):
    """Command line parser for BPLN."""

    _prog_name = 'bpln'

    def make_cli_parser(self):
        """Create the command line interface parser."""
        super(BplnArgParser, self).make_cli_parser()
        links_opt = self.cli_parser.add_option('--selected-links',
                help=("A CSV-formatted file containing pairs of "
                    "terms to test. Tests will be done to decide "
                    "if the annotation term from the first column "
                    "\"is linked to\" the annotation term from the "
                    "second column. [NOTE: Selecting this option "
                    "restricts the program to only test the matches "
                    "designated in the file.] [NOTE: This option "
                    "conflicts with '--selected-terms' and "
                    "'--selected-terms-with-all'.]"
                )
        )
        anns_opt = self.cli_parser.add_option('--selected-terms',
                help=("A file containing annotation terms to test "
                    "linkage to each other. The file should contain one "
                    "term per line. Selecting this option restricts the "
                    "program to only testing the given terms against "
                    "each  other. [NOTE: This option conflicts with "
                    "'--selected-links' and "
                    "'--selected-terms-with-all'.]"
                )
        )
        anns_all_opt = self.cli_parser.add_option(
                '--selected-terms-with-all',
                help=("A file containing annotation terms to test "
                    "linkage to all other terms (one-against-all and "
                    "all-against-one). The file should contain one "
                    "term per line. Selecting this option restricts "
                    "the program to only testing the given terms "
                    "against all other terms. [NOTE: "
                    "This option conflicts with '--selected-links' and "
                    "'--selected-terms'.]"
                )
        )
        self.cli_parser.register_conflict(
                (links_opt, anns_opt, anns_all_opt))


class ExpressionBasedArgParser(BaseArgParser):
    """Command line parser for the expression-based BPLN programs."""

    def make_cli_parser(self):
        """Create the command line interface parser for expression-based
        BPLN programs.

        """
        super(ExpressionBasedArgParser, self).make_cli_parser()
        usage = """\
python %prog [OPTIONS] INTERACTIONS_FILE ANNOTATIONS_FILE EXPRESSION_FILE

ARGUMENTS:
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
    EXPRESSION_FILE: a CSV file of gene (or gene product) expression
        values. The file should have a column titled "id" which has the
        gene (or gene product) ID, and a column titled "expression"
        which gives a value for the expression level, or difference in
        expression levels.\
"""
        self.cli_parser.set_usage(usage)


    def check_num_arguments(self):
        """Verifies that the number of arguments given is correct."""
        if len(self.args) != 3:
            self.cli_parser.error(
                    "Please provide paths to an interactions file, "
                    "an annotations file, and an expressions file."
            )


class ContextualArgParser(BplnArgParser, ExpressionBasedArgParser):
    """Command line parser for Contextual BPLN."""

    _prog_name = 'cbpn'

    def make_cli_parser(self):
        """Create the command line interface for Contextual BPLN."""
        super(ContextualArgParser, self).make_cli_parser()
        self.cli_parser.add_option('--num-permutations', type='int',
                default=cbpn.NUM_PERMUTATIONS,
                help=("number of permutations for statistics "
                    "[default: %default]")
        )
        self.cli_parser.add_option('-s', '--edge-swaps', type='int',
                help=("Perform the given number of edge swaps to "
                    "produce random graphs. [NOTE: using this option "
                    "changes the algorithm for determining "
                    "significance of a link between each given pair "
                    "of terms.]"
                )
        )
        self.cli_parser.add_option('--no-estimation', dest='estimate',
                action='store_false', default=True,
                help=("Do not use p-value estimation, but run the "
                    "full number of permutations for every pair of "
                    "annotation terms. [NOTE: this can substantially "
                    "increase running time.]"
                )
        )
        self.cli_parser.add_option('--score-correction',
                action='store_true', default=False,
                help=("Correct scores for each pair of terms by an "
                    "\"expected\" value calculated from the mean "
                    "expression value."
                )
        )


class McmcArgParser(ExpressionBasedArgParser):
    """Command line parser for MCMC BPLN."""

    _prog_name = 'mcmcbpn'

    def make_cli_parser(self):
        """Create the command line interface for MCMC BPLN."""
        super(McmcArgParser, self).make_cli_parser()
        self.cli_parser.add_option('--burn-in', type='int',
                default=mcmc.defaults.BURN_IN,
                help=("the number of steps to take before recording states "
                    "in the Markov chain [default: %default]")
        )
        self.cli_parser.add_option('--steps', type='int',
                default=mcmc.defaults.NUM_STEPS,
                help=("the number of steps through the Markov chain to "
                    "observe [default: %default]")
        )
        self.cli_parser.add_option('--activity-threshold',
                type='float',
                default=mcmc.defaults.ACTIVITY_THRESHOLD,
                help=("set the (differential) expression threshold at "
                    "which a gene is considered active [default: "
                    "%default=-log10(0.05)]")
        )
        self.cli_parser.add_option('--transition-ratio', type='float',
                default=mcmc.defaults.TRANSITION_TYPE_RATIO,
                help=("The target ratio of proposed link transitions "
                    "to proposed parameter transitions [default: "
                    "%default]"
                )
        )
        self.cli_parser.add_option('--link-false-pos', type='float',
                help=("designate the starting false-positive rate "
                    "for links")
        )
        self.cli_parser.add_option('--link-false-neg', type='float',
                help=("designate the starting false-negative rate "
                    "for links")
        )
        self.cli_parser.add_option('--link-prior', type='float',
                help=("designate the starting prior probability "
                    "for adding a link")
        )
        self.cli_parser.add_option('--term-false-pos', type='float',
                help=("designate the starting false-positive rate "
                    "for terms [NOTE: applies only when used in "
                    "conjunction with '--genes-based']"
                )
        )
        self.cli_parser.add_option('--term-false-neg', type='float',
                help=("designate the starting false-negative rate "
                    "for terms [NOTE: applies only when used in "
                    "conjunction with '--genes-based']"
                )
        )
        self.cli_parser.add_option('--term-prior', type='float',
                help=("designate the starting prior probability "
                    "for adding a term [NOTE: applies only when used "
                    "in conjunction with '--terms-based'"
                )
        )
        self.cli_parser.add_option('--seed-terms',
                help=("A file containing terms, one per line to "
                    "use as a seed when initializing the Markov chain.")
        )
        self.cli_parser.add_option('--seed-links',
                help=("A two-column CSV-formatted file containing "
                    "pairs of terms to use as seed for links when "
                    "initializing the Markov chain.")
        )
        self.cli_parser.add_option('--relaxed-coannotations',
                dest='stringent_coannotations', action='store_false',
                default=True,
                help=("use a more relaxed definition of co-annotation "
                    "both genes may be annotated by both terms")
        )
        self.cli_parser.add_option('--fixed-distributions',
                action='store_true',
                help=("use fixed distributions for link (and term) "
                    "prior [implies --free-parameters]")
        )
        self.cli_parser.add_option('--free-parameters',
                action='store_true',
                help=("parameters will be adjusted randomly, rather "
                    "than incrementally")
        )
        self.cli_parser.add_option('--disable-swaps', action='store_true',
                help=("disables swapping links as an option for "
                    "transitions")
        )
        self.cli_parser.add_option('--terms-based', action='store_true',
                help=("uses terms-based, rather than links-based model "
                    "[implies ``--disable-swaps``]")
        )
        self.cli_parser.add_option('--intraterms', action='store_true',
                help=("consider also intraterm interactions [NOTE: "
                    "only available in conjunction with "
                    "'--terms-based']")
        )
        self.cli_parser.add_option('--independent-terms',
                action='store_true',
                help=("allow terms to be selected indpendently from "
                    "links [NOTE: only available in conjunction with "
                    "--terms-based']"
                )
        )
        self.cli_parser.add_option('--genes-based', action='store_true',
                help="overlap of terms is based on genes")
        self.cli_parser.add_option('--parameters-outfile',
                default=mcmc.defaults.PARAMETERS_OUTFILE,
                help=("the file to which the parameters results should "
                    "be written [default: %default]")
        )
        self.cli_parser.add_option('--terms-outfile',
                default=mcmc.defaults.TERMS_OUTFILE,
                help=("the file to which the terms results should "
                    "be written [default: %default]")
        )
        self.cli_parser.add_option('--transitions-outfile',
                default=mcmc.defaults.TRANSITIONS_OUTTFILE,
                help=("the file to which the transitions data should "
                    "be written [default: %default]")
        )
        self.cli_parser.add_option('--detailed-transitions',
                action='store_true',
                help=("Transitions file includes full information about "
                    "each step's state.")
        )
        self.cli_parser.add_option('--bzip2', action='store_true',
                help="compress transitions file using bzip2"
        )
        self.cli_parser.add_option('--record-frequencies',
                action='store_true',
                help="record the frequency of each state"
        )
        self.cli_parser.add_option('--frequencies-outfile',
                default=mcmc.defaults.FREQUENCIES_OUTFILE,
                help=("the file to which frequency information "
                    "should be written [default: %default]")
        )

class SaArgParser(ExpressionBasedArgParser):
    """Command line parser for SA BPLN."""

    _prog_name = 'sabpn'

    def make_cli_parser(self):
        """Create the command line interface for SA BPLN."""
        super(SaArgParser, self).make_cli_parser()
        self.cli_parser.add_option('--steps', type='int',
                default=mcmc.defaults.NUM_STEPS,
                help=("the number of steps to Anneal. "
				"[default: %default]")
        )
        self.cli_parser.add_option('--temperature', type='int',
                default=mcmc.defaults.TEMPERATURE,
                help=("the starting temperature to anneal from. "
                                "[default: %default]")
        )
        self.cli_parser.add_option('--end_temperature', type='int',
                default=mcmc.defaults.END_TEMPERATURE,
                help=("the temperature to end annealing."
                                "[default: %default]")
        )
        self.cli_parser.add_option('--activity-threshold',
                type='float',
                default=mcmc.defaults.ACTIVITY_THRESHOLD,
                help=("set the (differential) expression threshold at "
                    "which a gene is considered active [default: "
                    "%default=-log10(0.05)]")
        )
        self.cli_parser.add_option('--free-parameters',
                action='store_true',
                help=("parameters will be adjusted randomly, rather "
                    "than incrementally")
        )
        self.cli_parser.add_option('--disable-swaps', action='store_true',
                help=("disables swapping links as an option for "
                    "transitions")
        )
        self.cli_parser.add_option('--transition-ratio', type='float',
                default=0.9,
                help=("The target ratio of proposed link transitions "
                    "to proposed parameter transitions [default: "
                    "%default]"
                )
        )
        self.cli_parser.add_option('--parameters-outfile',
                default=mcmc.defaults.PARAMETERS_OUTFILE,
                help=("the file to which the parameters results should "
                    "be written [default: %default]")
        )
        self.cli_parser.add_option('--transitions-outfile',
                default=mcmc.defaults.TRANSITIONS_OUTTFILE,
                help=("the file to which the transitions data should "
                    "be written [default: %default]")
        )
        self.cli_parser.add_option('--detailed-transitions',
                action='store_true',
                help=("Transitions file includes full information about "
                    "each step's state.")
        )
        self.cli_parser.add_option('--bzip2', action='store_true',
                help="compress transitions file using bzip2"
        )


class BplnCli(object):
    """Command line interface for BPLN."""
    def __init__(self):
        self.cli_parser = BplnArgParser()

    def _begin_logging(self):
        """Hook method to control setting up logging."""
        logconf.set_up_root_logger(self.opts.logfile)


    def parse_selected_links(self, selected_links_file_name):
        logger.info("Parsing selected links file %s." %
                selected_links_file_name)
        selected_links_file = open(selected_links_file_name, 'rb')
        num_links = convutils.count_lines(selected_links_file)
        # Reset the file to the beginning to parse it.
        selected_links_file.seek(0)
        links = parsers.parse_selected_links_file(
                selected_links_file)
        return links, num_links


    def _calc_num_links_selected_terms(self, num_selected_terms):
        num_links = num_selected_terms * (num_selected_terms - 1)
        return num_links


    def make_selected_terms_links(self, selected_terms):
        """Create an iterator of pairs off terms from selected terms
        against each other.

        :Parameters:
        - `selected_terms`: an iterator of annotation terms

        """
        return itertools.permutations(selected_terms, 2)


    def _calc_num_links_selected_with_all(self,
            num_selected_terms):
        num_links = 2 * (num_selected_terms *
                len(self.annotations_dict) - num_selected_terms)
        return num_links


    def make_selected_terms_links_with_all(self,
            selected_terms, annotations_dict):
        """Create an iterator of pairs of terms from selected
        terms against all other annotation terms.

        :Parameters:
        - `selected_terms`: an iterator of annotation terms
        - `annotations_dict`: a dictionary with annotation terms as keys
          and `set`s of genes as values

        """
        for term in selected_terms:
            for other_term in annotations_dict.iterkeys():
                if term != other_term:
                    yield term, other_term
                    yield other_term, term


    def make_all_possible_links(self, annotations_dict):
        num_links = len(annotations_dict) ** 2 - len(annotations_dict)
        links = itertools.permutations(annotations_dict.iterkeys(), 2)
        return links, num_links


    def _construct_links_of_interest(self):
        """Construct an iterator for all pairs of annotation terms to be
        tested.

        """
        if self.opts.selected_links:
            links, num_links = self.parse_selected_links(
                    self.opts.selected_links)
        elif (self.opts.selected_terms or
                self.opts.selected_terms_with_all):
            if self.opts.selected_terms:
                selected_terms_file_name = self.opts.selected_terms
            else:
                selected_terms_file_name = \
                        self.opts.selected_terms_with_all
            logger.info("Parsing selected terms file %s." %
                    selected_terms_file_name)
            selected_terms_file = open(
                    selected_terms_file_name, 'rb')
            num_selected_terms = convutils.count_lines(
                    selected_terms_file)
            selected_terms = \
                    parsers.parse_selected_terms_file(
                            selected_terms_file)
            if self.opts.selected_terms:
                num_links = self._calc_num_links_selected_terms(
                        num_selected_terms)
                links = self.make_selected_terms_links(
                        selected_terms)
            else:
                num_links = self._calc_num_links_selected_with_all(
                        num_selected_terms)
                links = self.make_selected_terms_links_with_all(
                        selected_terms, self.annotations_dict)
        else:
            links, num_links = self.make_all_possible_links(
                    self.annotations_dict)

        self.links, self.num_links = links, num_links


    def _process_input_files(self):
        interactions_file = open(self.args[0], 'rb')
        annotations_file = open(self.args[1], 'rb')
        # Create interaction graph
        logger.info("Parsing interactions from {0}.".format(
                interactions_file.name))
        self.interactions_graph = \
                parsers.parse_interactions_file_to_graph(
                        interactions_file)
        logger.info("{0} genes (products) with {1} interactions "
                "parsed.".format(
                    len(self.interactions_graph),
                    self.interactions_graph.number_of_edges()
                )
        )

        # Create dictionary of annotations to genes, but only for genes in
        # the interaction graph
        logger.info("Parsing annotations from {0}.".format(
                annotations_file.name))
        if annotations_file.name.endswith('.gmt'):
            self.annotations_dict = parsers.parse_gmt_to_dict(
                    annotations_file)
        else:
            self.annotations_dict = parsers.parse_annotations_to_dict(
                    annotations_file)

        self.annotations_stats = structures.get_annotations_stats(
                self.annotations_dict)
        logger.info(
                ("{num_total_annotations} annotations processed, "
                "for {num_genes} genes (or gene products), by "
                "{num_annotation_terms} different terms.".format(
                    **self.annotations_stats
                ))
        )

        # Remove from the graph the set of nodes that have no annotation.
        logger.info("Pruning unannotated genes (products) from "
                "interaction graph.")
        self.interactions_graph.prune_unannotated_genes(
                self.annotations_dict)
        logger.info("{0} genes (products) with {1} interactions "
                "remaining in graph.".format(
                    len(self.interactions_graph),
                    self.interactions_graph.number_of_edges()
                )
        )

        # Remove from the annotations any genes which are not in the graph.
        logger.info("Removing genes with no interactions from the "
                "sets of annotated genes.")
        self.interactions_graph.prune_non_network_genes_from_annotations(
                self.annotations_dict)
        self.annotations_stats = structures.get_annotations_stats(
                self.annotations_dict)
        logger.info("{num_total_annotations} annotations, "
                "for {num_genes} genes (or gene products), by "
                "{num_annotation_terms} different terms "
                "remain.".format(
                    **self.annotations_stats
                )
        )

        # Sanity test: the number of genes (products) in the
        # interactions_graph should equal the union of all the sets in
        # annotations_dict
        assert len(self.interactions_graph) == \
                self.annotations_stats['num_genes'], \
                "interactions_graph and annotations_dict have unequal " \
                "numbers of genes!"

        for term, genes in self.annotations_dict.iteritems():
            assert len(genes) > 0, "%s has no genes!" % term

        interactions_file.close()
        annotations_file.close()


    def _open_output_files(self):
        """Opens the output files."""
        self.links_outfile = open(self.opts.links_outfile, 'wb')


    def _construct_data_struct(self):
        data = structures.BplnInputData(
                interactions_graph=self.interactions_graph,
                annotations_dict=self.annotations_dict,
                annotations_stats=self.annotations_stats,
                links=self.links,
                num_links=self.num_links,
                links_outfile=self.links_outfile
        )
        return data


    def parse_args(self, argv=None):
        """Parse the command line arguments.

        :Parameters:
        - `argv`: the command line arguments [OPTIONAL]

        Returns a `BplnInputData` instance.

        """
        self.opts, self.args = self.cli_parser.parse_args(argv)
        self._begin_logging()
        if argv is None:
            argv = sys.argv
        logger.info(' '.join(argv))
        self._process_input_files()
        self._construct_links_of_interest()
        self._open_output_files()
        data = self._construct_data_struct()
        return data


class ContextualCli(BplnCli):
    """Command line interface for Contextual BPLN."""
    def __init__(self):
        self.cli_parser = ContextualArgParser()


    def _process_input_files(self):
        super(ContextualCli, self)._process_input_files()
        expression_file = open(self.args[2], 'rb')
        # Get the expression values.
        logger.info("Parsing expression values from %s." %
                expression_file.name)
        expression_values = parsers.parse_expression_file(
                expression_file)
        # Apply the expression values to the interaction graph, removing
        # any nodes lacking expression values from the graph.
        logger.info("Removing genes without expression values from "
                "interaction graph and annotation sets.")
        self.interactions_graph.apply_expression_values_to_interactions_graph(
                expression_values)
        # Re-synchronize the interaction graph and annotations dictionary.
        self.interactions_graph.prune_non_network_genes_from_annotations(
                self.annotations_dict)
        expression_file.close()

        self.annotations_stats = structures.get_annotations_stats(
                self.annotations_dict)
        gene_stats = {
                'num_interactions': self.interactions_graph.number_of_edges()
        }
        gene_stats.update(self.annotations_stats)
        logger.info("%(num_genes)d genes (products) with "
                "%(num_interactions)d interactions remaining in "
                "graph, with %(num_total_annotations)d annotations by "
                "%(num_annotation_terms)d terms." % gene_stats
        )


    def _construct_data_struct(self):
        data = structures.ContextualInputData(
                interactions_graph=self.interactions_graph,
                annotations_dict=self.annotations_dict,
                annotations_stats=self.annotations_stats,
                links=self.links,
                num_links=self.num_links,
                links_outfile=self.links_outfile,
                num_permutations=self.opts.num_permutations,
                edge_swaps=self.opts.edge_swaps,
                estimate=self.opts.estimate,
                score_correction=self.opts.score_correction
        )
        return data


class McmcCli(ContextualCli):
    """Command line interface for MCMC BPLN."""
    def __init__(self):
        self.cli_parser = McmcArgParser()


    def _process_input_files(self):
        super(McmcCli, self)._process_input_files()
        if self.opts.seed_terms:
            logger.info("Parsing seed terms from {0}.".format(
                    self.opts.seed_terms))
            seed_terms_file = open(self.opts.seed_terms)
            self.seed_terms = [line.strip() for line in seed_terms_file]
        else:
            self.seed_terms = None
        if self.opts.seed_links:
            logger.info("Parsing seed links from {0}.".format(
                    self.opts.seed_links))
            seed_links_file = open(self.opts.seed_links, 'rb')
            seed_links = [tuple(sorted(link)) for link in
                    parsers.parse_selected_links_file(seed_links_file)]
            self.seed_links = seed_links
        else:
            self.seed_links = None


    def _open_output_files(self):
        super(McmcCli, self)._open_output_files()
        if self.opts.terms_based:
            self.terms_outfile = open(self.opts.terms_outfile, 'wb')
        else:
            self.terms_outfile = None
        self.parameters_outfile = open(self.opts.parameters_outfile,
                'wb')
        if self.opts.bzip2:
            if not self.opts.transitions_outfile.endswith('.bz2'):
                bz2_filename = self.opts.transitions_outfile + '.bz2'
            else:
                bz2_filename = self.opts.transitions_outfile
            self.transitions_outfile = bz2.BZ2File(bz2_filename, 'w')
        else:
            self.transitions_outfile = open(
                    self.opts.transitions_outfile, 'wb')
        if self.opts.record_frequencies:
            self.frequencies_outfile = open(self.opts.frequencies_outfile, 'wb')
        else:
            self.frequencies_outfile = None


    def _construct_links_of_interest(self):
        # Overridden to do nothing since it's not applicable to MCMC
        # BPLN. It's not elegant, but it works.
        pass


    def _construct_data_struct(self):
        data = structures.McmcInputData(
                interactions_graph=self.interactions_graph,
                annotations_dict=self.annotations_dict,
                annotations_stats=self.annotations_stats,
                burn_in=self.opts.burn_in,
                steps=self.opts.steps,
                activity_threshold=self.opts.activity_threshold,
                link_false_pos=self.opts.link_false_pos,
                link_false_neg=self.opts.link_false_neg,
                link_prior=self.opts.link_prior,
                term_false_pos=self.opts.term_false_pos,
                term_false_neg=self.opts.term_false_neg,
                term_prior=self.opts.term_prior,
                seed_terms=self.seed_terms,
                seed_links=self.seed_links,
                stringent_coannotations=self.opts.stringent_coannotations,
                fixed_distributions=self.opts.fixed_distributions,
                free_parameters=self.opts.free_parameters,
                disable_swaps=self.opts.disable_swaps,
                terms_based=self.opts.terms_based,
                intraterms=self.opts.intraterms,
                independent_terms=self.opts.independent_terms,
                genes_based=self.opts.genes_based,
                transition_ratio=self.opts.transition_ratio,
                terms_outfile=self.terms_outfile,
                links_outfile=self.links_outfile,
                transitions_outfile=self.transitions_outfile,
                parameters_outfile=self.parameters_outfile,
                detailed_transitions=self.opts.detailed_transitions,
                record_frequencies=self.opts.record_frequencies,
                frequencies_outfile=self.frequencies_outfile
        )
        return data


class SaCli(ContextualCli):
    """Command line interface for SA BPLN."""
    def __init__(self):
        self.cli_parser = SaArgParser()


    def _open_output_files(self):
        super(SaCli, self)._open_output_files()
        self.parameters_outfile = open(self.opts.parameters_outfile,
                'wb')
        if self.opts.bzip2:
            if not self.opts.transitions_outfile.endswith('.bz2'):
                bz2_filename = self.opts.transitions_outfile + '.bz2'
            else:
                bz2_filename = self.opts.transitions_outfile
            self.transitions_outfile = bz2.BZ2File(bz2_filename, 'w')
        else:
            self.transitions_outfile = open(
                    self.opts.transitions_outfile, 'wb')


    def _construct_links_of_interest(self):
        # Overridden to do nothing since it's not applicable to MCMC
        # BPLN. It's not elegant, but it works.
        pass


    def _construct_data_struct(self):
        data = structures.SaInputData(
                interactions_graph=self.interactions_graph,
                annotations_dict=self.annotations_dict,
                annotations_stats=self.annotations_stats,
                steps=self.opts.steps,
                temperature=self.opts.temperature,
                end_temperature=self.opts.end_temperature,
		activity_threshold=self.opts.activity_threshold,
                free_parameters=self.opts.free_parameters,
                disable_swaps=self.opts.disable_swaps,
                transition_ratio=self.opts.transition_ratio,
                links_outfile=self.links_outfile,
                transitions_outfile=self.transitions_outfile,
                parameters_outfile=self.parameters_outfile,
                detailed_transitions=self.opts.detailed_transitions
        )
        return data

