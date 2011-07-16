#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright (c) 2011 Christopher D. Lasher
#
# This software is released under the MIT License. Please see
# LICENSE.txt for details.


"""Data structures for the BPN programs."""


import collections
import itertools
import random

import bitarray
import networkx
import numpy

from mcmc.defaults import BROADCAST_PERCENT, SUPERDEBUG, SUPERDEBUG_MODE

import logging
logger = logging.getLogger('bpn.structures')


class InteractionGraph(networkx.Graph):
    """A graph which represents interactions between genes (or gene
    products.

    """
    def get_neighbors_of_annotated(self, annotated_genes):
        """Get all the neighbors of the annotated genes that are not also
        among the annotated genes, themselves.

        Returns a `set` of genes.

        :Parameters:
        - `interactions_graph`: graph containing the gene-gene or gene
          product-gene product interactions
        - `annotated_genes`: a set of genes annotated with a term of
          interest whose neighbors to find

        """
        redundant_neighbors = [self[gene] for gene in annotated_genes]
        neighbors = set(itertools.chain.from_iterable(redundant_neighbors))
        # remove neighbors who appear in the annotated genes
        neighbors.difference_update(annotated_genes)
        return neighbors


    def prune_unannotated_genes(self, annotations_dict):
        """Remove genes lacking annotations from the graph.

        :Parameters:
        - `interactions_graph`: graph containing the gene-gene or gene
          product-gene product interactions
        - `annotations_dict`: a dictionary with annotation terms as keys and
          `set`s of genes as values

        """
        # For each gene in the graph, if it doesn't exist in the annotated
        # set, remove it
        nodes_to_remove = []
        for gene in self:
            if not annotations_dict.has_item(gene):
                nodes_to_remove.append(gene)
        self.remove_nodes_from(nodes_to_remove)


    def prune_non_network_genes_from_annotations(self,
            annotations_dict):
        """Remove genes not in the interaction graph from those with
        annotations.

        :Parameters:
        - `annotations_dict`: a dictionary with annotation terms as keys
          and `set`s of genes as values

        """
        genes_to_remove = []
        for gene in annotations_dict.reverse_iterkeys():
            if gene not in self:
                genes_to_remove.append(gene)

        for gene in genes_to_remove:
            annotations_dict.remove_item_from_all_keys(gene)

        annotations_to_remove = []
        # Remove any annotations lacking genes.
        for annotation, genes in annotations_dict.iteritems():
            if not genes:
                annotations_to_remove.append(annotation)

        for annotation in annotations_to_remove:
            del annotations_dict[annotation]


    def apply_expression_values_to_interactions_graph(self,
            expression_values):
        """Apply expression values to the interaction graph as node
        weights.

        NOTE: This method is destructive, and will also remove any nodes
        from the graph which lack an expression value.

        :Parameters:
        - `expression_values`: a dictionary with gene (product) IDs as
          keys and floats as values

        """
        genes_to_remove = []
        genes_attrs = self.node
        for gene in self:
            if gene in expression_values:
                genes_attrs[gene]['weight'] = expression_values[gene]
            else:
                genes_to_remove.append(gene)

        self.remove_nodes_from(genes_to_remove)


class EdgeSwapGraph(InteractionGraph):
    """An interaction graph which can produce a "random" graph from
    itself by an iterative edge-swap technique that preserves the degree
    distribution of the original graph.

    """
    def randomize_by_edge_swaps(self, num_iterations):
        """Randomizes the graph by swapping edges in such a way that
        preserves the degree distribution of the original graph.

        The underlying idea stems from the following. Say we have this
        original formation of edges in the original graph:

            head1   head2
              |       |
              |       |
              |       |
            tail1   tail2

        Then we wish to swap the edges between these four nodes as one
        of the two following possibilities:

            head1   head2       head1---head2
                 \ /
                  X
                 / \
            tail1   tail2       tail1---tail2

        We approach both by following through the first of the two
        possibilities, but before committing to the edge creation, give
        a chance that we flip the nodes `head1` and `tail1`.

        See the following references for the algorithm:

        - F. Viger and M. Latapy, "Efficient and Simple Generation of
          Random Simple Connected Graphs with Prescribed Degree
          Sequence," Computing and Combinatorics, 2005.
        - M. Mihail and N.K. Vishnoi, "On Generating Graphs with
          Prescribed Vertex Degrees for Complex Network Modeling,"
          ARACNE 2002, 2002, pp. 1–11.
        - R. Milo, N. Kashtan, S. Itzkovitz, M.E.J. Newman, and U.
          Alon, "On the uniform generation of random graphs with
          prescribed degree sequences," cond-mat/0312028, Dec. 2003.


        :Parameters:
        - `num_iterations`: the number of iterations for edge swapping
          to perform; this value will be multiplied by the number of
          edges in the graph to get the total number of iterations

        """
        logger.debug("Copying graph.")
        newgraph = self.copy()
        edge_list = newgraph.edges()
        num_edges = len(edge_list)
        total_iterations = num_edges * num_iterations
        logger.debug("Swapping edges over %d events." % total_iterations)

        for i in xrange(total_iterations):
            rand_index1 = int(round(random.random() * (num_edges - 1)))
            rand_index2 = int(round(random.random() * (num_edges - 1)))
            original_edge1 = edge_list[rand_index1]
            original_edge2 = edge_list[rand_index2]
            head1, tail1 = original_edge1
            head2, tail2 = original_edge2

            # Flip a coin to see if we should swap head1 and tail1 for
            # the connections
            if random.random() >= 0.5:
                head1, tail1 = tail1, head1

            # The plan now is to pair head1 with tail2, and head2 with
            # tail1
            #
            # To avoid self-loops in the graph, we have to check that,
            # by pairing head1 with tail2 (respectively, head2 with
            # tail1) that head1 and tail2 are not actually the same
            # node. For example, suppose we have the edges (a, b) and
            # (b, c) to swap.
            #
            #   b
            #  / \
            # a   c
            #
            # We would have new edges (a, c) and (b, b) if we didn't do
            # this check.

            if head1 == tail2 or head2 == tail1:
                continue

            # Trying to avoid multiple edges between same pair of nodes;
            # for example, suppose we had the following
            #
            # a   c
            # |*  |           | original edge pair being looked at
            # | * |
            # |  *|           * existing edge, not in the pair
            # b   d
            #
            # Then we might accidentally create yet another (a, d) edge.
            # Note that this also solves the case of the following,
            # missed by the first check, for edges (a, b) and (a, c)
            #
            #   a
            #  / \
            # b   c
            #
            # These edges already exist.

            if newgraph.has_edge(head1, tail2) or newgraph.has_edge(
                    head2, tail1):
                continue

            # Suceeded checks, perform the swap
            original_edge1_data = newgraph[head1][tail1]
            original_edge2_data = newgraph[head2][tail2]

            newgraph.remove_edges_from((original_edge1, original_edge2))

            new_edge1 = (head1, tail2, original_edge1_data)
            new_edge2 = (head2, tail1, original_edge2_data)
            newgraph.add_edges_from((new_edge1, new_edge2))

            # Now update the entries at the indices randomly selected
            edge_list[rand_index1] = (head1, tail2)
            edge_list[rand_index2] = (head2, tail1)

        assert len(newgraph.edges()) == num_edges
        return newgraph


def get_annotations_stats(annotations_dict):
    """Get annotations statistics from a dictionary of annotations.

    Returns a dictionary with the following keys:
    - `'num_annotation_terms'`: the number of distinct terms that appear
      among the annotations
    - `'num_total_annotations'`: the total number of annotations
      (term-to-gene (product) pairings)
    - `'num_genes'`: the number of genes (or products) with at least one
      annotation num_annotation_terms = len(annotations_dict)

    :Parameters:
    - `annotations_dict`: a dictionary with annotation terms as keys and
      `set`s of genes as values

    """
    num_total_annotations = 0

    for annotated_genes in annotations_dict.itervalues():
        num_total_annotations += len(annotated_genes)

    stats = {
            'num_annotation_terms': len(annotations_dict),
            'num_total_annotations': num_total_annotations,
            'num_genes': len(annotations_dict.reverse_keys())
    }

    return stats


class BplnInputData(object):
    """Structure storing all necessary input for BPLN."""
    def __init__(
            self,
            interactions_graph,
            annotations_dict,
            annotations_stats,
            links,
            num_links,
            links_outfile,
            **kwargs
        ):
        """Create a new instance.

        :Parameters:
        - `interactions_graph`: graph containing the gene-gene or gene
          product-gene product interactions
        - `annotations_dict`: a dictionary with annotation terms as keys
          and `set`s of genes as values
        - `annotations_stats`: a dictionary containing statistics about
          the annotations
        - `links`: an iterable of pairs of terms to test
        - `num_links`: the number of links contained in `links`
        - `links_outfile`: file for output of link results

        """
        self.interactions_graph = interactions_graph
        self.annotations_dict = annotations_dict
        self.annotations_stats = annotations_stats
        self.links = links
        self.num_links = num_links
        self.links_outfile = links_outfile


class ContextualInputData(BplnInputData):
    """Structure storing all necessary input for Contextual BPLN."""
    def __init__(
            self,
            interactions_graph,
            annotations_dict,
            annotations_stats,
            links,
            num_links,
            links_outfile,
            num_permutations,
            edge_swaps,
            estimate,
            score_correction,
            **kwargs
        ):
        """Create a new instance.

        :Parameters:
        - `interactions_graph`: graph containing the gene-gene or gene
          product-gene product interactions
        - `annotations_dict`: a dictionary with annotation terms as keys
          and `set`s of genes as values
        - `annotations_stats`: a dictionary containing statistics about
          the annotations
        - `links`: an iterable of pairs of terms to test
        - `num_links`: the number of links contained in `links`
        - `links_outfile`: file for output of link results

        """
        super(ContextualInputData, self).__init__(
            interactions_graph=interactions_graph,
            annotations_dict=annotations_dict,
            annotations_stats=annotations_stats,
            links=links,
            num_links=num_links,
            links_outfile=links_outfile,
            **kwargs
        )
        self.num_permutations = num_permutations
        self.edge_swaps = edge_swaps
        self.estimate = estimate
        self.score_correction = score_correction


class McmcInputData(BplnInputData):
    def __init__(
            self,
            interactions_graph,
            annotations_dict,
            annotations_stats,
            burn_in,
            steps,
            activity_threshold,
            link_false_pos,
            link_false_neg,
            link_prior,
            term_false_pos,
            term_false_neg,
            term_prior,
            seed_terms,
            seed_links,
            stringent_coannotations,
            fixed_distributions,
            free_parameters,
            disable_swaps,
            terms_based,
            intraterms,
            independent_terms,
            genes_based,
            transition_ratio,
            terms_outfile,
            links_outfile,
            parameters_outfile,
            transitions_outfile,
            detailed_transitions,
            record_frequencies,
            frequencies_outfile,
            **kwargs
        ):
        """Create a new instance.

        :Parameters:
        - `interactions_graph`: graph containing the gene-gene or gene
          product-gene product interactions
        - `annotations_dict`: a dictionary with annotation terms as keys
          and `set`s of genes as values
        - `annotations_stats`: a dictionary containing statistics about
          the annotations
        - `burn_in`: the number of steps to perform during the burn-in
          period
        - `steps`: the number of steps to record
        - `activity_threshold`: the threshold at which a gene is
          declared active
        - `link_false_pos`: the starting false positive rate
        - `link_false_neg`: the starting false negative rate
        - `link_prior`: the starting probability of adding a link
        - `term_false_pos`: the false-positive rate for terms, the
          portion of genes which were included, but shouldn't have been
        - `term_false_neg`: the false-negative rate for terms, the
          portion of genes which weren't included, but should have been
        - `term_prior`: the starting probability of adding a term
        - `seed_terms`: an iterable of annotation terms to be used as a
          seed when initializing the Markov chain
        - `seed_links`: an iterable of annotation pairs to use as a seed
          when initializing the Markov chain
        - `stringent_coannotations`: `True` if stringent definition of
          co-annotation should be used
        - `fixed_distributions`: `True` if fixed distributions for link
          (and term) priors should be used, `False` if dynamic
          distributions should be used
        - `free_parameters`: `True` if parameters are free take a random
          value from their distribution, `False` if they may only take
          adjoining values
        - `disable_swaps`: `True` if swap transitions are to be
          disabled, `False` otherwise.
        - `terms_based`: `True` if terms-based model is to be used,
          `False` otherwise
        - `intraterms`: consider also intraterm interactions
        - `independent_terms`: `True` if selectable-terms based model is
          to be used, `False` otherwise
        - `genes_based`: `True` if overlaps of terms should be
          considered through a genes-based model, `False` otherwise
        - `transition_ratio`: a `float` indicating the ratio of link
          transitions to parameter transitions
        - `terms_outfile`: file for output of terms results
        - `links_outfile`: file for output of link results
        - `parameters_outfile`: file for output of parameters results
        - `transitions_outfile`: file for output of transitions data
        - `detailed_transitions`: `True` if detailed output of the
          transitions is desired, `False` otherwise
        - `record_frequencies`: `True` if frequency of the states should
          be recorded
        - `frequencies_outfile`: the file with which to write frequency
            information to

        """
        super(McmcInputData, self).__init__(
            interactions_graph=interactions_graph,
            annotations_dict=annotations_dict,
            annotations_stats=annotations_stats,
            links=None,
            num_links=None,
            links_outfile=links_outfile,
            **kwargs
        )
        # User-defined subset of links to test is inappropriate for MCMC
        # BPLN, so we remove those.
        del self.links
        del self.num_links
        self.burn_in = burn_in
        self.steps = steps
        self.activity_threshold = activity_threshold
        self.link_false_pos = link_false_pos
        self.link_false_neg = link_false_neg
        self.link_prior = link_prior
        self.term_false_pos = term_false_pos
        self.term_false_neg = term_false_neg
        self.term_prior = term_prior
        self.seed_terms = seed_terms
        self.seed_links = seed_links
        self.stringent_coannotations = stringent_coannotations
        self.fixed_distributions = fixed_distributions
        self.free_parameters = free_parameters
        self.disable_swaps = disable_swaps
        self.intraterms = intraterms
        self.terms_based = terms_based
        self.genes_based = genes_based
        self.independent_terms = independent_terms
        self.transition_ratio = transition_ratio
        self.terms_outfile = terms_outfile
        self.parameters_outfile = parameters_outfile
        self.transitions_outfile = transitions_outfile
        self.detailed_transitions = detailed_transitions
        self.record_frequencies = record_frequencies
        self.frequencies_outfile = frequencies_outfile


class SaInputData(BplnInputData):
    def __init__(
            self,
            interactions_graph,
            annotations_dict,
            annotations_stats,
            steps,
            temperature,
            end_temperature,
            activity_threshold,
            free_parameters,
            disable_swaps,
            transition_ratio,
            links_outfile,
            parameters_outfile,
            transitions_outfile,
            detailed_transitions,
            **kwargs
        ):
        """Create a new instance.

        :Parameters:
        - `interactions_graph`: graph containing the gene-gene or gene
          product-gene product interactions
        - `annotations_dict`: a dictionary with annotation terms as keys
          and `set`s of genes as values
        - `annotations_stats`: a dictionary containing statistics about
          the annotations
        - `steps`: the number of steps to anneal
        - `temperature`: the starting temperature to anneal from
        - `end_temperature`: the temperature to anneal to
        - `activity_threshold`: the threshold at which a gene is
          declared active
        - `free_parameters`: `True` if parameters are free take a random
          value from their distribution, `False` if they may only take
          adjoining values
        - `disable_swaps`: `True` if swap transitions are to be
          disabled, `False` otherwise.
        - `transition_ratio`: a `float` indicating the ratio of link
          transitions to parameter transitions
        - `links_outfile`: file for output of link results
        - `parameters_outfile`: file for output of parameters results
        - `transitions_outfile`: file for output of transitions data
        - `detailed_transitions`: `True` if detailed output of the
          transitions is desired, `False` otherwise

        """
        super(SaInputData, self).__init__(
            interactions_graph=interactions_graph,
            annotations_dict=annotations_dict,
            annotations_stats=annotations_stats,
            links=None,
            num_links=None,
            links_outfile=links_outfile,
            **kwargs
        )
        # User-defined subset of links to test is inappropriate for SA
        # BPLN, so we remove those.
        del self.links
        del self.num_links
        self.steps = steps
        self.temperature = temperature
        self.end_temperature = end_temperature
        self.activity_threshold = activity_threshold
        self.free_parameters = free_parameters
        self.disable_swaps = disable_swaps
        self.transition_ratio = transition_ratio
        self.parameters_outfile = parameters_outfile
        self.transitions_outfile = transitions_outfile
        self.detailed_transitions = detailed_transitions


class Symmetrical2dArray(numpy.ndarray):
    """A symmetrical 2-dimensional NumPy array."""
    def __init__(self, shape, dtype=float, buffer=None, offset=0,
            strides=None, order=None):
        if len(shape) != 2:
            raise ValueError("shape must be two integers")
        numpy.ndarray.__init__(self, shape, dtype=float,
                buffer=None, offset=0, strides=None, order=None)


    def __setitem__(self, (i, j), value):
        numpy.ndarray.__setitem__(self, (i, j), value)
        numpy.ndarray.__setitem__(self, (j, i), value)


def symzeros(length, dtype=float, order='C'):
    """Constructs a SymNDArray with all elements set to 0.

    Similar to numpy.zeros

    :Parameters:
    - `length`: the length of one dimension of the array (i.e., number
      of rows)
    - `dtype`: Desired data-type for the array
    - `order`: either C ('C') or Fortran ('F') ordering

    """
    shape = (length, length)
    z = numpy.zeros(shape, dtype=dtype, order=order)
    a = Symmetrical2dArray(shape, dtype=z.dtype, buffer=z,
            order=order)
    return a


class AnnotatedInteractionsGraph(object):
    """A class that provides access to a mapping from process links
    (pairs of annotations) to the interactions which they co-annotate.

    A co-annotation is defined where, for two genes incident on an
    interaction edge, the first gene is annotated with one of the two
    processes, and the second gene is annotated with the other process.

    This class also provides information such as the number of gene-gene
    interactions, and which of those interactions are considered
    "active" according to a threshold.

    """
    def __init__(
            self,
            interactions_graph,
            annotations_dict,
            links_of_interest=None,
            stringent_coannotations=True
        ):
        """Create a new instance.

        :Parameters:
        - `interactions_graph`: graph containing the gene-gene or gene
          product-gene product interactions
        - `annotations_dict`: a dictionary with annotation terms as keys
          and `set`s of genes as values
        - `links_of_interest`: a `set` of links in which the user is
          only interested; restricts the lookup keys to this set of
          interactions, potentially significantly reducing the memory
          usage. [NOTE: Each link's terms MUST be sorted alphabetically
          (e.g., `('term1', 'term2')` and NOT `('term2',
          'term1')`!]
        - `stringent_coannotations`: if `True` uses more stringent rules
          about whether an interaction is co-annotated (both genes may
          not be annotated by both terms) [default: `True`]

        """
        self._interactions_graph = interactions_graph
        self._annotations_dict = annotations_dict
        self.stringent_coannotations = stringent_coannotations
        # This will contain the names of of all the terms which
        # annotate the genes.
        self._annotation_terms = set()
        # We'll use the following variables to cache the number of genes
        # and interactions present, since this is apparently not cached
        # by the NetworkX Graph class.
        self._num_genes = None
        self._create_interaction_annotations(links_of_interest)
        self._num_terms = None
        self._num_annotation_pairs = None


    def _post_process_structures(self):
        """Performs any necessary post-processing of data structures
        created in `_create_interaction_annotations()`

        """
        logger.info("Post-processing data structures.")
        # A problem with collections.defaultdict is that it creates a
        # new entry during a key lookup if that key doesn't exist; we'll
        # convert our structures back to regular Python dictionaries
        # before letting the user access them.
        self._coannotations_to_interactions = dict(
                self._coannotations_to_interactions)
        self._intraterm_interactions = dict(
                self._intraterm_interactions)


    def _create_interaction_annotations(self, links_of_interest=None):
        """Convert all the node annotations into pair-wise annotations
        of interactions.

        :Parameters:
        - `links_of_interest`: a `set` of links in which the user is
          only interested; restricts the lookup keys to this set of
          interactions

        """
        # We use this dictionary for fast lookup of what interactions
        # are co-annotated by any given pair of annotation terms.
        self._coannotations_to_interactions = collections.defaultdict(set)
        # We use a separate dictionary for interactions co-annotated by
        # the same term.
        self._intraterm_interactions = collections.defaultdict(set)
        # We want to move away from actual interactions to indices of
        # interactions, so we'll use the following lists to help us with
        # this.
        self._interactions = self._interactions_graph.edges()
        self._coannotated_indices = []
        self._intraterm_indices = []
        self._coannotated_and_intraterm_indices = []
        total_num_interactions = len(self._interactions)
        broadcast_percent_complete = 0
        for i, edge in enumerate(self._interactions):
            gene1_annotations = self._annotations_dict.get_item_keys(
                    edge[0])
            self._annotation_terms.update(gene1_annotations)
            gene2_annotations = self._annotations_dict.get_item_keys(
                    edge[1])
            self._annotation_terms.update(gene2_annotations)
            pairwise_combinations = itertools.product(gene1_annotations,
                    gene2_annotations)
            coannotated = False
            intraterm = False
            for gene1_annotation, gene2_annotation in (
                    pairwise_combinations):
                # If these are the same term, add them to the intraterm
                # interactions.
                if gene1_annotation == gene2_annotation:
                    self._intraterm_interactions[gene1_annotation].add(
                            i)
                    intraterm = True
                else:
                    if self.stringent_coannotations:
                        # Avoid adding co-annotation by this pair when
                        # both genes are annotated by both terms. In
                        # other words, the interaction is co-annotated
                        # only if at least one of the terms annotates
                        # only one of the two genes.
                        if ((gene1_annotation in gene2_annotations)
                                and
                                (gene2_annotation in gene1_annotations)):
                            continue
                    # We want to preserve alphabetical order of the
                    # annotations.
                    if gene1_annotation > gene2_annotation:
                        gene1_annotation, gene2_annotation = (
                                gene2_annotation, gene1_annotation)
                    link = (gene1_annotation, gene2_annotation)
                    if links_of_interest is not None:
                        if link not in links_of_interest:
                            continue
                    if SUPERDEBUG_MODE:
                        logger.log(SUPERDEBUG, "Adding interaction "
                                "for link %s" % (link,))
                    self._coannotations_to_interactions[
                            (gene1_annotation, gene2_annotation)].add(i)
                    coannotated = True

            if coannotated:
                self._coannotated_indices.append(i)
            if intraterm:
                self._intraterm_indices.append(i)
            if coannotated or intraterm:
                self._coannotated_and_intraterm_indices.append(i)

            percent_complete = int(100 * float(i + 1) /
                    total_num_interactions)
            if percent_complete >= (broadcast_percent_complete +
                    BROADCAST_PERCENT):
                broadcast_percent_complete = percent_complete
                logger.info("%d%% of interactions processed." % (
                        percent_complete))

        self._post_process_structures()


    def get_all_links(self):
        """Returns a list of all the annotation pairs annotating the
        interactions.

        """
        return self._coannotations_to_interactions.keys()


    def calc_num_terms(self):
        """Returns the number of terms annotating the interactions."""
        if self._num_terms is None:
            self._num_terms = len(self._annotation_terms)
        return self._num_terms


    def calc_num_links(self):
        """Returns the number of annotation pairs annotating the
        interactions.

        """
        if self._num_annotation_pairs is None:
            self._num_annotation_pairs = len(
                    self._coannotations_to_interactions)
        return self._num_annotation_pairs


    def calc_num_interactions(self):
        """Returns the total number of interactions.

        NOTE: This number may be different than the number of
        interactions co-annotated by links provided in
        ``selected_links`` (i.e., it may be greater), as it represents
        the number of interactions in the original gene-gene interaction
        network passed in as ``interactions_graph`` during
        initialization.

        """
        return len(self._interactions)


    def calc_num_coannotated_interactions(self):
        """Returns the number of co-annotated interactions."""
        return len(self._coannotated_indices)


    def calc_num_intraterm_interactions(self):
        """Returns the number of intraterm interactions."""
        return len(self._intraterm_indices)


    def calc_num_coannotated_and_intraterm_interactions(self):
        """Returns the number of interactions either co-annotated or
        intraterm.

        """
        return len(self._coannotated_and_intraterm_indices)


    def calc_num_genes(self):
        """Returns the total number of genes.

        NOTE: This number may be different than the number of
        genes annotated by one or more terms in the links provided in
        ``selected_links`` (i.e., it may be greater), as it represents
        the number of genes in the original gene-gene interaction
        network passed in as ``interactions_graph`` during
        initialization.

        """
        if self._num_genes is None:
            self._num_genes = (
                    self._interactions_graph.number_of_nodes())
        return self._num_genes


    def get_annotated_genes(self, term):
        """Returns the set of genes annotated by a term."""
        return self._annotations_dict[term]


    def get_interaction(self, interaction_index):
        """Returns the genes involved in an interaction at the given
        index.

        """
        return self._interactions[interaction_index]


    def get_coannotated_interactions(self, term1, term2):
        """Returns a list of indices of interactions for which one
        gene is annotated with `term1`, and the other gene is annotated
        with `term2`.

        :Parameters:
        - `term1`: an annotation term
        - `term2`: an annotation term

        """
        if term1 > term2:
            term1, term2 = term2, term1
        elif term1 == term2:
            raise ValueError("Terms may not be the same.")
        interactions = self._coannotations_to_interactions[(term1,
                term2)]
        return interactions


    def get_intraterm_interactions(self, term):
        """Returns a `set` of all interactions for which the `term`
        annotates both interacting genes.

        :Parameters:
        - `term`: an annotation term

        """
        return self._intraterm_indices[term]


    def get_active_genes(self, cutoff, greater=True):
        """Returns a `set` of all "active" genes: those for which
        pass a cutoff for differential gene expression.

        :Parameters:
        - `cutoff`: a numerical threshold value for determining whether
          a gene is active or not
        - `greater`: if `True`, consider a gene "active" if its
          differential expression value is greater than or equal to the
          `cutoff`; if `False`, consider a gene "active" if its value is
          less than or equal to the `cutoff`.

        """
        if greater:
            active_genes = set(gene for gene, vals in
                    self._interactions_graph.node.items() if
                    vals['weight'] >= cutoff
            )
        else:
            active_genes = set(gene for gene, vals in
                    self._interactions_graph.node.items() if
                    vals['weight'] <= cutoff
            )
        return active_genes


    def _get_active_interactions(self, indices, cutoff, greater=True):
        """Helper function for determining active interactions. Returns
        a list of all "active" interactions among those whose indices
        are provided.

        :Parameters:
        - `indices`: indices of interactions to screen
        - `cutoff`: a numerical threshold value for determining whether
          a gene is active or not
        - `greater`: if `True`, consider a gene "active" if its
          differential expression value is greater than or equal to the
          `cutoff`; if `False`, consider a gene "active" if its value is
          less than or equal to the `cutoff`.

        """
        active_interactions = []
        for index in indices:
            edge = self._interactions[index]
            gene1_expr = self._interactions_graph.node[edge[0]]['weight']
            gene2_expr = self._interactions_graph.node[edge[1]]['weight']
            if greater:
                if gene1_expr >= cutoff and gene2_expr >= cutoff:
                    active_interactions.append(index)
            if not greater:
                if gene1_expr <= cutoff and gene2_expr <= cutoff:
                    active_interactions.append(index)
        return active_interactions


    def get_active_interactions(self, cutoff, greater=True):
        """Returns a list of indices of all "active" interactions: those
        for which both incident genes pass a cutoff for differential
        gene expression.

        :Parameters:
        - `cutoff`: a numerical threshold value for determining whether
          a gene is active or not
        - `greater`: if `True`, consider a gene "active" if its
          differential expression value is greater than or equal to the
          `cutoff`; if `False`, consider a gene "active" if its value is
          less than or equal to the `cutoff`.

        """
        return self._get_active_interactions(
                range(len(self._interactions)),
                cutoff,
                greater
        )


    def get_active_coannotated_interactions(self, cutoff, greater=True):
        """Returns a a list of indices "active" co-annotated
        interactions: those for which both incident genes pass a cutoff
        for differential gene expression.

        :Parameters:
        - `cutoff`: a numerical threshold value for determining whether
          a gene is active or not
        - `greater`: if `True`, consider a gene "active" if its
          differential expression value is greater than or equal to the
          `cutoff`; if `False`, consider a gene "active" if its value is
          less than or equal to the `cutoff`.

        """
        return self._get_active_interactions(
                self._coannotated_indices,
                cutoff,
                greater
        )


    def get_active_intraterm_interactions(self, cutoff, greater=True):
        """Returns a a list of indices "active" intraterm
        interactions: those for which both incident genes pass a cutoff
        for differential gene expression.

        :Parameters:
        - `cutoff`: a numerical threshold value for determining whether
          a gene is active or not
        - `greater`: if `True`, consider a gene "active" if its
          differential expression value is greater than or equal to the
          `cutoff`; if `False`, consider a gene "active" if its value is
          less than or equal to the `cutoff`.

        """
        return self._get_active_interactions(
                self._intraterm_indices,
                cutoff,
                greater
        )


    def get_active_coannotated_and_intraterm_interactions(self,
            cutoff, greater=True):
        """Returns a a list of indices "active" co-annotated or
        intraterm interactions: those for which both incident genes pass
        a cutoff for differential gene expression.

        :Parameters:
        - `cutoff`: a numerical threshold value for determining whether
          a gene is active or not
        - `greater`: if `True`, consider a gene "active" if its
          differential expression value is greater than or equal to the
          `cutoff`; if `False`, consider a gene "active" if its value is
          less than or equal to the `cutoff`.

        """
        return self._get_active_interactions(
                self._coannotated_and_intraterm_indices,
                cutoff,
                greater
        )


class AnnotatedInteractionsArray(AnnotatedInteractionsGraph):
    """Similar to `AnnotatedInteractionsGraph`, however, it stores
    links, and their associated interactions, in linear arrays (lists),
    which are accessed by integer indices.

    """
    def _post_process_structures(self):
        logger.info("Converting to more efficient data structures.")
        # Set two new attributes below, one being the list of all links,
        # and the other being the list of each link's corresponding
        # interactions, so that we can access them in linear time using
        # indices.
        #
        # Note that we rely on a property of Python dictionaries that,
        # so long as they are not modified between accesses, the lists
        # returned by dict.keys() and dict.values() correspond. See
        # http://docs.python.org/library/stdtypes.html#dict.items
        self._links, self._link_interactions = (
                self._coannotations_to_interactions.keys(),
                self._coannotations_to_interactions.values()
        )
        self._links_indices = dict((l, i) for (i, l) in
                enumerate(self._links))
        # Delete the dictionary mapping since we will not use it
        # hereafter.
        del self._coannotations_to_interactions

        self._intraterm_interactions = dict(
                self._intraterm_interactions)
        # TODO: convert intraterm interactions?


    def get_all_links(self):
        """Returns a list of all the annotation pairs annotating the
        interactions.

        """
        return self._links


    def calc_num_links(self):
        """Returns the number of annotation pairs annotating the
        interactions.

        """
        if self._num_annotation_pairs is None:
            self._num_annotation_pairs = \
                    len(self._links)
        return self._num_annotation_pairs


    def get_termed_link(self, link_index):
        """Returns a link with the actual annotation terms.

        :Parameters:
        - `link_index`: the index of the link of interest

        """
        return self._links[link_index]


    def get_link_index(self, term1, term2):
        """Returns the two-dimensional index of the link specified by
        the two terms.

        :Parameters:
        - `term1`: an annotation term
        - `term2`: an annotation term

        """
        if term1 == term2:
            raise ValueError("Terms can not be the same.")
        elif term1 > term2:
            term1, term2 = term2, term1
        # TODO: Should we check to see if the pair of terms actually
        # co-annotate any interactions?
        return self._links_indices[(term1, term2)]


    def get_coannotated_interactions(self, link_index):
        """Returns a list of indices of interactions for which one gene
        is annotated with the first term, and the other gene is
        annotated with the second term, for the pair of terms
        represented by the link index.

        :Parameters:
        - `link_index`: the index of the link whose interactions are
          sought

        """
        return self._link_interactions[link_index]


    # TODO: Add support for intraterm interactions


class AnnotatedInteractions2dArray(AnnotatedInteractionsGraph):
    def _map_terms_to_indices(self):
        self._annotation_terms = sorted(self._annotation_terms)
        self._terms_to_indices = {}
        for i, term in enumerate(self._annotation_terms):
            self._terms_to_indices[term] = i


    def _post_process_structures(self):
        logger.info("Converting to more efficient data structures.")
        self._map_terms_to_indices()
        # Get linearized lists of the links and their interactions.
        named_links, self._link_interactions = (
                self._coannotations_to_interactions.keys(),
                self._coannotations_to_interactions.values()
        )

        num_terms = len(self._annotation_terms)

        # This 2d symmetric array will map a pair of link indices to
        # their respective interactions, contained in the list
        # self._link_interactions.
        #
        # This array will also help serve as a way of confirming
        # whether or not a link is valid, as a non-zero value for that
        # position in the 2d array indicates the link co-annotates one
        # or more interactions.
        self._links_to_interactions = symzeros(num_terms, int)

        # The idea here is that, by the end of this next loop, we will
        # mark the 2d array self._links_to_interactions with the index
        # at which we can find the interactions co-annotated by the
        # link.
        for i, link in enumerate(named_links):
            term1_index = self._terms_to_indices[link[0]]
            term2_index = self._terms_to_indices[link[1]]
            # Note that we're going to add 1 to the index; see
            # explanation below.
            self._links_to_interactions[term1_index, term2_index] = (i +
                    1)

        # Now we will compensate for having a 1-based index into
        # self._link_interactions by padding it with None at index 0.
        # This will have the added benefit that any "invalid link" of
        # self._links_to_interactions will point to None as its
        # interactions, allowing us to identify it as an invalid link.
        self._link_interactions.insert(0, None)

        # Delete the dictionary mapping since we will not use it
        # hereafter.
        del self._coannotations_to_interactions

        # We follow a similar pattern for the intraterm interactions.
        self._intraterm_indices = numpy.zeros(num_terms, int)
        named_terms, self._intraterm_interactions = (
                self._intraterm_interactions.keys(),
                self._intraterm_interactions.values()
        )
        self._intraterm_interactions.insert(0, None)
        # For each term that has intraterm interactions, we'll denote
        # the index in the _intraterm_interactions where we can find its
        # intraterm interactions.
        for i, term in enumerate(named_terms):
            term_index = self.get_term_index(term)
            # Note that we increment the index by 1 because the index 0
            # in _intraterm_interactions is None and we reserve that for
            # indicating a term has no intraterm interactions.
            self._intraterm_indices[term_index] = i + 1

        # The idea here is that we want to create a linear data
        # structure (list), where we index into this structure on some
        # annotation term's index. The value at that index should be a
        # list of indices. These indices will represent the genes
        # annotated by the particular term represented by the term
        # index.
        #
        # _genes will be our linearized list of genes, so we can
        # get back to the gene names if necessary.
        self._genes = self._annotations_dict.reverse_keys()
        assert len(self._genes) == self.calc_num_genes()
        # _genes_to_indices will represent a way to get a gene's index
        # by the gene name.
        self._genes_to_indices = dict((g, i) for (i, g) in
                enumerate(self._genes))
        # _terms_genes will be our list data structure by which we'll
        # get the gene indices for a term represented by a particular
        # index.
        self._terms_genes = []
        for term in self._annotation_terms:
            annotated_genes = [self._genes_to_indices[gene] for gene in
                    self._annotations_dict[term]]
            self._terms_genes.append(annotated_genes)


    def get_gene_index(self, gene):
        """Returns the index of a gene.

        :Parameters:
        - `gene`: a gene name

        """
        return self._genes_to_indices[gene]


    def get_gene_name(self, gene_index):
        """Returns the name of a gene at a particular index.

        :Parameters:
        - `gene_index`: index of the gene of interest

        """
        return self._genes[gene_index]


    def get_term_index(self, term):
        """Returns the index of a single named term.

        :Parameters:
        - `term`: an annotation term

        """
        return self._terms_to_indices[term]


    def get_term_from_index(self, term_index):
        """Returns the name of a term at a particular index.

        :Parameters:
        - `term_index`: index of the term of interest

        """
        return self._annotation_terms[term_index]


    def get_termed_link(self, link_index):
        """Returns a link with the actual annotation terms instead of
        term indices.

        :Parameters:
        - `link_index`: the two-dimensional index of the link of
          interest

        """
        termed_link = (self.get_term_from_index(link_index[0]),
                self.get_term_from_index(link_index[1]))
        return termed_link


    def get_link_index(self, term1, term2):
        """Returns the two-dimensional index of the link specified by
        the two terms.

        :Parameters:
        - `term1`: an annotation term
        - `term2`: an annotation term

        """
        if term1 == term2:
            raise ValueError("Terms can not be the same.")
        elif term1 > term2:
            term1, term2 = term2, term1
        # TODO: Should we check to see if the pair of terms actually
        # co-annotate any interactions?
        return (self.get_term_index(term1), self.get_term_index(term2))


    def get_annotated_genes(self, term_index):
        """Returns a `list` of gene indices annotated by a term.

        :Parameters:
        - `term_index`: index of the term of interest

        """
        return self._terms_genes[term_index]


    def get_intraterm_interactions(self, term):
        """Returns a `set` of all interactions for which the term
        annotates both interacting genes.

        Returns `None` if no intraterm interactions exist for the term.

        :Parameters:
        - `term`: an annotation term

        """
        index = self._intraterm_indices[term]
        intraterm_interactions = self._intraterm_interactions[index]
        return intraterm_interactions


    def get_active_genes(self, cutoff, greater=True):
        """Returns a list of indices of all "active" genes: those for
        which pass a cutoff for differential gene expression.

        :Parameters:
        - `cutoff`: a numerical threshold value for determining whether
          a gene is active or not
        - `greater`: if `True`, consider a gene "active" if its
          differential expression value is greater than or equal to the
          `cutoff`; if `False`, consider a gene "active" if its value is
          less than or equal to the `cutoff`.

        """
        if greater:
            active_genes_indices = [self._genes_to_indices[gene] for
                    gene, vals in self._interactions_graph.node.items()
                    if vals['weight'] >= cutoff
            ]
        else:
            active_genes_indices = [self._genes_to_indices[gene] for
                    gene, vals in self._interactions_graph.node.items()
                    if vals['weight'] <= cutoff
            ]
        return active_genes_indices


    def get_coannotated_interactions(self, link_index):
        """Returns a list of indices of interactions for which one gene
        is annotated with the first term, and the other gene is
        annotated with the second term, for the pair of terms
        represented by the link index.

        Returns ``None`` if the terms co-annotate no interactions.

        :Parameters:
        - `link_index`: the two-dimensional index of the link whose
          interactions are sought

        """
        interactions_index = self._links_to_interactions[link_index]
        return self._link_interactions[interactions_index]


    def get_all_links(self):
        """Returns a list of all the annotation pairs annotating the
        interactions.

        Note that this only returns valid links (pairs of terms which
        co-annotate one or more interactions), rather than all pairwise
        combinations of annotation terms.

        """
        # Since the matrix is symmetrical, using nonzero() would give
        # two indices per each valid link; we only need one of those, so
        # we first get the upper-triangle of the matrix, and then return
        # the nonzero of that, zipped so that the indices, returned by
        # nonzero as (X, Y), are returned instead as (x_1, y_1), (x_2,
        # y_2), etc.
        valid_links = zip(
                *numpy.triu(self._links_to_interactions).nonzero())
        return valid_links


    def calc_num_links(self):
        if self._num_annotation_pairs is None:
            self._num_annotation_pairs = len(
                    numpy.triu(self._links_to_interactions).nonzero()[0])
        return self._num_annotation_pairs


# The following class is derived from aix's answer on Stack Overflow.
# See http://stackoverflow.com/questions/6694835/efficient-serialization-of-numpy-boolean-arrays
class BoolBitArray(object):
    """Class used for representing a boolean `numpy.ndarray` object as a
    bit array.

    This class is intended for serialization of NumPy arrays, used to
    store link selections in the links state classes.

    """
    def __init__(self, array):
        ba = bitarray.bitarray()
        ba.pack(array.tostring())
        self.arr = ba.tostring()
        self.shape = array.shape
        self.size = array.size
        self.hash_value = hash(self.arr)


    def to_array(self):
        """Converts the bit array back into a `numpy.ndarray`.

        Returns a boolean NumPy array of the same content and shape as
        the original.

        """
        ba = bitarray.bitarray()
        ba.fromstring(self.arr)
        ret = np.fromstring(ba.unpack(), dtype=np.bool)[:self.size]
        return ret.reshape(self.shape)


    def __cmp__(self, other):
        return cmp(self.arr, other.arr)


    def __hash__(self):
        return self.hash_value


