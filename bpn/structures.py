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

import networkx

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
        - `interaction_graph`: graph containing the gene-gene or gene
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
        - `interaction_graph`: graph containing the gene-gene or gene
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


    def apply_expression_values_to_interaction_graph(self,
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
            interaction_graph,
            annotations_dict,
            annotations_stats,
            links,
            num_links,
            links_outfile,
            **kwargs
        ):
        """Create a new instance.

        :Parameters:
        - `interaction_graph`: graph containing the gene-gene or gene
          product-gene product interactions
        - `annotations_dict`: a dictionary with annotation terms as keys
          and `set`s of genes as values
        - `annotations_stats`: a dictionary containing statistics about
          the annotations
        - `links`: an iterable of pairs of terms to test
        - `num_links`: the number of links contained in `links`
        - `links_outfile`: file for output of link results

        """
        self.interaction_graph = interaction_graph
        self.annotations_dict = annotations_dict
        self.annotations_stats = annotations_stats
        self.links = links
        self.num_links = num_links
        self.links_outfile = links_outfile


class ContextualInputData(BplnInputData):
    """Structure storing all necessary input for Contextual BPLN."""
    def __init__(
            self,
            interaction_graph,
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
        - `interaction_graph`: graph containing the gene-gene or gene
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
            interaction_graph=interaction_graph,
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
            interaction_graph,
            annotations_dict,
            annotations_stats,
            burn_in,
            steps,
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
        - `interaction_graph`: graph containing the gene-gene or gene
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
        super(McmcInputData, self).__init__(
            interaction_graph=interaction_graph,
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
        self.free_parameters = free_parameters
        self.disable_swaps = disable_swaps
        self.transition_ratio = transition_ratio
        self.parameters_outfile = parameters_outfile
        self.transitions_outfile = transitions_outfile
        self.detailed_transitions = detailed_transitions


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
            interaction_graph,
            annotations_dict,
            links_of_interest=None
        ):
        """Create a new instance.

        :Parameters:
        - `interaction_graph`: graph containing the gene-gene or gene
          product-gene product interactions
        - `annotations_dict`: a dictionary with annotation terms as keys
          and `set`s of genes as values
        - `links_of_interest`: a `set` of links in which the user is
          only interested; restricts the lookup keys to this set of
          interactions, potentially significantly reducing the memory
          usage. [NOTE: Each link's terms MUST be sorted alphabetically
          (e.g., `('term1', 'term2')` and NOT `('term2',
          'term1')`!]

        """
        self._interaction_graph = interaction_graph
        self._annotations_dict = annotations_dict
        # We'll use the following variable to cache the number of
        # interactions present, since this is apparently not cached by
        # the NetworkX Graph class.
        self._num_interactions = None
        # We use this dictionary for fast lookup of what interactions
        # are co-annotated by any given pair of annotation terms.
        self._annotations_to_interactions = collections.defaultdict(set)
        self._create_interaction_annotations(links_of_interest)
        self._num_annotation_pairs = None


    def _create_interaction_annotations(self, links_of_interest=None):
        """Convert all the node annotations into pair-wise annotations
        of interactions.

        :Parameters:
        - `links_of_interest`: a `set` of links in which the user is
          only interested; restricts the lookup keys to this set of
          interactions

        """
        total_num_interactions = self.calc_num_interactions()
        broadcast_percent_complete = 0
        for i, edge in enumerate(self._interaction_graph.edges_iter()):
            gene1_annotations = self._annotations_dict.get_item_keys(
                    edge[0])
            gene2_annotations = self._annotations_dict.get_item_keys(
                    edge[1])
            pairwise_combinations = itertools.product(gene1_annotations,
                    gene2_annotations)
            for gene1_annotation, gene2_annotation in \
                    pairwise_combinations:
                # If these are the same term, skip it.
                if gene1_annotation == gene2_annotation:
                    continue
                # We want to preserve alphabetical order of the
                # annotations.
                if gene1_annotation > gene2_annotation:
                    gene1_annotation, gene2_annotation = \
                            gene2_annotation, gene1_annotation
                link = (gene1_annotation, gene2_annotation)
                if links_of_interest is not None:
                    if link not in links_of_interest:
                        continue
                if SUPERDEBUG_MODE:
                    logger.log(SUPERDEBUG, "Adding interactions "
                            "for link %s" % (link,))
                self._annotations_to_interactions[(gene1_annotation,
                    gene2_annotation)].add(edge)

            percent_complete = int(100 * float(i + 1) /
                    total_num_interactions)
            if percent_complete >= (broadcast_percent_complete +
                    BROADCAST_PERCENT):
                broadcast_percent_complete = percent_complete
                logger.info("%d%% of interactions processed." % (
                        percent_complete))


    def get_all_links(self):
        """Returns a list of all the annotation pairs annotating the
        interactions.

        """
        return self._annotations_to_interactions.keys()


    def calc_num_links(self):
        """Returns the number of annotation pairs annotating the
        interactions.

        """
        if self._num_annotation_pairs is None:
            self._num_annotation_pairs = \
                    len(self._annotations_to_interactions)
        return self._num_annotation_pairs


    def calc_num_interactions(self):
        """Returns the total number of interactions."""
        if self._num_interactions is None:
            self._num_interactions = \
                    self._interaction_graph.number_of_edges()
        return self._num_interactions


    def get_interactions_annotated_by(self, annotation1, annotation2):
        """Returns a `set` of all interactions for which one adjacent
        gene is annotated with `annotation1`, and the other adjacent
        gene is annotated with `annotation2`.

        :Parameters:
        - `annotation1`: an annotation term
        - `annotation2`: an annotation term

        """
        if annotation1 > annotation2:
            annotation1, annotation2 = annotation2, annotation1
        interactions = self._annotations_to_interactions[(annotation1,
                annotation2)]
        return interactions


    def get_active_interactions(self, cutoff, greater=True):
        """Returns a `set` of all "active" interactions: those for which
        both incident genes pass a cutoff for differential gene
        expression.

        :Parameters:
        - `cutoff`: a numerical threshold value for determining whether
          a gene is active or not
        - `greater`: if `True`, consider a gene "active" if its
          differential expression value is greater than or equal to the
          `cutoff`; if `False`, consider a gene "active" if its value is
          less than or equal to the `cutoff`.

        """
        active_interactions = set()
        for edge in self._interaction_graph.edges_iter():
            gene1_expr = self._interaction_graph.node[edge[0]]['weight']
            gene2_expr = self._interaction_graph.node[edge[1]]['weight']
            if greater:
                if gene1_expr >= cutoff and gene2_expr >= cutoff:
                    active_interactions.add(edge)
            if not greater:
                if gene1_expr <= cutoff and gene2_expr <= cutoff:
                    active_interactions.add(edge)
        return active_interactions


class ShoveAnnotatedInteractionsGraph(AnnotatedInteractionsGraph):
    def __init__(
            self,
            interaction_graph,
            annotations_dict,
            store
        ):
        """Create a new instance.

        :Parameters:
        - `interaction_graph`: graph containing the gene-gene or gene
          product-gene product interactions
        - `annotations_dict`: a dictionary with annotation terms as keys and
          `set`s of genes as values
        - `store`: a `Shove` instance

        """
        self._interaction_graph = interaction_graph
        self._annotations_dict = annotations_dict
        self._store = store
        self._annotations_to_interactions = store


    def _create_interaction_annotations(self):
        """Convert all the node annotations into pair-wise annotations
        of interactions

        """
        for interaction in self._interaction_graph.edges_iter():
            gene1_annotations = self._annotations_dict[interaction[0]]
            gene2_annotations = self._annotations_dict[interaction[1]]
            annotation_pairs = itertools.product(gene1_annotations,
                    gene2_annotations)
            for annotation1, annotation2 in annotation_pairs:
                if annotation1 == annotation2:
                    continue
                if annotation1 > annotation2:
                    # Swap the annotations so they are in alphabetical
                    # order
                    annotation1, annotation2 = annotation2, annotation1
                # Get the interactions this pair of terms annotates
                annotated_interactions = \
                        self._annotations_to_interactions.get(
                                (annotation1, annotation2), set())
                # Add this interaction
                annotated_interactions.add(interaction)
                # Update the store
                self._annotations_to_interactions[
                        (annotation1, annotation2)] = \
                                annotated_interactions


class AnnotatedInteractionsArray(AnnotatedInteractionsGraph):
    """Similar to `AnnotatedInteractionsGraph`, however, it stores
    links, and their associated interactions, in linear arrays (lists),
    which are accessed by integer indices.

    """
    def __init__(
            self,
            interaction_graph,
            annotations_dict,
            links_of_interest=None
        ):
        """Create a new instance.

        :Parameters:
        - `interaction_graph`: graph containing the gene-gene or gene
          product-gene product interactions
        - `annotations_dict`: a dictionary with annotation terms as keys
          and `set`s of genes as values
        - `links_of_interest`: a `set` of links in which the user is
          only interested; restricts the lookup keys to this set of
          interactions, potentially significantly reducing the memory
          usage. [NOTE: Each link's terms MUST be sorted alphabetically
          (e.g., `('term1', 'term2')` and NOT `('term2',
          'term1')`!]

        """
        AnnotatedInteractionsGraph.__init__(
                self,
                interaction_graph,
                annotations_dict,
                links_of_interest
        )
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
                self._annotations_to_interactions.keys(),
                self._annotations_to_interactions.values()
        )
        # Delete the dictionary mapping since we will not use it
        # hereafter.
        del self._annotations_to_interactions


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


    def get_interactions_annotated_by(self, link_index):
        """Returns a `set` of all interactions for which one adjacent
        gene is annotated with one term in the link and the other
        gene is annotated with the second term in the link.

        :Parameters:
        - `link_index`: the index of the link whose interactions are
          sought

        """
        return self._link_interactions[link_index]


