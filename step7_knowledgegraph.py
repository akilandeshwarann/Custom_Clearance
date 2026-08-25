"""
step7_knowledgegraph.py

Builds an HTS Knowledge Graph from hts_chunks.json using NetworkX.

The graph models the Harmonized Tariff Schedule as a typed directed graph:

  Nodes  — one per HTS code chunk (keyed by htsno).
           Attributes: description, chapter, indent, hierarchy, general,
                       special, other, additional_duties, chunk_id, text.

  Edges  — two types:
    IS_CHILD_OF  (directed, child -> parent)
        Derived from the indent hierarchy. Every chunk with indent > 0
        is linked to its nearest ancestor (the closest chunk whose
        indent is strictly lower and whose htsno shares the same
        leading prefix segment).

    SAME_CHAPTER (undirected, stored as two directed half-edges)
        Every pair of nodes sharing the same chapter AND the same
        immediate parent (same-level siblings under one parent) are
        linked. This enables sibling-set queries without a full chapter
        scan.

Persistence:
    indexes/hts_knowledge_graph.graphml   (portable XML, importable by Gephi)
    indexes/hts_knowledge_graph.gpickle   (fast binary reload in Python)

Usage
-----
  # Build graph from hts_chunks.json:
  python step7_knowledgegraph.py build

  # Print graph statistics:
  python step7_knowledgegraph.py stats

  # Show ancestor chain for a given HTS code:
  python step7_knowledgegraph.py ancestors 0101.21.00.10

  # Show direct children of an HTS code:
  python step7_knowledgegraph.py children 0101.21.00

  # List all HTS codes in a chapter:
  python step7_knowledgegraph.py chapter 01

  # Query by duty rate pattern:
  python step7_knowledgegraph.py duty "Free"

  # Enrich retriever output with graph context (demo):
  python step7_knowledgegraph.py enrich "purebred breeding horses"
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# NetworkX is a soft dependency; provide a clear error if missing.
try:
    import networkx as nx
except ImportError:
    print(
        "Error: 'networkx' is required for step7_knowledgegraph.py.\n"
        "Install it with:  pip install networkx",
        file=sys.stderr,
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent

DEFAULT_CHUNKS_PATH = "hts_chunks.json"
DEFAULT_GRAPHML_PATH = "indexes/hts_knowledge_graph.graphml"
DEFAULT_GPICKLE_PATH = "indexes/hts_knowledge_graph.gpickle"

_SEP = "=" * 70
_SEP_THIN = "-" * 70

# Edge relation labels
REL_CHILD = "IS_CHILD_OF"
REL_SIBLING = "SAME_CHAPTER_SIBLING"


# ---------------------------------------------------------------------------
# Helper: parse structured key-value lines from chunk text
# ---------------------------------------------------------------------------


def _extract_field(text: str, prefix: str) -> Optional[str]:
    """Extract the value after 'prefix: ' from a multiline chunk text block."""
    for line in text.splitlines():
        if line.startswith(prefix + ":"):
            val = line.split(":", 1)[1].strip()
            return val if val and val != "N/A" else None
    return None


# ---------------------------------------------------------------------------
# HTSKnowledgeGraph
# ---------------------------------------------------------------------------


class HTSKnowledgeGraph:
    """
    Knowledge graph over the Harmonized Tariff Schedule.

    The graph is a directed NetworkX DiGraph where nodes are HTS codes and
    edges represent hierarchical (IS_CHILD_OF) and sibling (SAME_CHAPTER_SIBLING)
    relationships derived from the indent / chapter metadata in each chunk.
    """

    def __init__(self):
        self.graph: nx.DiGraph = nx.DiGraph()
        self._built = False

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build_from_chunks(self, chunks_path: str = DEFAULT_CHUNKS_PATH) -> "HTSKnowledgeGraph":
        """
        Load hts_chunks.json and construct the knowledge graph.

        Strategy
        --------
        1. Add one node per chunk, keyed by htsno, with all metadata as
           node attributes.
        2. Walk chunks in file order (which preserves the original HTS
           document order, i.e. hierarchically sorted). Maintain a stack
           of (indent, htsno) pairs to detect the immediate parent of each
           new record: the parent is the deepest stack entry whose indent
           is strictly less than the current record's indent.
        3. For sibling edges: group nodes by (chapter, parent_htsno) and
           add undirected-style edges between every pair in each group.
        """
        p = Path(chunks_path)
        if not p.exists():
            raise FileNotFoundError(
                f"Chunks file not found: {p}. Run step2_chunk.py first."
            )

        print(f"Loading chunks from '{p}'...")
        with open(p, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        print(f"Loaded {len(chunks)} chunks.")

        self.graph = nx.DiGraph()

        # --- Pass 1: add all nodes ---
        print("Building nodes...")
        for chunk in chunks:
            htsno = chunk["metadata"].get("htsno", "").strip()
            if not htsno:
                continue
            text = chunk.get("text", "")
            self.graph.add_node(
                htsno,
                chunk_id=chunk.get("id", ""),
                description=chunk["metadata"].get("description", ""),
                chapter=chunk["metadata"].get("chapter", ""),
                indent=int(chunk["metadata"].get("indent", 0) or 0),
                hierarchy=_extract_field(text, "Hierarchy") or "",
                general=_extract_field(text, "General Duty Rate") or "",
                special=_extract_field(text, "Special Duty Rate") or "",
                other=_extract_field(text, "Other Duty Rate") or "",
                additional_duties=_extract_field(text, "Additional Duties") or "",
                text=text,
            )

        node_count = self.graph.number_of_nodes()
        print(f"Added {node_count} nodes (one per HTS code).")

        # --- Pass 2: IS_CHILD_OF edges via indent stack ---
        print("Building hierarchy (IS_CHILD_OF) edges...")
        # Stack of (indent_level, htsno)
        indent_stack: List[Tuple[int, str]] = []
        child_edge_count = 0

        for chunk in chunks:
            htsno = chunk["metadata"].get("htsno", "").strip()
            if not htsno or htsno not in self.graph:
                continue
            current_indent = int(chunk["metadata"].get("indent", 0) or 0)

            # Pop everything from the stack whose indent >= current_indent.
            while indent_stack and indent_stack[-1][0] >= current_indent:
                indent_stack.pop()

            # The parent is now the top of the stack (if any).
            if indent_stack:
                parent_indent, parent_htsno = indent_stack[-1]
                if parent_htsno in self.graph:
                    self.graph.add_edge(
                        htsno,
                        parent_htsno,
                        relation=REL_CHILD,
                        weight=1.0,
                    )
                    child_edge_count += 1

            indent_stack.append((current_indent, htsno))

        print(f"Added {child_edge_count} hierarchy (IS_CHILD_OF) edges.")

        # --- Pass 3: SAME_CHAPTER_SIBLING edges ---
        # Strategy: for each (chapter, parent) group, store the parent htsno
        # as a node attribute so siblings can be discovered via a shared-parent
        # lookup instead of explicit O(N²) clique edges.
        # We also add a lightweight SAME_CHAPTER_SIBLING edge from each node to
        # the *first representative* of its sibling group (capped at
        # MAX_SIBLING_EDGES_PER_GROUP per group) so graph traversal still works
        # without blowing up edge counts in large chapters (e.g. Ch. 84 has
        # thousands of codes, which would produce millions of pairs).
        MAX_SIBLING_EDGES_PER_GROUP = 10

        print("Building sibling (SAME_CHAPTER_SIBLING) edges...")
        sibling_groups: Dict[Tuple, List[str]] = {}
        for node, data in self.graph.nodes(data=True):
            chapter = data.get("chapter", "")
            # immediate parent = first IS_CHILD_OF successor
            parents = [
                v
                for u, v, d in self.graph.out_edges(node, data=True)
                if d.get("relation") == REL_CHILD
            ]
            parent_key = parents[0] if parents else "__root__"
            # Store parent htsno as a node attribute for O(1) sibling lookup.
            self.graph.nodes[node]["parent_htsno"] = parent_key
            key = (chapter, parent_key)
            sibling_groups.setdefault(key, []).append(node)

        sibling_edge_count = 0
        for (chapter, parent_key), members in sibling_groups.items():
            if len(members) < 2:
                continue
            # Connect each member to up to MAX_SIBLING_EDGES_PER_GROUP
            # neighbours (sequential pairs within the sorted group).
            # This gives a linked-list-style chain that supports traversal
            # without the O(N²) cost of a full clique.
            sorted_members = sorted(members)
            for i, node in enumerate(sorted_members):
                neighbours = sorted_members[
                    max(0, i - MAX_SIBLING_EDGES_PER_GROUP // 2) :
                    i + MAX_SIBLING_EDGES_PER_GROUP // 2 + 1
                ]
                for other in neighbours:
                    if other == node:
                        continue
                    if not self.graph.has_edge(node, other):
                        self.graph.add_edge(
                            node, other,
                            relation=REL_SIBLING,
                            chapter=chapter,
                            parent=parent_key,
                            weight=0.5,
                        )
                        sibling_edge_count += 1

        print(f"Added {sibling_edge_count} sibling (SAME_CHAPTER_SIBLING) edges.")

        total_edges = self.graph.number_of_edges()
        print(
            f"\nGraph built: {node_count} nodes, {total_edges} edges total."
        )
        self._built = True
        return self

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(
        self,
        graphml_path: str = DEFAULT_GRAPHML_PATH,
        gpickle_path: str = DEFAULT_GPICKLE_PATH,
    ) -> "HTSKnowledgeGraph":
        """Persist the graph as GraphML (portable) and pickle (fast)."""
        if not self._built:
            raise RuntimeError("Graph has not been built yet. Call build_from_chunks() first.")

        # Strip the 'text' attribute before writing GraphML to keep it small;
        # text is available in hts_chunks.json and would bloat the XML.
        g_export = self.graph.copy()
        for node in g_export.nodes():
            g_export.nodes[node].pop("text", None)

        os.makedirs(os.path.dirname(graphml_path) or ".", exist_ok=True)

        print(f"Writing GraphML to '{graphml_path}'...")
        nx.write_graphml(g_export, graphml_path)
        print(f"  Written ({Path(graphml_path).stat().st_size / 1024 / 1024:.1f} MB).")

        print(f"Writing pickle to '{gpickle_path}'...")
        with open(gpickle_path, "wb") as f:
            pickle.dump(self.graph, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  Written ({Path(gpickle_path).stat().st_size / 1024 / 1024:.1f} MB).")

        return self

    @classmethod
    def load(
        cls,
        gpickle_path: str = DEFAULT_GPICKLE_PATH,
    ) -> "HTSKnowledgeGraph":
        """Load graph from pickle (fastest) or fall back to GraphML."""
        p = Path(gpickle_path)
        if not p.exists():
            raise FileNotFoundError(
                f"Graph pickle not found: {p}. "
                "Run 'python step7_knowledgegraph.py build' first."
            )
        kg = cls()
        print(f"Loading graph from '{p}'...")
        with open(p, "rb") as f:
            kg.graph = pickle.load(f)
        kg._built = True
        g = kg.graph
        print(
            f"Loaded graph: {g.number_of_nodes()} nodes, "
            f"{g.number_of_edges()} edges."
        )
        return kg

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    def _node_exists(self, htsno: str) -> bool:
        return htsno in self.graph

    def get_ancestors(self, htsno: str) -> List[Dict[str, Any]]:
        """
        Return the ancestor chain for htsno, ordered from immediate parent
        to root.  Each entry is {htsno, description, indent, general}.
        """
        if not self._node_exists(htsno):
            return []
        ancestors = []
        visited = {htsno}
        current = htsno
        while True:
            parents = [
                v
                for u, v, d in self.graph.out_edges(current, data=True)
                if d.get("relation") == REL_CHILD and v not in visited
            ]
            if not parents:
                break
            parent = parents[0]
            visited.add(parent)
            data = self.graph.nodes[parent]
            ancestors.append(
                {
                    "htsno": parent,
                    "description": data.get("description", ""),
                    "indent": data.get("indent", ""),
                    "general": data.get("general", ""),
                    "chapter": data.get("chapter", ""),
                }
            )
            current = parent
        return ancestors

    def get_children(self, htsno: str, depth: int = 1) -> List[Dict[str, Any]]:
        """
        Return all descendant nodes of htsno up to `depth` hops away,
        following IS_CHILD_OF edges in reverse (parent <- child direction).
        Each entry is {htsno, description, indent, general, depth}.
        """
        if not self._node_exists(htsno):
            return []
        children = []
        visited = {htsno}
        frontier = {htsno}
        for current_depth in range(1, depth + 1):
            next_frontier = set()
            for node in frontier:
                # IS_CHILD_OF edges point child -> parent, so we want
                # in-edges of node that carry REL_CHILD.
                for u, v, d in self.graph.in_edges(node, data=True):
                    if d.get("relation") == REL_CHILD and u not in visited:
                        visited.add(u)
                        next_frontier.add(u)
                        data = self.graph.nodes[u]
                        children.append(
                            {
                                "htsno": u,
                                "description": data.get("description", ""),
                                "indent": data.get("indent", ""),
                                "general": data.get("general", ""),
                                "chapter": data.get("chapter", ""),
                                "depth": current_depth,
                            }
                        )
            frontier = next_frontier
            if not frontier:
                break
        return children

    def get_siblings(self, htsno: str) -> List[Dict[str, Any]]:
        """
        Return all nodes sharing the same immediate parent as htsno.

        Uses the 'parent_htsno' node attribute (stored during build) to
        find all children of the same parent in O(degree) time, which is
        much faster than iterating SAME_CHAPTER_SIBLING edges (which are
        only a bounded chain, not a full clique).
        """
        if not self._node_exists(htsno):
            return []

        # Prefer the O(1) parent attribute set during build.
        parent_htsno = self.graph.nodes[htsno].get("parent_htsno")
        if parent_htsno and parent_htsno != "__root__" and self._node_exists(parent_htsno):
            # All children of the same parent are our siblings.
            siblings = []
            for u, v, d in self.graph.in_edges(parent_htsno, data=True):
                if d.get("relation") == REL_CHILD and u != htsno:
                    data = self.graph.nodes[u]
                    siblings.append(
                        {
                            "htsno": u,
                            "description": data.get("description", ""),
                            "indent": data.get("indent", ""),
                            "general": data.get("general", ""),
                            "chapter": data.get("chapter", ""),
                        }
                    )
            return siblings

        # Fallback: traverse SAME_CHAPTER_SIBLING edges.
        siblings = []
        for u, v, d in self.graph.out_edges(htsno, data=True):
            if d.get("relation") == REL_SIBLING:
                data = self.graph.nodes[v]
                siblings.append(
                    {
                        "htsno": v,
                        "description": data.get("description", ""),
                        "indent": data.get("indent", ""),
                        "general": data.get("general", ""),
                        "chapter": data.get("chapter", ""),
                    }
                )
        return siblings

    def get_chapter_nodes(self, chapter: str) -> List[Dict[str, Any]]:
        """Return all nodes belonging to the given chapter (e.g. '01')."""
        result = []
        for node, data in self.graph.nodes(data=True):
            if data.get("chapter") == chapter:
                result.append(
                    {
                        "htsno": node,
                        "description": data.get("description", ""),
                        "indent": data.get("indent", ""),
                        "general": data.get("general", ""),
                    }
                )
        # Sort by htsno for readability.
        result.sort(key=lambda x: x["htsno"])
        return result

    def find_by_duty_rate(self, rate_pattern: str, field: str = "general") -> List[Dict[str, Any]]:
        """
        Return all nodes whose `field` duty rate contains `rate_pattern`
        (case-insensitive substring match).

        Parameters
        ----------
        rate_pattern : str
            Pattern to search, e.g. "Free", "6.5%", "cents/kg".
        field : str
            One of 'general', 'special', 'other', 'additional_duties'.
        """
        pat = rate_pattern.lower()
        result = []
        for node, data in self.graph.nodes(data=True):
            val = data.get(field, "") or ""
            if pat in val.lower():
                result.append(
                    {
                        "htsno": node,
                        "description": data.get("description", ""),
                        "chapter": data.get("chapter", ""),
                        field: val,
                    }
                )
        result.sort(key=lambda x: x["htsno"])
        return result

    def enrich_retrieval(
        self,
        results: List[Dict[str, Any]],
        ancestor_depth: int = 3,
        children_depth: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Augment a list of HybridRetriever result dicts with graph context.

        For each result, adds:
          - 'graph_ancestors'  : list of ancestor node dicts (closest first)
          - 'graph_children'   : list of direct child node dicts
          - 'graph_siblings'   : list of sibling node dicts
          - 'graph_in_graph'   : bool, whether the htsno was found in the graph

        Parameters
        ----------
        results : list of dicts
            Output from HybridRetriever.hybrid_search().
        ancestor_depth : int
            Maximum number of ancestor hops to follow (capped internally).
        children_depth : int
            Depth for get_children().
        """
        enriched = []
        for r in results:
            htsno = r.get("metadata", {}).get("htsno", "")
            in_graph = self._node_exists(htsno)
            r = dict(r)  # shallow copy to avoid mutating caller's list
            r["graph_in_graph"] = in_graph
            if in_graph:
                ancestors = self.get_ancestors(htsno)[:ancestor_depth]
                children = self.get_children(htsno, depth=children_depth)
                siblings = self.get_siblings(htsno)
            else:
                ancestors, children, siblings = [], [], []
            r["graph_ancestors"] = ancestors
            r["graph_children"] = children
            r["graph_siblings"] = siblings
            enriched.append(r)
        return enriched

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_graphml(self, path: str) -> None:
        """Export the graph (without text blobs) to a GraphML file."""
        g_export = self.graph.copy()
        for node in g_export.nodes():
            g_export.nodes[node].pop("text", None)
        nx.write_graphml(g_export, path)
        print(f"GraphML exported to '{path}'.")

    # ------------------------------------------------------------------
    # Summary / diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Print and return a summary of graph statistics."""
        g = self.graph
        n_nodes = g.number_of_nodes()
        n_edges = g.number_of_edges()

        child_edges = sum(
            1 for u, v, d in g.edges(data=True) if d.get("relation") == REL_CHILD
        )
        sibling_edges = sum(
            1 for u, v, d in g.edges(data=True) if d.get("relation") == REL_SIBLING
        )

        chapters = set(
            d.get("chapter", "") for _, d in g.nodes(data=True) if d.get("chapter")
        )

        in_degrees = sorted(dict(g.in_degree()).values(), reverse=True)
        out_degrees = sorted(dict(g.out_degree()).values(), reverse=True)

        stats = {
            "nodes": n_nodes,
            "edges": n_edges,
            "child_edges": child_edges,
            "sibling_edges": sibling_edges,
            "chapters": len(chapters),
            "max_in_degree": in_degrees[0] if in_degrees else 0,
            "max_out_degree": out_degrees[0] if out_degrees else 0,
        }

        print()
        print(_SEP)
        print("  HTS Knowledge Graph  -  Summary")
        print(_SEP)
        print(f"  Nodes (HTS codes)    : {stats['nodes']:,}")
        print(f"  Edges (total)        : {stats['edges']:,}")
        print(f"    IS_CHILD_OF        : {stats['child_edges']:,}")
        print(f"    SAME_CHAPTER_SIBLING: {stats['sibling_edges']:,}")
        print(f"  Chapters covered     : {stats['chapters']}")
        print(f"  Max in-degree        : {stats['max_in_degree']}")
        print(f"  Max out-degree       : {stats['max_out_degree']}")
        print(_SEP)

        return stats


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def _print_node_list(nodes: List[Dict[str, Any]], title: str = ""):
    if title:
        print(f"\n{title}")
    print(_SEP_THIN)
    if not nodes:
        print("  (none)")
    else:
        for n in nodes:
            htsno = n.get("htsno", "-")
            desc = n.get("description", "-")
            indent_val = n.get("indent", "")
            general = n.get("general", "")
            depth = n.get("depth", "")
            depth_str = f" [depth {depth}]" if depth else ""
            duty_str = f" | duty: {general}" if general else ""
            print(f"  {htsno:<22} | indent {indent_val}{depth_str}{duty_str} | {desc}")
    print(_SEP_THIN)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="step7_knowledgegraph.py",
        description="HTS Knowledge Graph builder and query tool.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  build                    Build and save the graph from hts_chunks.json
  stats                    Print graph statistics (loads existing graph)
  ancestors <htsno>        Print ancestor chain for a given HTS code
  children  <htsno>        Print direct children of an HTS code
  siblings  <htsno>        Print sibling codes (same parent)
  chapter   <ch>           List all HTS codes in a chapter (e.g. 01)
  duty      <pattern>      Find codes whose general duty rate matches pattern
  enrich    <query>        Run HybridRetriever search and enrich with graph context
  export    <path>         Re-export the graph as GraphML to a given path

Examples:
  python step7_knowledgegraph.py build
  python step7_knowledgegraph.py stats
  python step7_knowledgegraph.py ancestors 0101.21.00.10
  python step7_knowledgegraph.py children 0101.21.00
  python step7_knowledgegraph.py chapter 61
  python step7_knowledgegraph.py duty "Free"
  python step7_knowledgegraph.py enrich "purebred breeding horses"
""",
    )

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # build
    build_p = sub.add_parser("build", help="Build the knowledge graph.")
    build_p.add_argument(
        "--chunks", default=DEFAULT_CHUNKS_PATH, help="Path to hts_chunks.json."
    )
    build_p.add_argument(
        "--graphml", default=DEFAULT_GRAPHML_PATH, help="Output GraphML path."
    )
    build_p.add_argument(
        "--gpickle", default=DEFAULT_GPICKLE_PATH, help="Output pickle path."
    )
    build_p.add_argument(
        "--no-siblings",
        action="store_true",
        help="Skip sibling edge construction (faster, lower memory).",
    )

    # stats
    sub.add_parser("stats", help="Print graph statistics.")

    # ancestors
    anc_p = sub.add_parser("ancestors", help="Print ancestor chain.")
    anc_p.add_argument("htsno", help="HTS code (e.g. 0101.21.00.10).")

    # children
    chi_p = sub.add_parser("children", help="Print direct (and deeper) children.")
    chi_p.add_argument("htsno", help="HTS code (e.g. 0101.21.00).")
    chi_p.add_argument(
        "--depth", type=int, default=1, help="How many levels deep (default: 1)."
    )

    # siblings
    sib_p = sub.add_parser("siblings", help="Print sibling codes.")
    sib_p.add_argument("htsno", help="HTS code.")

    # chapter
    ch_p = sub.add_parser("chapter", help="List HTS codes in a chapter.")
    ch_p.add_argument("chapter", help="Two-digit chapter number (e.g. 01, 61).")

    # duty
    duty_p = sub.add_parser("duty", help="Find codes by duty rate pattern.")
    duty_p.add_argument("pattern", help="Substring to search in General Duty Rate.")
    duty_p.add_argument(
        "--field",
        default="general",
        choices=["general", "special", "other", "additional_duties"],
        help="Which duty rate field to search (default: general).",
    )
    duty_p.add_argument(
        "--limit", type=int, default=20, help="Max results to display (default: 20)."
    )

    # enrich
    enrich_p = sub.add_parser(
        "enrich", help="Run hybrid search and enrich results with graph context."
    )
    enrich_p.add_argument("query", help="Search query.")
    enrich_p.add_argument(
        "--top-k", type=int, default=3, help="Number of retrieval results (default: 3)."
    )

    # export
    export_p = sub.add_parser("export", help="Re-export graph as GraphML.")
    export_p.add_argument("path", help="Output GraphML file path.")

    return parser.parse_args(argv)


def _load_graph(root: Path) -> HTSKnowledgeGraph:
    """Load graph from disk, changing to the project root first."""
    original_cwd = os.getcwd()
    os.chdir(str(root))
    try:
        return HTSKnowledgeGraph.load(DEFAULT_GPICKLE_PATH)
    finally:
        os.chdir(original_cwd)


def main(argv=None):
    args = _parse_args(argv)
    root = _HERE
    original_cwd = os.getcwd()

    # ------------------------------------------------------------------
    if args.command == "build":
        os.chdir(str(root))
        try:
            t0 = time.perf_counter()
            kg = HTSKnowledgeGraph()
            kg.build_from_chunks(args.chunks)
            kg.save(graphml_path=args.graphml, gpickle_path=args.gpickle)
            elapsed = time.perf_counter() - t0
            print(f"\nDone. Total time: {elapsed:.1f}s")
            kg.summary()

            # Demo: show neighbourhood of a known code.
            demo_code = "0101.21.00.10"
            if demo_code in kg.graph:
                print(f"\nDemo: neighbourhood of '{demo_code}'")
                ancestors = kg.get_ancestors(demo_code)
                _print_node_list(ancestors, f"Ancestors of {demo_code}")
                children = kg.get_children(demo_code)
                _print_node_list(children, f"Children of {demo_code}")
        finally:
            os.chdir(original_cwd)

    # ------------------------------------------------------------------
    elif args.command == "stats":
        kg = _load_graph(root)
        kg.summary()

    # ------------------------------------------------------------------
    elif args.command == "ancestors":
        kg = _load_graph(root)
        ancestors = kg.get_ancestors(args.htsno)
        if not kg._node_exists(args.htsno):
            print(f"HTS code '{args.htsno}' not found in the graph.")
            sys.exit(1)
        _print_node_list(ancestors, f"Ancestors of '{args.htsno}' (closest first)")

    # ------------------------------------------------------------------
    elif args.command == "children":
        kg = _load_graph(root)
        if not kg._node_exists(args.htsno):
            print(f"HTS code '{args.htsno}' not found in the graph.")
            sys.exit(1)
        children = kg.get_children(args.htsno, depth=args.depth)
        _print_node_list(children, f"Children of '{args.htsno}' (depth={args.depth})")

    # ------------------------------------------------------------------
    elif args.command == "siblings":
        kg = _load_graph(root)
        if not kg._node_exists(args.htsno):
            print(f"HTS code '{args.htsno}' not found in the graph.")
            sys.exit(1)
        siblings = kg.get_siblings(args.htsno)
        _print_node_list(siblings, f"Siblings of '{args.htsno}'")

    # ------------------------------------------------------------------
    elif args.command == "chapter":
        kg = _load_graph(root)
        nodes = kg.get_chapter_nodes(args.chapter)
        _print_node_list(nodes, f"HTS codes in Chapter {args.chapter} ({len(nodes)} total)")

    # ------------------------------------------------------------------
    elif args.command == "duty":
        kg = _load_graph(root)
        matches = kg.find_by_duty_rate(args.pattern, field=args.field)
        print(f"\nCodes with {args.field} duty matching '{args.pattern}': {len(matches)} found.")
        _print_node_list(matches[: args.limit], f"Top {args.limit} results")
        if len(matches) > args.limit:
            print(f"  ... and {len(matches) - args.limit} more.")

    # ------------------------------------------------------------------
    elif args.command == "enrich":
        # Load both retriever and graph, then show enriched results.
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        os.chdir(str(root))
        try:
            from step5_retriever import HybridRetriever  # type: ignore

            print("Loading HybridRetriever...")
            retriever = HybridRetriever()
            print(f"Loading knowledge graph...")
            kg = HTSKnowledgeGraph.load(DEFAULT_GPICKLE_PATH)

            results = retriever.hybrid_search(args.query, top_k=args.top_k)
            enriched = kg.enrich_retrieval(results)

            print(f"\nEnriched results for: '{args.query}'")
            print(_SEP)
            for rank, r in enumerate(enriched, 1):
                htsno = r["metadata"].get("htsno", "-")
                desc = r["metadata"].get("description", "-")
                score = r["rrf_score"]
                in_graph = r["graph_in_graph"]
                print(
                    f"\nRank {rank} | RRF {score:.4f} | {htsno} | {desc}"
                    f"  [{'in graph' if in_graph else 'NOT in graph'}]"
                )
                print(f"  {_SEP_THIN}")

                ancestors = r["graph_ancestors"]
                if ancestors:
                    print("  Ancestors (closest first):")
                    for a in ancestors:
                        print(f"    <- {a['htsno']:<22} | {a['description']}")

                children = r["graph_children"]
                if children:
                    print("  Children:")
                    for c in children:
                        print(f"    -> {c['htsno']:<22} | {c['description']}")

                siblings = r["graph_siblings"]
                if siblings:
                    print(f"  Siblings ({len(siblings)} total, showing first 3):")
                    for s in siblings[:3]:
                        print(f"    ~~ {s['htsno']:<22} | {s['description']}")
        finally:
            os.chdir(original_cwd)

    # ------------------------------------------------------------------
    elif args.command == "export":
        kg = _load_graph(root)
        os.chdir(str(root))
        try:
            kg.export_graphml(args.path)
        finally:
            os.chdir(original_cwd)

    # ------------------------------------------------------------------
    else:
        # No subcommand: build by default (backwards compatibility).
        os.chdir(str(root))
        try:
            t0 = time.perf_counter()
            kg = HTSKnowledgeGraph()
            kg.build_from_chunks(DEFAULT_CHUNKS_PATH)
            kg.save()
            elapsed = time.perf_counter() - t0
            print(f"\nDone. Total time: {elapsed:.1f}s")
            kg.summary()
        finally:
            os.chdir(original_cwd)


if __name__ == "__main__":
    main()
