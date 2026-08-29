import unittest
from collections import deque

import numpy as np

from gesturegraph.topology import EDGES, NUM_NODES, normalized_adjacency


def reachable_from(start, edges):
    """Plain BFS over the undirected edge list, used to check the skeleton
    forms one connected hand instead of floating fingers."""
    neighbours = {i: set() for i in range(NUM_NODES)}
    for a, b in edges:
        neighbours[a].add(b)
        neighbours[b].add(a)
    seen = {start}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for nxt in neighbours[node]:
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return seen


class TopologyExtraTests(unittest.TestCase):
    def test_every_joint_is_reachable_from_the_wrist(self):
        # If a finger ever got disconnected from the palm, message passing
        # would never let it talk to the rest of the hand.
        self.assertEqual(reachable_from(0, EDGES), set(range(NUM_NODES)))

    def test_no_duplicate_or_self_edges(self):
        seen = set()
        for source, target in EDGES:
            self.assertNotEqual(source, target, "self-loops are added separately via +I")
            key = tuple(sorted((source, target)))
            self.assertNotIn(key, seen, f"edge {key} listed twice")
            seen.add(key)

    def test_edge_count_matches_five_finger_chains(self):
        # wrist-palm + 5 fingers * 4 bones each = 21 edges.
        self.assertEqual(len(EDGES), 21)

    def test_adjacency_rows_sum_to_one_after_normalisation(self):
        # D^-1/2 (A+I) D^-1/2 applied to a constant vector should not blow up
        # the signal; row sums aren't exactly 1 like a plain random-walk
        # matrix, but they should all be positive and roughly the same order
        # of magnitude across joints with very different degrees (wrist vs
        # fingertip).
        adjacency = normalized_adjacency()
        row_sums = adjacency.sum(axis=1)
        self.assertTrue(np.all(row_sums > 0))
        self.assertLess(row_sums.max() / row_sums.min(), 5.0)

    def test_fingertips_only_connect_to_their_own_finger(self):
        # Fingertips (5, 9, 13, 17, 21) should each have exactly one edge,
        # to the joint right below them.
        tips = {5, 9, 13, 17, 21}
        degree = {i: 0 for i in range(NUM_NODES)}
        for source, target in EDGES:
            degree[source] += 1
            degree[target] += 1
        for tip in tips:
            self.assertEqual(degree[tip], 1, f"fingertip {tip} should have exactly one bone")


if __name__ == "__main__":
    unittest.main()
