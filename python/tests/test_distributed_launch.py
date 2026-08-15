# Copyright © 2026 Apple Inc.

import unittest

from mlx._distributed_utils.common import Host
from mlx._distributed_utils.launch import missing_jaccl_links


def hosts_from(matrix):
    """Hosts whose rdma matrix is `matrix`, named a, b, c ... by position."""
    return [
        Host(i, chr(ord("a") + i), [f"10.0.0.{i + 1}"], row)
        for i, row in enumerate(matrix)
    ]


# Four nodes in a ring: each reaches its two neighbours and not the one
# opposite. Valid for jaccl-ring, and not a mesh.
RING4 = [
    [None, "rdma_en2", None, "rdma_en3"],
    ["rdma_en2", None, "rdma_en3", None],
    [None, "rdma_en2", None, "rdma_en3"],
    ["rdma_en3", None, "rdma_en2", None],
]


class TestMissingJacclLinks(unittest.TestCase):
    """A hostfile has to describe the links the backend is going to use.

    A matrix of the right shape whose rank order puts unconnected nodes beside
    each other is not merely slow: it leaves every rank alone in a group of
    size one, where "rank 0 of 1" is internally consistent and completely
    wrong. Catching it needs a check against the topology, not the shape.
    """

    def test_mesh_reports_a_missing_pair(self):
        # b and c share no device, so this is not the full mesh jaccl needs.
        matrix = [
            [None, "rdma_en2", "rdma_en3"],
            ["rdma_en2", None, None],
            ["rdma_en3", None, None],
        ]
        self.assertEqual(
            missing_jaccl_links(hosts_from(matrix), False), [(1, 2), (2, 1)]
        )

    def test_ring_reports_unconnected_neighbours(self):
        # A triangle reordered so ranks 0 and 1 have no cable between them.
        # The shape is right and the diagonal is null, which is exactly what
        # let this through before.
        matrix = [
            [None, None, "rdma_en3"],
            [None, None, "rdma_en2"],
            ["rdma_en3", "rdma_en2", None],
        ]
        self.assertEqual(
            missing_jaccl_links(hosts_from(matrix), True), [(0, 1), (1, 0)]
        )

    def test_ring_accepts_a_ring_that_is_not_a_mesh(self):
        self.assertEqual(missing_jaccl_links(hosts_from(RING4), True), [])

    def test_mesh_rejects_that_same_ring(self):
        # Every rank is missing the node opposite it: four ranks, one gap each.
        self.assertEqual(len(missing_jaccl_links(hosts_from(RING4), False)), 4)

    def test_two_ranks(self):
        # The previous and next neighbour are the same node here, so the check
        # must not trip over asking about it twice.
        matrix = [[None, "rdma_en2"], ["rdma_en2", None]]
        self.assertEqual(missing_jaccl_links(hosts_from(matrix), True), [])
        self.assertEqual(missing_jaccl_links(hosts_from(matrix), False), [])

    def test_full_mesh_passes_both(self):
        matrix = [
            [None, "rdma_en2", "rdma_en3"],
            ["rdma_en2", None, "rdma_en4"],
            ["rdma_en3", "rdma_en4", None],
        ]
        self.assertEqual(missing_jaccl_links(hosts_from(matrix), False), [])
        self.assertEqual(missing_jaccl_links(hosts_from(matrix), True), [])


if __name__ == "__main__":
    unittest.main()
