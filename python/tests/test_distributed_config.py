# Copyright © 2026 Apple Inc.

import unittest

from mlx._distributed_utils.config import jaccl_ring_devices


def links(*pairs):
    """A device map keyed the way IPConfigurator keys it, one cable per pair."""
    ips = {}
    for a, b in pairs:
        ips.setdefault((a, b), []).append((f"en{a}{b}", "192.168.0.1"))
        ips.setdefault((b, a), []).append((f"en{b}{a}", "192.168.0.2"))
    return ips


class TestJacclRingDevices(unittest.TestCase):
    def test_ring_only(self):
        # Four nodes cabled as a ring and nothing else. Non-adjacent pairs have
        # no cable, so they stay null and the hostfile is what it always was.
        m = jaccl_ring_devices([0, 1, 2, 3], 1, links((0, 1), (1, 2), (2, 3), (3, 0)))
        self.assertEqual(
            m,
            [
                [None, "rdma_en01", None, "rdma_en03"],
                ["rdma_en10", None, "rdma_en12", None],
                [None, "rdma_en21", None, "rdma_en23"],
                ["rdma_en30", None, "rdma_en32", None],
            ],
        )

    def test_extra_cable_is_recorded(self):
        # The same ring with one chord. The chord is not part of the data plane
        # and is still written down: a subgroup can only hold members that are
        # directly connected, and a link dropped here cannot be recovered,
        # because the hostfile is everything the runtime is given.
        m = jaccl_ring_devices(
            [0, 1, 2, 3], 1, links((0, 1), (1, 2), (2, 3), (3, 0), (0, 2))
        )
        self.assertEqual(m[0][2], "rdma_en02")
        self.assertEqual(m[2][0], "rdma_en20")

    def test_shape_contract(self):
        # launch_jaccl rejects a hostfile that is not square with a null
        # diagonal, so both hold whatever the cabling looks like.
        for extra in ([], [(0, 2)], [(0, 2), (1, 3)]):
            m = jaccl_ring_devices(
                [0, 1, 2, 3], 1, links((0, 1), (1, 2), (2, 3), (3, 0), *extra)
            )
            self.assertTrue(all(len(row) == 4 for row in m))
            self.assertTrue(all(m[i][i] is None for i in range(4)))

    def test_ring_order_is_kept(self):
        # Rows come out in ring order, not host order, and rank i is ring[i].
        m = jaccl_ring_devices([2, 0, 3, 1], 1, links((2, 0), (0, 3), (3, 1), (1, 2)))
        self.assertEqual(m[0][1], "rdma_en20")
        self.assertEqual(m[1][2], "rdma_en03")
        self.assertEqual(m[3][0], "rdma_en12")

    def test_multiple_cables_between_neighbours(self):
        # Two cables between every ring neighbour: neighbours keep the width the
        # ring was built with.
        ips = links((0, 1), (1, 2), (2, 3), (3, 0))
        for a, b in [(0, 1), (1, 2), (2, 3), (3, 0)]:
            ips[(a, b)].append((f"en{a}{b}x", "192.168.0.3"))
            ips[(b, a)].append((f"en{b}{a}x", "192.168.0.4"))
        m = jaccl_ring_devices([0, 1, 2, 3], 2, ips)
        self.assertEqual(m[0][1], ["rdma_en01", "rdma_en01x"])


if __name__ == "__main__":
    unittest.main()
