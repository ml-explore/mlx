# Copyright © 2026 Apple Inc.

import importlib.util
import os
import sys
import unittest

# Load module directly
file_path = os.path.join(
    os.path.dirname(__file__),
    "../mlx/production_debt.py",
)
spec = importlib.util.spec_from_file_location("mlx_production_debt", file_path)
production_debt_mod = importlib.util.module_from_spec(spec)
sys.modules["mlx_production_debt"] = production_debt_mod
spec.loader.exec_module(production_debt_mod)

ProductionDebtArrayGate = production_debt_mod.ProductionDebtArrayGate
TechnicalDueDiligenceLedger = production_debt_mod.TechnicalDueDiligenceLedger
GENESIS_HASH = production_debt_mod.GENESIS_HASH


class TestProductionDebtArrayGate(unittest.TestCase):
    def setUp(self) -> None:
        self.gate = ProductionDebtArrayGate(
            never_equate_intent_to_approval=True,
            max_acceptable_mdi=12.0,
        )

    def test_clean_array_execution_passes_readiness(self) -> None:
        report = self.gate.evaluate_array_execution(
            array_id="mlx_llama_3_8b_metal_eval",
            allocated_uma_bytes=16000000000,
            peak_lazy_graph_bytes=16800000000,
            eval_latency_ms=6.2,
            metal_sync_stalls=0,
            un_gated_mutations=0,
        )
        self.assertTrue(report.is_production_ready)
        self.assertLessEqual(report.mdi_score, 12.0)
        self.assertEqual(len(report.critical_smells), 0)
        self.assertTrue(bool(report.receipt_hash))

    def test_degraded_array_execution_fails_debt(self) -> None:
        report = self.gate.evaluate_array_execution(
            array_id="uncalibrated_lazy_graph_array",
            allocated_uma_bytes=16000000000,
            peak_lazy_graph_bytes=45000000000,  # 2.81x UMA sprawl
            eval_latency_ms=65.0,  # High eval latency
            metal_sync_stalls=3,  # 3 Metal stream sync stalls
            un_gated_mutations=2,  # 2 un-gated mutations
        )
        self.assertFalse(report.is_production_ready)
        self.assertGreater(report.mdi_score, 50.0)
        self.assertIn("HIGH_UNIFIED_MEMORY_SPRAWL_2.81X", report.critical_smells)
        self.assertIn("HIGH_ARRAY_EVAL_LATENCY_65.0MS", report.critical_smells)
        self.assertIn("DETECTED_3_METAL_STREAM_SYNC_STALLS", report.critical_smells)
        self.assertIn("DETECTED_2_UNGATED_ARRAY_MUTATIONS", report.critical_smells)

    def test_cryptographic_ledger_integrity(self) -> None:
        self.gate.evaluate_array_execution("array-1")
        self.gate.evaluate_array_execution("array-2")
        self.gate.evaluate_array_execution("array-3")

        entries = self.gate.ledger.get_ledger_entries()
        self.assertEqual(len(entries), 3)
        self.assertEqual(entries[0]["prev_hash"], GENESIS_HASH)
        self.assertEqual(entries[1]["prev_hash"], entries[0]["curr_hash"])
        self.assertEqual(entries[2]["prev_hash"], entries[1]["curr_hash"])
        self.assertTrue(self.gate.ledger.verify_ledger_integrity())


if __name__ == "__main__":
    unittest.main()
