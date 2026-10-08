from __future__ import annotations

import unittest

from app import recommender
from app.huggingface import parse_parameters_b, parse_quantization
from app.probe import CPUInfo, GPUInfo, HardwareProfile, MemoryInfo


def make_hw(
    ram_gb: int = 32,
    vram_gb: int = 0,
    unified: bool = False,
    backends: tuple[str, ...] = ("cpu",),
) -> HardwareProfile:
    gib = 1024**3

    return HardwareProfile(
        os="linux",
        arch="x86_64",
        cpu=CPUInfo(
            model="test-cpu",
            logical_cores=8,
            avx=True,
            avx2=True,
            avx512=False,
            amx=False,
            neon=False,
        ),
        memory=MemoryInfo(
            ram_bytes=ram_gb * gib,
            gpu_vram_bytes=vram_gb * gib,
            unified_memory=unified,
        ),
        gpu=GPUInfo(vendor="", name=""),
        backends=list(backends),
    )


class BudgetTests(unittest.TestCase):
    def test_cpu_only_budget_is_conservative(self):
        hw = make_hw(ram_gb=32)
        ram_mb = 32 * 1024

        self.assertEqual(
            recommender.available_memory_mb(hw), int(ram_mb * 0.60)
        )

    def test_unified_memory_budget(self):
        hw = make_hw(ram_gb=32, unified=True)
        ram_mb = 32 * 1024

        self.assertEqual(
            recommender.available_memory_mb(hw),
            int(ram_mb * recommender.SAFETY_FACTOR),
        )

    def test_discrete_gpu_budget_combines_vram_and_ram(self):
        hw = make_hw(ram_gb=32, vram_gb=12, backends=("cuda", "cpu"))
        ram_mb = 32 * 1024
        vram_mb = 12 * 1024

        self.assertEqual(
            recommender.available_memory_mb(hw),
            int(vram_mb * 0.80) + int(ram_mb * 0.35),
        )


class FitTests(unittest.TestCase):
    def test_recommendations_never_exceed_budget(self):
        hw = make_hw(ram_gb=8)

        results = recommender.recommend(
            hw,
            workload="general",
            limit=50,
            models=recommender.load_models(),
        )

        self.assertTrue(results)

        for row in results:
            self.assertLessEqual(
                row["estimated_runtime_mb"], row["memory_budget_mb"]
            )
            self.assertLessEqual(row["fit_ratio"], 1.0)

    def test_oversized_models_are_filtered_out(self):
        hw = make_hw(ram_gb=1)

        results = recommender.recommend(
            hw,
            workload="general",
            limit=50,
            models=recommender.load_models(),
        )

        self.assertEqual(results, [])

    def test_workload_filters_catalog(self):
        hw = make_hw(ram_gb=32)

        results = recommender.recommend(
            hw,
            workload="rag",
            limit=50,
            models=recommender.load_models(),
        )

        self.assertTrue(results)
        self.assertTrue(all(r["type"] == "embedding" for r in results))

    def test_context_grows_footprint(self):
        small = recommender.estimate_runtime_mb(2800, "chat", 4096)
        large = recommender.estimate_runtime_mb(2800, "chat", 131072)

        self.assertGreater(large, small)

    def test_model_below_requested_context_is_excluded(self):
        hw = make_hw(ram_gb=64)

        results = recommender.recommend(
            hw,
            workload="general",
            limit=50,
            models=recommender.load_models(),
            context=1_048_576,
        )

        self.assertEqual(results, [])

    def test_model_fit_returns_none_when_over_budget(self):
        hw = make_hw(ram_gb=1)
        model = recommender.load_models()[1]

        self.assertIsNone(
            recommender.model_fit(hw, model, "Q4_K_M")
        )

    def test_model_fit_reports_safety_when_within_budget(self):
        hw = make_hw(ram_gb=32)
        model = recommender.load_models()[1]

        fit = recommender.model_fit(hw, model, "Q4_K_M")

        self.assertIsNotNone(fit)
        self.assertIn(fit["fit"], ("safe", "balanced", "tight"))


class SafetyLabelTests(unittest.TestCase):
    def test_bands(self):
        self.assertEqual(recommender.safety_label(0.5), "safe")
        self.assertEqual(recommender.safety_label(0.6), "safe")
        self.assertEqual(recommender.safety_label(0.7), "balanced")
        self.assertEqual(recommender.safety_label(0.8), "balanced")
        self.assertEqual(recommender.safety_label(0.9), "tight")


class WorkloadScoreTests(unittest.TestCase):
    def test_nonstandard_workload_from_derived_uses(self):
        model = {"uses": ["assistant", "general", "personal"]}

        self.assertEqual(
            recommender.workload_score(model, "assistant"), 100
        )
        self.assertEqual(recommender.workload_score(model, "unknown"), 0)

    def test_model_fit_tolerates_derived_uses(self):
        hw = make_hw(ram_gb=32)
        model = {
            "id": "some/repo-GGUF",
            "name": "repo-GGUF",
            "family": "some",
            "repo": "some/repo-GGUF",
            "type": "chat",
            "context": None,
            "quantizations": {"Q4_K_M": 1000},
            "uses": ["assistant"],
        }

        self.assertIsNotNone(
            recommender.model_fit(hw, model, "Q4_K_M")
        )


class PopularityTests(unittest.TestCase):
    def test_popularity_is_capped(self):
        max_score = recommender.popularity_score(
            {"downloads": 10**9, "likes": 10**6}
        )

        self.assertLessEqual(max_score, 10)


class ParsingTests(unittest.TestCase):
    def test_parse_quantization(self):
        self.assertEqual(
            parse_quantization("model.Q4_K_M.gguf"), "Q4_K_M"
        )
        self.assertEqual(parse_quantization("model.Q8_0.gguf"), "Q8_0")
        self.assertEqual(parse_quantization("model.F16.gguf"), "F16")
        self.assertEqual(
            parse_quantization("model.iq2_xs.gguf"), "IQ2_XS"
        )
        self.assertIsNone(
            parse_quantization("shard-00001-of-00002.gguf")
        )

    def test_parse_parameters_b(self):
        self.assertEqual(parse_parameters_b("Qwen/Qwen3.5-4B-GGUF"), 4.0)
        self.assertEqual(
            parse_parameters_b("ggml-org/Qwen3.5-0.8B-GGUF"), 0.8
        )
        self.assertIsNone(parse_parameters_b("some/model"))


if __name__ == "__main__":
    unittest.main()
