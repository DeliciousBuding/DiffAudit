import unittest


class SemanticAuxFusionTests(unittest.TestCase):
    def test_analyze_semantic_aux_fusion_prefers_mean_cos_when_other_signals_are_noise(self) -> None:
        from diffaudit.attacks.semantic_aux_fusion import analyze_semantic_aux_fusion

        records = []
        member_mean_cos = [0.81, 0.8, 0.79, 0.78]
        nonmember_mean_cos = [0.62, 0.61, 0.6, 0.59]
        noisy_max_cos = [0.5, 0.3, 0.7, 0.4, 0.45, 0.65, 0.35, 0.55]
        noisy_max_ssim = [0.1, 0.8, 0.2, 0.7, 0.75, 0.15, 0.65, 0.25]
        for index, score in enumerate(member_mean_cos):
            records.append(
                {
                    "label": 1,
                    "mean_cos": score,
                    "max_cos": noisy_max_cos[index],
                    "max_ssim": noisy_max_ssim[index],
                }
            )
        for offset, score in enumerate(nonmember_mean_cos, start=len(member_mean_cos)):
            records.append(
                {
                    "label": 0,
                    "mean_cos": score,
                    "max_cos": noisy_max_cos[offset],
                    "max_ssim": noisy_max_ssim[offset],
                }
            )

        result = analyze_semantic_aux_fusion(records)

        self.assertEqual(result["best_candidate"], "mean_cos")
        self.assertEqual(result["best_auc_gain_vs_mean_cos"], 0.0)
        self.assertGreater(result["candidates"]["mean_cos"]["auc"], result["candidates"]["max_cos"]["auc"])


if __name__ == "__main__":
    unittest.main()
