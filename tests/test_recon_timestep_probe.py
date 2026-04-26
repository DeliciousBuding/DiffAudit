import unittest

import numpy as np


class ReconTimestepProbeTests(unittest.TestCase):
    def test_analyze_recon_timestep_probe_selects_better_shadow_dimension(self) -> None:
        from diffaudit.attacks.recon_timestep_probe import analyze_recon_timestep_probe

        shadow_member = np.asarray([[0.45, 0.95], [0.4, 0.85], [0.55, 0.8]], dtype=float)
        shadow_nonmember = np.asarray([[0.5, 0.1], [0.52, 0.2], [0.48, 0.15]], dtype=float)
        target_member = np.asarray([[0.46, 0.9], [0.44, 0.8], [0.5, 0.85]], dtype=float)
        target_nonmember = np.asarray([[0.51, 0.2], [0.52, 0.25], [0.49, 0.3]], dtype=float)

        result = analyze_recon_timestep_probe(
            shadow_member=shadow_member,
            shadow_nonmember=shadow_nonmember,
            target_member=target_member,
            target_nonmember=target_nonmember,
        )

        self.assertEqual(result["best_shadow_dim"], 1)
        self.assertGreater(result["selected_target"]["auc"], result["baseline_target"]["auc"])
        self.assertGreater(result["target_auc_gain_vs_dim0"], 0.0)


if __name__ == "__main__":
    unittest.main()
