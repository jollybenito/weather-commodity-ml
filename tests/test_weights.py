"""Unit tests for spatial production weights and weather aggregation."""

import unittest
import numpy as np
import pandas as pd

from src.data_ingestion.usda_weights import (
    get_corn_spatial_config,
    get_soybeans_spatial_config,
    get_wheat_spatial_config,
    compute_spatially_weighted_weather
)


class TestUSDAWeights(unittest.TestCase):

    def test_corn_weights_sum_to_one(self):
        config = get_corn_spatial_config()
        total_weight = sum(r.weight for r in config.regions.values())
        self.assertAlmostEqual(total_weight, 1.0, places=4)
        self.assertIn("IA", config.regions)
        self.assertIn("IL", config.regions)
        self.assertIn("NE", config.regions)

    def test_soybean_and_wheat_configs(self):
        soy = get_soybeans_spatial_config()
        self.assertAlmostEqual(sum(r.weight for r in soy.regions.values()), 1.0, places=4)
        wheat = get_wheat_spatial_config()
        self.assertAlmostEqual(sum(r.weight for r in wheat.regions.values()), 1.0, places=4)

    def test_spatially_weighted_weather_calculation(self):
        config = get_corn_spatial_config()
        dates = pd.date_range("2024-06-01", "2024-06-10", freq="D")
        mock_df = pd.DataFrame(index=dates)
        
        # Populate uniform 30.0 for IA and 20.0 for others
        for code in config.regions.keys():
            mock_df[f"{code}_temp_max"] = 30.0 if code == "IA" else 20.0

        res = compute_spatially_weighted_weather(mock_df, config, ["temp_max"])
        self.assertIn("weighted_temp_max", res.columns)
        
        ia_w = config.regions["IA"].weight
        expected_weighted = ia_w * 30.0 + (1.0 - ia_w) * 20.0
        np.testing.assert_allclose(res["weighted_temp_max"].values, expected_weighted, atol=1e-3)


if __name__ == "__main__":
    unittest.main()

