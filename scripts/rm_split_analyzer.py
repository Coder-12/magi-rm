import json
import numpy as np
from scipy.spatial.distance import jensenshannon

# Load the JSON data (replace with your actual JSON content or file path)
json_data = '''
{
  "timestamp": "2025-11-01T05:56:40.282422+00:00",
  "train": {
    "PRM": {
      "num_examples": 4679,
      "mean": 0.796,
      "stdev": 0.179,
      "entropy": 2.326,
      "step_mean": 6,
      "step_median": 6,
      "step_min": 2,
      "step_max": 15,
      "hist_bins": [
        0.0,
        0.1,
        0.2,
        0.30000000000000004,
        0.4,
        0.5,
        0.6000000000000001,
        0.7000000000000001,
        0.8,
        0.9,
        1.0
      ],
      "hist_pct": [
        0.0,
        0.019,
        0.018,
        0.018,
        0.028,
        0.051,
        0.054,
        0.116,
        0.369,
        0.327
      ]
    },
    "ORM": {
      "num_examples": 4679,
      "label_1_ratio": 0.754
    }
  },
  "val": {
    "PRM": {
      "num_examples": 549,
      "mean": 0.782,
      "stdev": 0.191,
      "entropy": 2.439,
      "step_mean": 6,
      "step_median": 6,
      "step_min": 3,
      "step_max": 12,
      "hist_bins": [
        0.0,
        0.1,
        0.2,
        0.30000000000000004,
        0.4,
        0.5,
        0.6000000000000001,
        0.7000000000000001,
        0.8,
        0.9,
        1.0
      ],
      "hist_pct": [
        0.004,
        0.026,
        0.016,
        0.015,
        0.027,
        0.069,
        0.058,
        0.133,
        0.332,
        0.321
      ]
    },
    "ORM": {
      "num_examples": 549,
      "label_1_ratio": 0.73
    }
  },
  "test": {
    "PRM": {
      "num_examples": 573,
      "mean": 0.781,
      "stdev": 0.197,
      "entropy": 2.37,
      "step_mean": 6,
      "step_median": 6,
      "step_min": 3,
      "step_max": 12,
      "hist_bins": [
        0.0,
        0.1,
        0.2,
        0.30000000000000004,
        0.4,
        0.5,
        0.6000000000000001,
        0.7000000000000001,
        0.8,
        0.9,
        1.0
      ],
      "hist_pct": [
        0.0,
        0.037,
        0.023,
        0.01,
        0.021,
        0.058,
        0.059,
        0.113,
        0.382,
        0.297
      ]
    },
    "ORM": {
      "num_examples": 573,
      "label_1_ratio": 0.756
    }
  }
}
'''
data = json.loads(json_data)

# Extract hist_pct (normalized probabilities)
train_hist = np.array(data['train']['PRM']['hist_pct'])
val_hist = np.array(data['val']['PRM']['hist_pct'])
test_hist = np.array(data['test']['PRM']['hist_pct'])

# 1. Jensen-Shannon Divergence (JSD)
jsd_train_val = jensenshannon(train_hist, val_hist)
jsd_train_test = jensenshannon(train_hist, test_hist)
jsd_val_test = jensenshannon(val_hist, test_hist)
avg_jsd = np.mean([jsd_train_val, jsd_train_test, jsd_val_test])

print(f"JSD Train-Val: {jsd_train_val:.4f}")
print(f"JSD Train-Test: {jsd_train_test:.4f}")
print(f"JSD Val-Test: {jsd_val_test:.4f}")
print(f"Average JSD: {avg_jsd:.4f}")

# 2. Skewness (Approximated using bin centers)
bin_centers = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])

def calc_skew(hist_pct, bin_centers, mean, stdev):
    if stdev == 0:
        return 0.0
    skew = np.sum(((bin_centers - mean) ** 3) * hist_pct) / (stdev ** 3)
    return skew

train_skew = calc_skew(train_hist, bin_centers, data['train']['PRM']['mean'], data['train']['PRM']['stdev'])
val_skew = calc_skew(val_hist, bin_centers, data['val']['PRM']['mean'], data['val']['PRM']['stdev'])
test_skew = calc_skew(test_hist, bin_centers, data['test']['PRM']['mean'], data['test']['PRM']['stdev'])
avg_skew = np.mean([train_skew, val_skew, test_skew])

print(f"Skewness Train: {train_skew:.4f}")
print(f"Skewness Val: {val_skew:.4f}")
print(f"Skewness Test: {test_skew:.4f}")
print(f"Average Skewness: {avg_skew:.4f}")

# 3. Label Imbalance Distance
train_imbal = abs(data['train']['ORM']['label_1_ratio'] - 0.5)
val_imbal = abs(data['val']['ORM']['label_1_ratio'] - 0.5)
test_imbal = abs(data['test']['ORM']['label_1_ratio'] - 0.5)
avg_imbal = np.mean([train_imbal, val_imbal, test_imbal])

print(f"Imbalance Train: {train_imbal:.4f}")
print(f"Imbalance Val: {val_imbal:.4f}")
print(f"Imbalance Test: {test_imbal:.4f}")
print(f"Average Imbalance: {avg_imbal:.4f}")

# 4. Composite Quality Score
norm_jsd = 1 - avg_jsd
norm_imbal = 1 - avg_imbal
skew_diff = max([train_skew, val_skew, test_skew]) - min([train_skew, val_skew, test_skew])
norm_skew = 1 - (skew_diff / 2.0)  # Normalize skew variation (assuming max reasonable diff ~2)
composite = norm_jsd * norm_imbal * norm_skew

print(f"Composite Quality Score: {composite:.3f}")