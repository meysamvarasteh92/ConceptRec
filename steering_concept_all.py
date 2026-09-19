import types
from itertools import product

import numpy as np
import pandas as pd
import torch

from recbole.quick_start import load_data_and_model
from recbole.trainer import Trainer
from recbole.evaluator import Collector, Evaluator
from recbole.utils.build_concepts import build_lookups


# ============================================================
# 1. SETTINGS
# ============================================================

DATASET = "ml-1m"
DATA_PATH = "dataset/ml-1m"
MODEL_FILE = "saved/SASRec_CBM-ml-1mm_sigmoid_New.pth"
BASE_MODEL_PATH = "./saved/sasrec_ml-1m.pth"

K = 10
NDCG_KEY = f"ndcg@{K}"

MAX_NDCG_LOSS = 10
MAX_BALANCE_GAP = 2.0

# Concepts to test as steering targets.
AMPLIFY_CONCEPTS = [
   "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Crime",
    'Documentary',
    'Fantasy',
    'Film-Noir',
    'Musical',
    'Mystery',
    'War',
    'Western'
]
SUPPRESS_CONCEPTS = [
    "Drama",
    "Sci-Fi",
    "Romance",
    "Thriller",
    "Horror",
    'Comedy'
]

# Intervention scale grid.
AMPLIFY_SCALES = np.round(np.arange(1.0, 10.25, 0.25), 2)
SUPPRESS_SCALES = np.round(np.arange(1.0, 0.0, -0.05), 2)


# ============================================================
# 2. LOAD MODEL AND DATA
# ============================================================

config, CBM_model, dataset, train_data, valid_data, test_data = load_data_and_model(
    config_dict={
        "data_path": DATA_PATH,
        "dataset": DATASET,
        "base_path": BASE_MODEL_PATH,
    },
    model_file=MODEL_FILE,
)

CBM_model.eval()


# ============================================================
# 3. EVALUATION CONFIG
# ============================================================

config["metrics"] = [
    "Recall",
    "NDCG",
    "MRR",
    "Hit",
    "ItemCoverage",
    "ItemCoverageN",
    "GiniIndex",
    "AveragePopularity",
    "TierExposure",
    "GenreExposure",
]

config["topk"] = [K]

config["valid_metric"] = f"NDCG@{K}"





# ============================================================
# 4. LOOKUPS REQUIRED BY CUSTOM EXPOSURE METRICS
# ============================================================

L = build_lookups(dataset, config["dataset"])

config["popularity_pool"] = list(L["pop_set"])
config["mid_pool"] = list(L["mid_set"])
config["niche_pool"] = list(L["niche_set"])
config["item_to_genres"] = dict(L["item_to_genres"])
config["genre_list"] = list(L["genre_concepts"])


# ============================================================
# 5. PATCH CBM FOR PAIRED CONCEPT INTERVENTIONS
# ============================================================

def patched_forward(self, item_seq, item_seq_len):
    """CBM forward pass with optional paired concept interventions."""

    h = self._encode(item_seq, item_seq_len)

    logits = self.concept_predictor(h)
    c_hat = torch.sigmoid(logits)
    z = self.latent_predictor(h)

    # Apply all requested concept interventions during evaluation.
    if getattr(self, "_interventions", None) and not self.training:
        c_hat = c_hat.clone()

        for idx, scale in self._interventions.items():
            c_hat[:, idx] *= scale

    bottleneck = torch.cat([c_hat, z], dim=-1)
    h_hat = self.reconstructor(bottleneck)

    return h, c_hat, z, h_hat




# ============================================================
# 6. RECBole EVALUATION
# ============================================================

def evaluate_with_recbole(model, config, train_data, eval_data):
    """Evaluate using a fresh Collector/Evaluator built from current config."""

    trainer = Trainer(config, model)
    trainer.eval_collector = Collector(config)
    trainer.evaluator = Evaluator(config)
    trainer.eval_collector.data_collect(train_data)

    return trainer.evaluate(
        eval_data,
        load_best_model=False,
        show_progress=False,
    )


# ============================================================
# 7. FIND THE ACTUAL GENRE-EXPOSURE RESULT KEYS
# ============================================================

def normalize_name(x):
    return "".join(ch for ch in str(x).lower() if ch.isalnum())


def build_genre_metric_keys(result, genres, k):
    """Map each genre name to the corresponding returned exposure metric key."""

    metric_keys = {}

    for genre in genres:
        genre_norm = normalize_name(genre)

        matches = [
            key
            for key in result.keys()
            if genre_norm in normalize_name(key) and f"@{k}" in str(key)
        ]

        if len(matches) != 1:
            raise KeyError(
                f"Could not uniquely find an exposure metric for genre "
                f"'{genre}'. Matches: {matches}\n"
                f"Available keys: {list(result.keys())}"
            )

        metric_keys[genre] = matches[0]

    return metric_keys


# ============================================================
# 8. BASELINE
# ============================================================

CBM_model.forward = types.MethodType(patched_forward, CBM_model)

CBM_model._interventions = {}
CBM_model.eval()

print("\nComputing validation baseline...")

valid_baseline = evaluate_with_recbole(
    CBM_model,
    config,
    train_data,
    valid_data,
)

print(f"Baseline NDCG@{K}: {valid_baseline[NDCG_KEY]:.6f}")
print("Returned metrics:")
for key in valid_baseline.keys():
    print(f"  {key}")


# Use ALL genres when measuring non-target spillover.
ALL_GENRES = list(config["genre_list"])

genre_metric_keys = build_genre_metric_keys(
    valid_baseline,
    ALL_GENRES,
    K,
)

print("\nDetected genre exposure metrics:")
for genre, key in genre_metric_keys.items():
    print(f"  {genre:20s} -> {key}")


# Check that all steering targets exist in the CBM bottleneck.
for concept in AMPLIFY_CONCEPTS + SUPPRESS_CONCEPTS:
    if concept not in CBM_model.concept_names:
        raise ValueError(
            f"Concept '{concept}' is not in CBM_model.concept_names."
        )


# ============================================================
# 9. GROUP STEERING EVALUATION
# ============================================================

def evaluate_steering_group(
    group_name,
    amplify_concepts,
    suppress_concepts,
    metric_keys,
    baseline_result,
):
    """
    For each ordered Amplify/Suppress concept pair:
      1. Search all intervention-scale combinations.
      2. Keep only configurations satisfying:
           NDCG loss <= MAX_NDCG_LOSS
           amplify delta > 0
           suppress delta < 0
           balance gap < MAX_BALANCE_GAP
      3. Compute:
           Target-Non-target
             = mean(|target deltas|) - max(|non-target delta|)
      4. Select the best valid scale configuration for the pair.
      5. Average the best pair scores to obtain one group score.
    """

    pair_results = []

    pairs = [
        (amp, sup)
        for amp, sup in product(amplify_concepts, suppress_concepts)
        if amp != sup
    ]

    print(f"\nSearching {len(pairs)} {group_name} concept pairs...")

    for pair_number, (amp_concept, sup_concept) in enumerate(pairs, 1):
        print(
            f"\n[{pair_number}/{len(pairs)}] "
            f"Amplify {amp_concept} / Suppress {sup_concept}"
        )

        amp_idx = CBM_model.concept_names.index(amp_concept)
        sup_idx = CBM_model.concept_names.index(sup_concept)

        amp_key = metric_keys[amp_concept]
        sup_key = metric_keys[sup_concept]

        best = None

        for amp_scale in AMPLIFY_SCALES:
            for sup_scale in SUPPRESS_SCALES:
                CBM_model._interventions = {
                    amp_idx: float(amp_scale),
                    sup_idx: float(sup_scale),
                }

                steered = evaluate_with_recbole(
                    CBM_model,
                    config,
                    train_data,
                    valid_data,
                )

                # Target exposure changes in percentage points.
                amp_delta = 100.0 * (
                    steered[amp_key] - baseline_result[amp_key]
                )
                sup_delta = 100.0 * (
                    steered[sup_key] - baseline_result[sup_key]
                )

                # Require steering in the intended directions.
                if amp_delta <= 0 or sup_delta >= 0:
                    continue

                # Target balance constraint.
                balance_gap = abs(abs(amp_delta) - abs(sup_delta))
                if balance_gap >= MAX_BALANCE_GAP:
                    continue

                # Recommendation-quality loss.
                baseline_ndcg = baseline_result[NDCG_KEY]
                steered_ndcg = steered[NDCG_KEY]

                ndcg_change = 100.0 * (
                    (steered_ndcg - baseline_ndcg)
                    / (baseline_ndcg + 1e-12)
                )
                ndcg_loss = max(0.0, -ndcg_change)

                if ndcg_loss > MAX_NDCG_LOSS:
                    continue

                # Intended target effect.
                mean_target = (amp_delta + abs(sup_delta)) / 2.0

                # Largest unintended change among ALL other genres.
                non_target_deltas = []

                for concept, key in metric_keys.items():
                    if concept in {amp_concept, sup_concept}:
                        continue

                    delta = 100.0 * (
                        steered[key] - baseline_result[key]
                    )
                    non_target_deltas.append(abs(delta))

                max_non_target = (
                    max(non_target_deltas)
                    if non_target_deltas
                    else 0.0
                )

                # Paper metric.
                steering_score = mean_target - max_non_target

                row = {
                    "Group": group_name,
                    "Amplify": amp_concept,
                    "Suppress": sup_concept,
                    "Amplify Scale": float(amp_scale),
                    "Suppress Scale": float(sup_scale),
                    "Amplify Delta (pp)": amp_delta,
                    "Suppress Delta (pp)": sup_delta,
                    "Mean Target (pp)": mean_target,
                    "Max Non-target (pp)": max_non_target,
                    "Balance Gap": balance_gap,
                    "NDCG Loss (%)": ndcg_loss,
                    "Target-Non-target": steering_score,
                }

                if (
                    best is None
                    or steering_score > best["Target-Non-target"]
                ):
                    best = row

        if best is not None:
            pair_results.append(best)
            print(
                f"  Best score: {best['Target-Non-target']:.3f} | "
                f"amp={best['Amplify Scale']:.2f}, "
                f"sup={best['Suppress Scale']:.2f}, "
                f"NDCG loss={best['NDCG Loss (%)']:.2f}%"
            )
        else:
            print("  No valid configuration for this pair.")

    CBM_model._interventions = {}

    pair_df = pd.DataFrame(pair_results)

    if pair_df.empty:
        print(f"\nNo valid configurations for {group_name}.")
        return pair_df, None

    final_score = pair_df["Target-Non-target"].mean()

    print("\n========================================")
    print(
        f"Final {group_name} Target-Non-target score: "
        f"{final_score:.3f}"
    )
    print(f"Valid pairs: {len(pair_df)}/{len(pairs)}")
    print("========================================")

    return pair_df, final_score


# ============================================================
# 10. RUN GENRE EXPERIMENT
# ============================================================

genre_df, genre_score = evaluate_steering_group(
    group_name="Genre",
    amplify_concepts=AMPLIFY_CONCEPTS,
    suppress_concepts=SUPPRESS_CONCEPTS,
    metric_keys=genre_metric_keys,
    baseline_result=valid_baseline,
)


# ============================================================
# 11. SAVE RESULTS
# ============================================================

if not genre_df.empty:
    columns_to_show = [
        "Amplify",
        "Suppress",
        "Amplify Scale",
        "Suppress Scale",
        "Amplify Delta (pp)",
        "Suppress Delta (pp)",
        "Mean Target (pp)",
        "Max Non-target (pp)",
        "Balance Gap",
        "NDCG Loss (%)",
        "Target-Non-target",
    ]

    print("\nBest configuration for each concept pair:")
    print(genre_df[columns_to_show].to_string(index=False))

    output_file = "genre_steering_results.csv"
    genre_df.to_csv(output_file, index=False)

    print(f"\nSaved pair-level results to: {output_file}")
    print(f"FINAL GENRE SCORE = {genre_score:.3f}")
