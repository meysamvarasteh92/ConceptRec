"""Paired concept steering for a trained RecBole LightGCN_CBM.

Adapted from the SASRec_CBM script supplied by the user.

- For LightGCN, forward takes user IDs (not item sequences).
- Uses graph-propagated user embeddings for concept prediction.
- Uses LightGCN_CBM.full_sort_predict() for scoring against propagated items.
- Preserves validation scale selection, NDCG loss constraint, test evaluation,
  original (legacy) Push-Pull score by default, and CSV column names.

Requires the project's custom RecBole and build_lookups()/exposure metrics.
Edit the three paths below, then run: python lightgcn_cbm_paired_steering_final.py
"""

import types
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from recbole.quick_start import load_data_and_model
from recbole.trainer import Trainer
from recbole.evaluator import Collector, Evaluator
from recbole.utils.build_concepts import build_lookups


# ============================================================
# 1. SETTINGS - EDIT THESE PATHS
# ============================================================
DATASET = "ml-1m"             # beeradvocate, ml-1m, ml-1mm, lastfm
DATA_PATH = "dataset/ml-1m"   # same data path used during training
MODEL_FILE = "saved/LightGCN_CBM-Oct-18-2026_09-28-40.pth"
##LightGCN_CBM-Oct-18-2026_09-28-40.pth (ml-1m)
BASE_MODEL_PATH = "saved/lightgcn_ml-1m.pth"

K = 10
NDCG_KEY = f"ndcg@{K}"
MAX_NDCG_LOSS = 10.0
MAX_BALANCE_GAP = 2.0          # Original script did not enforce this
ENFORCE_BALANCE_GAP = False

AMPLIFY_SCALES = np.round(np.arange(1.0, 10.25, 0.25), 2)
SUPPRESS_SCALES = np.round(np.arange(1.0, 0.0, -0.05), 2)

# False exactly preserves the original SASRec steering_stats() definition:
#     mean_target = (amp_delta + abs(sup_delta)) / 2.
# True uses the direction-aware version:
#     mean_target = (amp_delta - sup_delta) / 2.
# If you change this, also change it for SASRec before comparing results.
SIGNED_SUPPRESSION = False
REQUIRE_TARGET_DIRECTIONS = False   # Original script constrained only NDCG

CONCEPT_GROUPS = (
    ["popularity", "abv", "style"] if DATASET == "beeradvocate"
    #else ["popularity", "era", "genre"]
        else [ "era", "genre"]

)
OUTPUT_PREFIX = "lightgcn"


# ============================================================
# 2. LOAD TRAINED LIGHTGCN-CBM AND RECBole DATA
# ============================================================
def load_model_and_data():
    for filepath in [MODEL_FILE, BASE_MODEL_PATH]:
        if not Path(filepath).is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {filepath}")

    cfg, model, ds, train, valid, test = load_data_and_model(
        config_dict={
            "data_path": DATA_PATH,
            "dataset": DATASET,
            "base_path": BASE_MODEL_PATH,
        },
        model_file=MODEL_FILE,
    )
    required = (
        "_ensure_propagated", "USER_ID", "concept_predictor",
        "latent_predictor", "reconstructor", "concept_names",
    )
    missing = [name for name in required if not hasattr(model, name)]
    if missing:
        raise TypeError(
            f"Loaded model lacks LightGCN_CBM attributes {missing}. "
            "Check MODEL_FILE and model registration."
        )
    model.eval()
    if hasattr(model, "invalidate_cache"):
        model.invalidate_cache()  # discard cached tensors on another device
    return cfg, model, ds, train, valid, test


# ============================================================
# 3. LOOKUPS, METRICS AND STEERING GROUPS
# ============================================================
def build_groups(config, model, dataset):
    config["metrics"] = [
        "Recall", "NDCG", "MRR", "Hit", "ItemCoverage", "ItemCoverageN",
        "GiniIndex", "AveragePopularity", "TierExposure",
    ]
    if DATASET == "beeradvocate":
        config["metrics"] += ["ABVExposure", "StyleExposure"]
    else:
        config["metrics"] += ["GenreExposure", "YearExposure"]
    config["topk"] = [K]
    config["valid_metric"] = f"NDCG@{K}"

    lookup = build_lookups(dataset, config["dataset"])
    for lookup_key, config_key in (
        ("pop_set", "popularity_pool"),
        ("mid_set", "mid_pool"),
        ("niche_set", "niche_pool"),
    ):
        config[config_key] = list(lookup[lookup_key])

    if DATASET == "beeradvocate":
        config["item_to_abv"] = dict(lookup["item_to_abv"])
        config["abv_list"] = list(dict.fromkeys(lookup["item_to_abv"].values()))
        config["item_to_style"] = dict(lookup["item_to_style"])
        config["style_list"] = list(lookup["style_concepts"])
        genre_all = [
            name for name in config["style_list"] if name in model.concept_names
        ]
        genre_suppress = ["Belgian Abbey & Strong Ale", "IPA", "Stout"]
    elif DATASET in {"ml-1m", "ml-1mm", "lastfm"}:
        config["item_to_genres"] = dict(lookup["item_to_genres"])
        config["genre_list"] = list(lookup["genre_concepts"])
        config["item_to_era"] = dict(lookup["item_to_era"])
        genre_all = [
            name for name in config["genre_list"] if name in model.concept_names
        ]
        genre_suppress = (
            ["Electronic", "Rock"] if DATASET == "lastfm" else
            ["Drama", "Sci-Fi", "Romance", "Thriller", "Horror", "Comedy"]
        )
    else:
        raise ValueError(f"Unsupported DATASET: {DATASET!r}")

    missing = [name for name in genre_suppress if name not in genre_all]
    if missing:
        raise ValueError(
            f"Suppression concepts not in model.concept_names: {missing}"
        )
    genre_amplify = [name for name in genre_all if name not in genre_suppress]
    if not genre_amplify:
        raise ValueError("No eligible concepts available for amplification")

    groups = {
        "popularity": {
            "amplify": ["mid", "niche"],
            "suppress": ["popularity"],
            "all_concepts": ["popularity", "mid", "niche"],
        },
        "era": {
            "amplify": ["classic", "retro"],
            "suppress": ["modern"],
            "all_concepts": ["classic", "retro", "modern"],
            #"contemporary" for lastfms
        },
        "genre": {
            "amplify": genre_amplify,
            "suppress": genre_suppress,
            "all_concepts": genre_all,
        },
        "style": {
            "amplify": genre_amplify,
            "suppress": genre_suppress,
            "all_concepts": genre_all,
        },
        "abv": {
            "amplify": ["abv_high", "abv_med"],
            "suppress": ["abv_low"],
            "all_concepts": ["abv_low", "abv_med", "abv_high"],
        },
    }
    for group, settings in groups.items():
        settings["output_file"] = (
            f"{OUTPUT_PREFIX}_{group}_steering_results_{DATASET}.csv"
        )
    for group in CONCEPT_GROUPS:
        for name in groups[group]["all_concepts"]:
            if name not in model.concept_names:
                raise ValueError(
                    f"{group}: concept {name!r} is absent from model.concept_names"
                )
    return groups


# ============================================================
# 4. LIGHTGCN FORWARD ADAPTER (INSTEAD OF SASREC ITEM_SEQ)
# ============================================================
def patched_lightgcn_forward(self, user):
    """User ID -> propagated embedding -> bottleneck -> reconstruction.

    self.full_sort_predict() continues using propagated item embeddings.
    """
    propagated_user_e, _ = self._ensure_propagated()
    h = propagated_user_e[user]
    c_hat = torch.sigmoid(self.concept_predictor(h))
    z = self.latent_predictor(h)

    if not self.training and getattr(self, "_interventions", None):
        c_hat = c_hat.clone()
        for idx, scale in self._interventions.items():
            if not (0 <= idx < c_hat.shape[1]):
                raise IndexError(f"Invalid concept index {idx}")
            c_hat[:, idx] *= float(scale)

    h_hat = self.reconstructor(torch.cat([c_hat, z], dim=-1))
    return h, c_hat, z, h_hat


def install_steering_adapter(model):
    """Works with both older and newer LightGCN_CBM implementations."""
    model.eval()
    native_forward = model.forward
    if getattr(model, "steer_concept_idx", None) is not None:
        print("Disabling legacy single-concept steering for this experiment")
    if hasattr(model, "steer_concept_idx"):
        model.steer_concept_idx = None
    if hasattr(model, "steer_deltas"):
        model.steer_deltas = {}

    model._interventions = {}
    model.forward = types.MethodType(patched_lightgcn_forward, model)

    # Ensure our replacement agrees with the unsteered model.
    if model.n_users > 1:
        device = next(model.parameters()).device
        probe = torch.tensor([1], dtype=torch.long, device=device)
        with torch.no_grad():
            original = native_forward(probe)[-1]
            modified = model.forward(probe)[-1]
            if not torch.allclose(original, modified, atol=1e-5, rtol=1e-5):
                raise RuntimeError(
                    "Unsteered LightGCN forward differs after patching. "
                    "Check that no other steering is active."
                )
            model._interventions = {0: 0.0}
            try:
                steered = model.forward(probe)[-1]
                delta = (steered - modified).abs().max().item()
                print(f"Steering smoke check (max |delta h_hat|): {delta:.6g}")
                if delta < 1e-10:
                    print("Warning: first concept caused no change for probe user")
            finally:
                model._interventions = {}


# ============================================================
# 5. METRIC KEYS, STATISTICS, CONSTRAINTS
# ============================================================
def build_metric_keys(result, group_name, concepts):
    if group_name == "popularity":
        keys = {
            "popularity": f"pop_rate@{K}",
            "mid": f"mid_rate@{K}",
            "niche": f"niche_rate@{K}",
        }
    elif group_name == "era":
        keys = {c: f"era_{c}_rate@{K}" for c in concepts}
    elif group_name == "genre":
        keys = {c: f"genre_{c}_rate@{K}" for c in concepts}
    elif group_name == "abv":
        keys = {c: f"abv_{c}_rate@{K}" for c in concepts}
    elif group_name == "style":
        # Matches the StyleExposure metric implementation in metrics.py.
        keys = {
            c: f"style_{str(c).replace(' ', '_').replace('/', '-')}_rate@{K}"
            for c in concepts
        }
    else:
        raise ValueError(f"Unknown group: {group_name}")

    missing = [(c, key) for c, key in keys.items() if key not in result]
    if missing:
        raise KeyError(
            f"Missing exposure metrics for {group_name}: {missing}. "
            f"Available metric keys: {list(result.keys())}"
        )
    return keys


def steering_stats(result, baseline, amp_concept, sup_concept, metric_keys):
    amp_key, sup_key = metric_keys[amp_concept], metric_keys[sup_concept]
    amp_delta = 100.0 * (result[amp_key] - baseline[amp_key])
    sup_delta = 100.0 * (result[sup_key] - baseline[sup_key])
    balance_gap = abs(abs(amp_delta) - abs(sup_delta))

    ndcg_change = 100.0 * (
        result[NDCG_KEY] - baseline[NDCG_KEY]
    ) / (baseline[NDCG_KEY] + 1e-12)
    ndcg_loss = max(0.0, -ndcg_change)

    # EXACT original SASRec definition when SIGNED_SUPPRESSION=False.
    suppression_reward = -sup_delta if SIGNED_SUPPRESSION else abs(sup_delta)
    mean_target = (amp_delta + suppression_reward) / 2.0
    nontarget = [
        abs(100.0 * (result[key] - baseline[key]))
        for concept, key in metric_keys.items()
        if concept not in (amp_concept, sup_concept)
    ]
    max_nontarget = max(nontarget, default=0.0)
    return {
        "Amplify Delta (pp)": amp_delta,
        "Suppress Delta (pp)": sup_delta,
        "Mean Target (pp)": mean_target,
        "Max Non-target (pp)": max_nontarget,
        "Balance Gap": balance_gap,
        "NDCG Loss (%)": ndcg_loss,
        "Target-Non-target": mean_target - max_nontarget,
    }


def valid_configuration(stats):
    if stats["NDCG Loss (%)"] > MAX_NDCG_LOSS:
        return False
    if ENFORCE_BALANCE_GAP and stats["Balance Gap"] > MAX_BALANCE_GAP:
        return False
    if REQUIRE_TARGET_DIRECTIONS:
        return (
            stats["Amplify Delta (pp)"] > 0
            and stats["Suppress Delta (pp)"] < 0
        )
    return True


# ============================================================
# 6. RECBole TRAINER + BUFFER HANDLING
# ============================================================
def make_evaluator(config, model, train_data):
    trainer = Trainer(config, model)
    trainer.eval_collector = Collector(config)
    trainer.evaluator = Evaluator(config)
    trainer.eval_collector.data_collect(train_data)
    return trainer


def evaluate(trainer, model, data):
    model.eval()
    try:
        return trainer.evaluate(data, load_best_model=False, show_progress=False)
    finally:
        # The custom LightGCN-CBM may accumulate concept tensors per call.
        # They are not needed for the steering exposure metrics.
        if hasattr(model, "reset_eval_buffers"):
            model.reset_eval_buffers()
        elif hasattr(model, "_eval_c_hat_buffer"):
            model._eval_c_hat_buffer = []
            model._eval_gt_buffer = []


# ============================================================
# 7. GRID SEARCH AND TEST EVALUATION
# ============================================================
def find_best_scales(
    model, trainer, valid_data, valid_baseline,
    amp_concept, sup_concept, metric_keys,
):
    amp_idx = model.concept_names.index(amp_concept)
    sup_idx = model.concept_names.index(sup_concept)
    best = None
    try:
        for amp_scale in AMPLIFY_SCALES:
            for sup_scale in SUPPRESS_SCALES:
                model._interventions = {
                    amp_idx: float(amp_scale),
                    sup_idx: float(sup_scale),
                }
                result = evaluate(trainer, model, valid_data)
                stats = steering_stats(
                    result, valid_baseline, amp_concept, sup_concept, metric_keys
                )
                if not valid_configuration(stats):
                    continue
                if best is None or stats["Target-Non-target"] > best["Target-Non-target"]:
                    best = {
                        "Amplify Scale": float(amp_scale),
                        "Suppress Scale": float(sup_scale),
                        **stats,
                    }
    finally:
        model._interventions = {}
    return best


def evaluate_on_test(
    model, trainer, test_data, test_baseline,
    amp_concept, sup_concept, amp_scale, sup_scale, metric_keys,
):
    amp_idx = model.concept_names.index(amp_concept)
    sup_idx = model.concept_names.index(sup_concept)
    model._interventions = {
        amp_idx: float(amp_scale),
        sup_idx: float(sup_scale),
    }
    try:
        result = evaluate(trainer, model, test_data)
        return steering_stats(
            result, test_baseline, amp_concept, sup_concept, metric_keys
        )
    finally:
        model._interventions = {}


# ============================================================
# 8. SAVE RESULTS (MATCHES SASREC CSV COLUMNS)
# ============================================================
def run_group(
    name, group, model, trainer, valid_data, test_data,
    valid_baseline, test_baseline,
):
    metric_keys = build_metric_keys(valid_baseline, name, group["all_concepts"])
    pairs = list(product(group["amplify"], group["suppress"]))
    print("\n" + "=" * 60)
    print(f"{name.upper()} STEERING (LightGCN) - {len(pairs)} concept pairs")
    print("=" * 60)
    rows = []
    for i, (amp_concept, sup_concept) in enumerate(pairs, start=1):
        print(
            f"[{i}/{len(pairs)}] Amplify {amp_concept} / Suppress {sup_concept}"
        )
        best = find_best_scales(
            model, trainer, valid_data, valid_baseline,
            amp_concept, sup_concept, metric_keys,
        )
        if best is None:
            rows.append({
                "Group": name,
                "Amplify": amp_concept,
                "Suppress": sup_concept,
                "Valid": False,
                "Amplify Scale": np.nan,
                "Suppress Scale": np.nan,
                "Val Target-Non-target": 0.0,
                **{f"Test {key}": 0.0 for key in (
                    "Amplify Delta (pp)", "Suppress Delta (pp)",
                    "Mean Target (pp)", "Max Non-target (pp)",
                    "Balance Gap", "NDCG Loss (%)", "Target-Non-target",
                )},
            })
            print("  No configuration met NDCG constraint; counted as 0")
            continue

        stats = evaluate_on_test(
            model, trainer, test_data, test_baseline,
            amp_concept, sup_concept,
            best["Amplify Scale"], best["Suppress Scale"], metric_keys,
        )
        rows.append({
            "Group": name,
            "Amplify": amp_concept,
            "Suppress": sup_concept,
            "Valid": True,
            "Amplify Scale": best["Amplify Scale"],
            "Suppress Scale": best["Suppress Scale"],
            "Val Target-Non-target": best["Target-Non-target"],
            **{f"Test {key}": value for key, value in stats.items()},
        })
        print(
            f"  best amp={best['Amplify Scale']:.2f}, "
            f"sup={best['Suppress Scale']:.2f}, "
            f"test score={stats['Target-Non-target']:.3f}"
        )

    frame = pd.DataFrame(rows)
    frame.to_csv(group["output_file"], index=False)
    valid_count = int(frame["Valid"].sum()) if not frame.empty else 0
    test_score = float(frame["Test Target-Non-target"].mean()) if not frame.empty else None
    print(f"FINAL {name.upper()} TEST SCORE: {test_score}")
    print(f"Valid pairs: {valid_count}/{len(frame)}")
    print(f"Saved: {group['output_file']}")
    return frame, {
        "Group": name,
        "Test Score": test_score,
        "Valid Pairs": valid_count,
        "Failed Pairs": len(frame) - valid_count,
        "Total Pairs": len(pairs),
    }


def main():
    config, model, dataset, train_data, valid_data, test_data = load_model_and_data()
    groups = build_groups(config, model, dataset)
    install_steering_adapter(model)
    trainer = make_evaluator(config, model, train_data)

    model._interventions = {}
    print(f"\nComputing baseline on validation ({DATASET})...")
    valid_baseline = evaluate(trainer, model, valid_data)
    print(f"Validation NDCG@{K}: {valid_baseline[NDCG_KEY]:.6f}")
    print("Computing baseline on test...")
    test_baseline = evaluate(trainer, model, test_data)
    print(f"Test NDCG@{K}: {test_baseline[NDCG_KEY]:.6f}")

    result_frames, summaries = [], []
    for group_name in CONCEPT_GROUPS:
        frame, summary = run_group(
            group_name, groups[group_name], model, trainer,
            valid_data, test_data, valid_baseline, test_baseline,
        )
        result_frames.append(frame)
        summaries.append(summary)

    summary_frame = pd.DataFrame(summaries)
    summary_file = f"{OUTPUT_PREFIX}_steering_summary_{DATASET}.csv"
    summary_frame.to_csv(summary_file, index=False)
    print("\nFINAL SUMMARY\n", summary_frame.to_string(index=False))
    print(f"Saved: {summary_file}")

    all_frame = pd.concat(result_frames, ignore_index=True)
    combined_file = f"{OUTPUT_PREFIX}_all_steering_results_{DATASET}.csv"
    all_frame.to_csv(combined_file, index=False)
    print(f"Saved: {combined_file}")


if __name__ == "__main__":
    main()
