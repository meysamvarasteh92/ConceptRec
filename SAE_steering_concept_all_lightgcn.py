"""Paired concept-exposure steering for a pretrained LightGCN_TopK_SAE.

This is NOT a direct concept-bottleneck intervention: SAE neurons have no
semantic names. We first align learned SAE neurons with named concepts using
TRAINING-user concept targets (no validation or test labels for alignment),
then multiply selected SAE activations on the user side only. We leave the item
SAE representations unchanged. We retain the original paired-steering search,
custom RecBole exposure metrics, NDCG constraint and CSV result columns.

Install the original LightGCN_TopK_SAE model and custom RecBole metric code,
then edit DATASET, DATA_PATH, MODEL_FILE, BASE_MODEL_PATH below.
"""

from itertools import product
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch

from recbole.quick_start import load_data_and_model
from recbole.trainer import Trainer
from recbole.evaluator import Collector, Evaluator
from recbole.utils.build_concepts import build_lookups, build_cache, seq_to_concepts


# ---------------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------------
DATASET = "beeradvocate"  # or: beeradvocate, ml-1mm, lastfm
DATA_PATH = "dataset/beeradvocate"  # or: dataset/ml-1m, dataset/ml-1mm, dataset/lastfm
MODEL_FILE = "saved/LightGCN_TopK_SAE-Oct-19-2026_09-10-41.pth"
## LightGCN_TopK_SAE-Oct-18-2026_15-33-12.pth (ml-1m)
## LightGCN_TopK_SAE-Oct-19-2026_09-10-41.pth (beeradvocate)
BASE_MODEL_PATH = "saved/LightGCN-Oct-19-2026_08-45-31.pth"

K = 10
NDCG_KEY = f"ndcg@{K}"
MAX_NDCG_LOSS = 10.0
MAX_BALANCE_GAP = 2.0
ENFORCE_BALANCE_GAP = False

# To reproduce the *old* SASRec score use False. The direction-aware score
# should be True for a faithful push/pull interpretation; change BOTH baselines.
SIGNED_SUPPRESSION = False
REQUIRE_TARGET_DIRECTIONS = False

AMPLIFY_SCALES = np.round(np.arange(1.0, 10.25, 0.25), 2)
SUPPRESS_SCALES = np.round(np.arange(1.0, 0.0, -0.05), 2)

# Each named concept is assigned this many DISTINCT SAE neurons based on
# positive train-set Pearson correlation with the user-level concept target.
# One neuron yields the closest single-unit experiment, but a sparse Top-K SAE
# is often inactive for a given user, so the intervention may have no effect.
NEURONS_PER_CONCEPT = 1
MIN_ACTIVE_USERS = 10
MIN_CORRELATION = 0.0  # Increase to reject weak semantic alignments
CALIBRATION_CHUNK = 256

CONCEPT_GROUPS = (
    ["popularity", "abv", "style"] if DATASET == "beeradvocate"
    else ["popularity","era", "genre"]
)
OUTPUT_PREFIX = "lightgcn_topk_sae"


# ---------------------------------------------------------------------------
# LOADING
# ---------------------------------------------------------------------------
def load_model_and_data():
    for name in (MODEL_FILE, BASE_MODEL_PATH):
        if not Path(name).is_file():
            raise FileNotFoundError(f"Missing checkpoint: {name}")

    # The custom model class must already be discoverable by RecBole.
    cfg, model, ds, train, valid, test = load_data_and_model(
        config_dict={
            "data_path": DATA_PATH,
            "dataset": DATASET,
            "base_path": BASE_MODEL_PATH,
        },
        model_file=MODEL_FILE,
    )
    required = ["sae", "_ensure_propagated", "set_interventions", "clear_steering", "USER_ID"]
    missing = [name for name in required if not hasattr(model, name)]
    if missing:
        raise TypeError(
            f"Loaded model is not the compatible LightGCN_TopK_SAE: missing {missing}. "
            "Make sure MODEL_FILE is an SAE checkpoint, not the CBM checkpoint."
        )
    model.eval()
    model.clear_steering()
    if hasattr(model, "invalidate_cache"):
        model.invalidate_cache()
    return cfg, model, ds, train, valid, test


# ---------------------------------------------------------------------------
# CONFIGURE EXPOSURE METRICS / GROUPS
# ---------------------------------------------------------------------------
def build_groups(config, dataset, target_names):
    config["metrics"] = [
        "Recall", "NDCG", "MRR", "Hit", "ItemCoverage", "ItemCoverageN",
        "GiniIndex", "AveragePopularity", "TierExposure",
    ]
    config["metrics"] += (
        ["ABVExposure", "StyleExposure"] if DATASET == "beeradvocate"
        else ["GenreExposure", "YearExposure"]
    )
    config["topk"] = [K]
    config["valid_metric"] = f"NDCG@{K}"

    lookups = build_lookups(dataset, config["dataset"])
    config["popularity_pool"] = list(lookups["pop_set"])
    config["mid_pool"] = list(lookups["mid_set"])
    config["niche_pool"] = list(lookups["niche_set"])
    if DATASET == "beeradvocate":
        config["item_to_abv"] = dict(lookups["item_to_abv"])
        config["abv_list"] = list(dict.fromkeys(lookups["item_to_abv"].values()))
        config["item_to_style"] = dict(lookups["item_to_style"])
        config["style_list"] = list(lookups["style_concepts"])
        genre_all = [c for c in config["style_list"] if c in target_names]
        suppress = ["Belgian Abbey & Strong Ale", "IPA", "Stout"]
    else:
        config["item_to_genres"] = dict(lookups["item_to_genres"])
        config["genre_list"] = list(lookups["genre_concepts"])
        config["item_to_era"] = dict(lookups["item_to_era"])
        genre_all = [c for c in config["genre_list"] if c in target_names]
        suppress = (
            ["Electronic", "Rock"] if DATASET == "lastfm"
            else ["Drama", "Sci-Fi", "Romance", "Thriller", "Horror", "Comedy"]
        )

    unknown = [c for c in suppress if c not in genre_all]
    if unknown:
        raise ValueError(f"Missing genre/style targets: {unknown}. Available: {genre_all}")

    amplify = [c for c in genre_all if c not in suppress]
    groups = {
        "popularity": {
            "amplify": ["mid", "niche"],
            "suppress": ["popularity"],
            "all_concepts": ["popularity", "mid", "niche"],
        },
        "era": {
            "amplify": ["classic", "retro"],
            "suppress": ["modern"],
            # 'contemporary' may be a metric category even though no model
            # neuron / training target has that name.
            "all_concepts": ["classic", "retro", "modern", "contemporary"],
        },
        "genre": {
            "amplify": amplify, "suppress": suppress, "all_concepts": genre_all,
        },
        "style": {
            "amplify": amplify, "suppress": suppress, "all_concepts": genre_all,
        },
        "abv": {
            "amplify": ["abv_high", "abv_med"],
            "suppress": ["abv_low"],
            "all_concepts": ["abv_low", "abv_med", "abv_high"],
        },
    }
    for group_name in CONCEPT_GROUPS:
        settings = groups[group_name]
        for concept in settings["amplify"] + settings["suppress"]:
            if concept not in target_names:
                raise ValueError(
                    f"Cannot map steering target {concept!r} in group {group_name}: "
                    f"absent from build_cache concept names {target_names}"
                )
        settings["output_file"] = (
            f"{OUTPUT_PREFIX}_{group_name}_steering_results_{DATASET}.csv"
        )
    return groups


# ---------------------------------------------------------------------------
# SUPERVISED POST-HOC CONCEPT -> SAE FEATURE ALIGNMENT (TRAIN USERS ONLY)
# ---------------------------------------------------------------------------
def get_train_user_concepts(config, train_dataset, model):
    # Same user training-history target construction as LightGCN_CBM.
    cache, _ = build_cache(config["dataset"], train_dataset)
    concept_names = list(cache["concept_names"])
    item_concepts = torch.as_tensor(cache["item_concepts"], dtype=torch.float32)
    if item_concepts.shape[0] != model.n_items:
        raise ValueError("build_cache item IDs do not match LightGCN item vocabulary")

    csr = train_dataset.inter_matrix(form="csr")
    if csr.shape != (model.n_users, model.n_items):
        raise ValueError("Train interaction matrix shape doesn't match the SAE model")

    indptr, indices = csr.indptr, csr.indices
    counts = indptr[1:] - indptr[:-1]
    targets = torch.zeros((model.n_users, len(concept_names)), dtype=torch.float32)
    chunk = 128
    for start in range(0, model.n_users, chunk):
        end = min(start + chunk, model.n_users)
        width = int(counts[start:end].max())
        if width <= 0:
            continue
        seq = torch.zeros((end - start, width), dtype=torch.long)
        for row, uid in enumerate(range(start, end)):
            ids = indices[indptr[uid]:indptr[uid + 1]]
            if len(ids):
                seq[row, :len(ids)] = torch.as_tensor(ids, dtype=torch.long)
        targets[start:end] = seq_to_concepts(
            seq, item_concepts, int(cache["n_concepts"]), int(cache["n_genres"])
        )
    valid_users = torch.from_numpy(np.where(counts > 0)[0]).long()
    return concept_names, targets, valid_users


@torch.no_grad()
def align_sae_neurons(model, targets, user_ids, concepts):
    """Map named user concepts to DISTINCT SAE units by training-only correlation.

    Returns {name: [index,...]}, {name: [(neuron, rho, active_users), ...]}.
    Positive correlation gives a *candidate* association, not a guarantee that
    increasing a neuron will causally increase target exposure.
    """
    model.eval()
    model.clear_steering()
    device = next(model.parameters()).device
    prop_users, _ = model._ensure_propagated()
    n_features = int(model.sae.latent_dim)
    C = targets.shape[1]
    n = 0
    sum_z = torch.zeros(n_features, device=device, dtype=torch.float64)
    sum_z2 = torch.zeros_like(sum_z)
    sum_y = torch.zeros(C, device=device, dtype=torch.float64)
    sum_y2 = torch.zeros_like(sum_y)
    sum_zy = torch.zeros((n_features, C), device=device, dtype=torch.float64)
    active = torch.zeros(n_features, device=device, dtype=torch.int64)

    for user_chunk in user_ids.split(CALIBRATION_CHUNK):
        ids = user_chunk.to(device)
        x = model.sae.encode(prop_users[ids]).float()
        y = targets[user_chunk].to(device).float()
        n += len(user_chunk)
        sum_z += x.sum(0).double()
        sum_z2 += (x * x).sum(0).double()
        sum_y += y.sum(0).double()
        sum_y2 += (y * y).sum(0).double()
        sum_zy += (x.T @ y).double()
        active += (x > 0).sum(dim=0)

    # Pearson correlation = centered covariance divided by standard deviations.
    cov = sum_zy - sum_z[:, None] * sum_y[None, :] / n
    var_z = (sum_z2 - sum_z.square() / n).clamp_min(0)
    var_y = (sum_y2 - sum_y.square() / n).clamp_min(0)
    corr = cov / torch.sqrt(var_z[:, None] * var_y[None, :]).clamp_min(1e-12)
    corr = corr.nan_to_num(0).cpu().numpy()
    counts = active.cpu().numpy()
    eligible = counts >= MIN_ACTIVE_USERS
    corr[~eligible, :] = -np.inf

    # Allocate distinct neurons round-robin across concepts, so pairs never
    # accidentally multiply the same SAE feature by two different scales.
    mapping = {name: [] for name in concepts}
    diagnostics = {name: [] for name in concepts}
    used = set()
    for _ in range(NEURONS_PER_CONCEPT):
        for name, column in concepts.items():
            order = np.argsort(-corr[:, column], kind="stable")
            chosen = next(
                (int(i) for i in order
                 if i not in used and np.isfinite(corr[i, column])
                 and corr[i, column] > MIN_CORRELATION),
                None,
            )
            if chosen is None:
                raise RuntimeError(
                    f"No positively aligned SAE feature left for concept {name!r}. "
                    "Try NEURONS_PER_CONCEPT=1 or check concept targets/SAE sparsity."
                )
            mapping[name].append(chosen)
            diagnostics[name].append((chosen, float(corr[chosen, column]), int(counts[chosen])))
            used.add(chosen)
    print(f"Mapped {len(mapping)} concepts to {len(used)} distinct SAE neurons (train users={n}).")
    for name, rows in diagnostics.items():
        print(f"  {name}: " + ", ".join(
            f"#{idx} (rho={rho:.3f}, active={cnt}/{n})" for idx, rho, cnt in rows
        ))
    return mapping, diagnostics


def save_alignment(diagnostics):
    rows = []
    for concept, triples in diagnostics.items():
        for idx, rho, active in triples:
            rows.append({"Concept": concept, "Neuron": idx, "Train Pearson r": rho,
                         "Active train users": active})
    filename = f"{OUTPUT_PREFIX}_concept_neuron_mapping_{DATASET}.csv"
    pd.DataFrame(rows).to_csv(filename, index=False)
    print(f"Saved alignment: {filename}")


# ---------------------------------------------------------------------------
# METRIC KEYS AND SCORING: same evaluation protocol as CBM
# ---------------------------------------------------------------------------
def build_metric_keys(result, group_name, concepts):
    if group_name == "popularity":
        keys = {
            "popularity": f"pop_rate@{K}", "mid": f"mid_rate@{K}",
            "niche": f"niche_rate@{K}",
        }
    elif group_name == "era":
        keys = {c: f"era_{c}_rate@{K}" for c in concepts}
    elif group_name == "genre":
        keys = {c: f"genre_{c}_rate@{K}" for c in concepts}
    elif group_name == "abv":
        keys = {c: f"abv_{c}_rate@{K}" for c in concepts}
    elif group_name == "style":
        keys = {
            c: f"style_{str(c).replace(' ', '_').replace('/', '-')}_rate@{K}"
            for c in concepts
        }
    else:
        raise ValueError(f"Unknown group {group_name!r}")
    missing = [(c, key) for c, key in keys.items() if key not in result]
    if missing:
        raise KeyError(
            f"Missing {group_name} exposure metrics: {missing}. "
            f"Available keys: {list(result.keys())}"
        )
    return keys


def steering_stats(result, baseline, amp_concept, sup_concept, metric_keys):
    amp_delta = 100.0 * (result[metric_keys[amp_concept]] - baseline[metric_keys[amp_concept]])
    sup_delta = 100.0 * (result[metric_keys[sup_concept]] - baseline[metric_keys[sup_concept]])
    balance_gap = abs(abs(amp_delta) - abs(sup_delta))
    ndcg_change = 100.0 * (result[NDCG_KEY] - baseline[NDCG_KEY]) / (baseline[NDCG_KEY] + 1e-12)
    ndcg_loss = max(0.0, -ndcg_change)
    suppression_reward = -sup_delta if SIGNED_SUPPRESSION else abs(sup_delta)
    mean_target = (amp_delta + suppression_reward) / 2.0
    non_target_deltas = [
        abs(100.0 * (result[key] - baseline[key]))
        for concept, key in metric_keys.items()
        if concept not in (amp_concept, sup_concept)
    ]
    spillover = max(non_target_deltas, default=0.0)
    return {
        "Amplify Delta (pp)": amp_delta,
        "Suppress Delta (pp)": sup_delta,
        "Mean Target (pp)": mean_target,
        "Max Non-target (pp)": spillover,
        "Balance Gap": balance_gap,
        "NDCG Loss (%)": ndcg_loss,
        "Target-Non-target": mean_target - spillover,
    }


def valid_configuration(stats):
    if stats["NDCG Loss (%)"] > MAX_NDCG_LOSS:
        return False
    if ENFORCE_BALANCE_GAP and stats["Balance Gap"] > MAX_BALANCE_GAP:
        return False
    if REQUIRE_TARGET_DIRECTIONS:
        return stats["Amplify Delta (pp)"] > 0 and stats["Suppress Delta (pp)"] < 0
    return True


# ---------------------------------------------------------------------------
# RECBole EVALUATION
# ---------------------------------------------------------------------------
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
        if hasattr(model, "reset_eval_buffers"):
            model.reset_eval_buffers()
        elif hasattr(model, "_eval_c_hat_buffer"):
            model._eval_c_hat_buffer = []
            model._eval_gt_buffer = []


def make_interventions(mapping, amplify, suppress, amp_scale, sup_scale):
    amps = mapping[amplify]
    sups = mapping[suppress]
    if set(amps) & set(sups):
        raise RuntimeError(f"Overlapping SAE features for {amplify} and {suppress}")
    return {**{int(i): float(amp_scale) for i in amps},
            **{int(i): float(sup_scale) for i in sups}}


def find_best_scales(model, trainer, valid_data, valid_baseline, mapping,
                     amplify, suppress, metric_keys):
    best = None
    try:
        for amp_scale in AMPLIFY_SCALES:
            for sup_scale in SUPPRESS_SCALES:
                model.set_interventions(make_interventions(
                    mapping, amplify, suppress, amp_scale, sup_scale
                ))
                result = evaluate(trainer, model, valid_data)
                stats = steering_stats(result, valid_baseline, amplify, suppress, metric_keys)
                if not valid_configuration(stats):
                    continue
                if best is None or stats["Target-Non-target"] > best["Target-Non-target"]:
                    best = {"Amplify Scale": float(amp_scale),
                            "Suppress Scale": float(sup_scale), **stats}
    finally:
        model.clear_steering()
    return best


def evaluate_on_test(model, trainer, test_data, test_baseline, mapping,
                     amplify, suppress, amp_scale, sup_scale, metric_keys):
    model.set_interventions(make_interventions(
        mapping, amplify, suppress, amp_scale, sup_scale
    ))
    try:
        result = evaluate(trainer, model, test_data)
        return steering_stats(result, test_baseline, amplify, suppress, metric_keys)
    finally:
        model.clear_steering()


def run_group(name, group, model, trainer, valid_data, test_data,
              valid_baseline, test_baseline, mapping):
    metric_keys = build_metric_keys(valid_baseline, name, group["all_concepts"])
    pairs = list(product(group["amplify"], group["suppress"]))
    print("\n" + "=" * 60)
    print(f"{name.upper()} SAE STEERING: {len(pairs)} pairs")
    rows = []
    for index, (amplify, suppress) in enumerate(pairs, start=1):
        print(f"[{index}/{len(pairs)}] Amplify {amplify} / Suppress {suppress}")
        best = find_best_scales(model, trainer, valid_data, valid_baseline, mapping,
                                amplify, suppress, metric_keys)
        if best is None:
            rows.append({
                "Group": name, "Amplify": amplify, "Suppress": suppress,
                "Valid": False, "Amplify Scale": np.nan, "Suppress Scale": np.nan,
                "Val Target-Non-target": 0.0,
                **{f"Test {key}": 0.0 for key in (
                    "Amplify Delta (pp)", "Suppress Delta (pp)", "Mean Target (pp)",
                    "Max Non-target (pp)", "Balance Gap", "NDCG Loss (%)", "Target-Non-target"
                )}
            })
            continue
        stats = evaluate_on_test(
            model, trainer, test_data, test_baseline, mapping,
            amplify, suppress, best["Amplify Scale"], best["Suppress Scale"], metric_keys
        )
        rows.append({
            "Group": name, "Amplify": amplify, "Suppress": suppress,
            "Valid": True, "Amplify Scale": best["Amplify Scale"],
            "Suppress Scale": best["Suppress Scale"],
            "Val Target-Non-target": best["Target-Non-target"],
            **{f"Test {key}": value for key, value in stats.items()},
        })
        print(f"  amp={best['Amplify Scale']:.2f}, sup={best['Suppress Scale']:.2f}, "
              f"test score={stats['Target-Non-target']:.3f}")

    frame = pd.DataFrame(rows)
    frame.to_csv(group["output_file"], index=False)
    count_valid = int(frame["Valid"].sum()) if not frame.empty else 0
    group_score = float(frame["Test Target-Non-target"].mean()) if not frame.empty else None
    print(f"FINAL {name.upper()} score: {group_score}, valid pairs: {count_valid}/{len(frame)}")
    print(f"Saved: {group['output_file']}")
    return frame, {
        "Group": name, "Test Score": group_score,
        "Valid Pairs": count_valid, "Failed Pairs": len(frame) - count_valid,
        "Total Pairs": len(pairs),
    }


def main():
    config, model, dataset, train, valid, test = load_model_and_data()

    # Build targets exclusively from the TRAIN graph. Never use validation
    # or test interactions for selecting / correlating SAE features.
    names, targets, train_uids = get_train_user_concepts(config, train._dataset, model)
    groups = build_groups(config, dataset, set(names))
    steer_concepts = sorted({c for group in CONCEPT_GROUPS
                             for c in groups[group]["amplify"] + groups[group]["suppress"]})
    concept_cols = {c: names.index(c) for c in steer_concepts}
    mapping, diagnostics = align_sae_neurons(model, targets, train_uids, concept_cols)
    save_alignment(diagnostics)

    # Check that the live SAE is sensitive to interventions on at least one
    # active feature for a probe training user. Silent no-ops are reported.
    model.eval()
    model.clear_steering()
    with torch.no_grad():
        device = next(model.parameters()).device
        probe = train_uids[:1].to(device)
        if probe.numel():
            _, z0, h0 = model.forward(probe)
            active_features = (z0[0] > 0).nonzero().flatten()
            if active_features.numel():
                model.set_interventions({int(active_features[0]): 0.0})
                _, _, h1 = model.forward(probe)
                delta = (h0 - h1).abs().max().item()
                print(f"Neuron intervention smoke test max |delta h_hat|: {delta:.6g}")
                model.clear_steering()
                if delta < 1e-10:
                    raise RuntimeError("SAE steering had no effect on a known-active neuron")

    trainer = make_evaluator(config, model, train)
    print(f"\nUnsteered validation baseline ({DATASET})...")
    valid_baseline = evaluate(trainer, model, valid)
    print(f"Validation NDCG@{K}: {valid_baseline[NDCG_KEY]:.6f}")
    print("Unsteered test baseline...")
    test_baseline = evaluate(trainer, model, test)
    print(f"Test NDCG@{K}: {test_baseline[NDCG_KEY]:.6f}")

    frames, summaries = [], []
    for name in CONCEPT_GROUPS:
        frame, summary = run_group(
            name, groups[name], model, trainer, valid, test,
            valid_baseline, test_baseline, mapping
        )
        frames.append(frame)
        summaries.append(summary)

    summary_file = f"{OUTPUT_PREFIX}_steering_summary_{DATASET}.csv"
    pd.DataFrame(summaries).to_csv(summary_file, index=False)
    print("\nFINAL SUMMARY\n", pd.DataFrame(summaries).to_string(index=False))
    print(f"Saved: {summary_file}")
    all_file = f"{OUTPUT_PREFIX}_all_steering_results_{DATASET}.csv"
    pd.concat(frames, ignore_index=True).to_csv(all_file, index=False)
    print(f"Saved: {all_file}")


if __name__ == "__main__":
    main()
