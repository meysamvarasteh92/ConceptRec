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
# SETTINGS
# ============================================================
## SASRec_CBM-Sep-04-2026_15-12-14.pth (beeradvocate model )
## SASRec-Sep-04-2026_14-07-54.pth (beer dataset)
## SASRec_CBM-ml-1mm_sigmoid_New.pth (ml-1m)
## SASRec_CBM-lastfm_sigmoid_final.pth( lastfm)
## SASRec-lastfm_final.pth(lastfm)



DATASET = "ml-1m"  # Options: "ml-1m", "ml-1mm", "lastfm", "beeradvocate"
DATA_PATH = "dataset/ml-1m"

MODEL_FILE = "saved/SASRec_CBM-Aug-22-2026_12-43-59.pth"

BASE_MODEL_PATH = "./saved/sasrec_ml-1m.pth"
## SASRec-Sep-04-2026_14-07-54.pth(beeradvocate dataset)
## SASRec_CBM-Sep-04-2026_15-12-14.pth( beeradvocate model )
## SASRec_CBM-Oct-05-2026_19-44-25.pth(lastfm)
K = 10
NDCG_KEY = f"ndcg@{K}"

MAX_NDCG_LOSS = 10.0

CONCEPT_GROUPS = ["popularity", "era", "genre"]

AMPLIFY_SCALES = np.round(np.arange(1.0, 10.25, 0.2), 2)
SUPPRESS_SCALES = np.round(np.arange(1.0, 0.0, -0.1), 2)




from pathlib import Path
import csv

# True: save validation AND test results for every
# amplification/suppression scale combination.
# False: save all validation combinations and keep
# your existing selected-configuration test results.
LOG_ALL_TEST_COMBINATIONS = True

OUTPUT_DIR = Path("all_steering_combinations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GRID_CSV = OUTPUT_DIR / f"all_combinations_{DATASET}.csv"

# Start a fresh CSV on each full run
if GRID_CSV.exists():
    GRID_CSV.unlink()




# ============================================================
# LOAD MODEL / DATA
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
# METRICS / LOOKUPS
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
    "YearExposure",
    #"ABVExposure",
    #"StyleExposure",
]

config["topk"] = [K]
config["valid_metric"] = f"NDCG@{K}"

L = build_lookups(dataset, config["dataset"])

config["popularity_pool"] = list(L["pop_set"])
config["mid_pool"] = list(L["mid_set"])
config["niche_pool"] = list(L["niche_set"])

config["item_to_genres"] = dict(L["item_to_genres"])
config["genre_list"] = list(L["genre_concepts"])

#if "item_to_era" not in L:
 #   raise KeyError("build_lookups() did not return 'item_to_era'.")

config["item_to_era"] = dict(L["item_to_era"])


# ABV
#config['item_to_abv'] = dict(L['item_to_abv'])
 #Automatically get ABV category names
#config['abv_list'] = list(dict.fromkeys(L['item_to_abv'].values()))

# Style
#config['item_to_style'] = dict(L['item_to_style'])
#config["style_list"] = list(L["style_concepts"])

# ============================================================
# CONCEPT GROUPS
# ============================================================

# For LastFM, suppress Electronic/Rock and amplify all other
# genre concepts supported by both the lookup and the CBM.
if DATASET in {"ml-1m", "ml-1mm", "lastfm"}:

    GENRE_ALL = [
        g for g in config["genre_list"]
        if g in CBM_model.concept_names
    ]
elif DATASET == "beeradvocate":
    GENRE_ALL = [
        g for g in config["style_list"]
        if g in CBM_model.concept_names
    ]

if DATASET == "lastfm":
    GENRE_SUPPRESS = ["Electronic", "Rock"]

elif DATASET in {"ml-1m", "ml-1mm"}:
    GENRE_SUPPRESS = [
        "Drama",
        "Sci-Fi",
        "Romance",
        "Thriller",
        "Horror",
        "Comedy",
    ]
elif DATASET == "beeradvocate":
    GENRE_SUPPRESS = [
            "Belgian Abbey & Strong Ale",
            "IPA",
            "Stout",
        ]

else:
    raise ValueError(
        f"Define GENRE_SUPPRESS for dataset '{DATASET}'."
    )

missing = [
    g for g in GENRE_SUPPRESS
    if g not in GENRE_ALL
]

if missing:
    raise ValueError(
        f"These genre concepts are missing from the CBM: {missing}"
    )

GENRE_AMPLIFY = [
    g for g in GENRE_ALL
    if g not in GENRE_SUPPRESS
]

GROUP_CONFIG = {
    "popularity": {
        "amplify": ["mid", "niche"],
        "suppress": ["popularity"],
        "all_concepts": ["popularity", "mid", "niche"],
        "output_file": f"popularity_steering_results_{DATASET}.csv",
    },

    "era": {
        "amplify": ["classic", "contemporary"],
        "suppress": ["modern"],
        "all_concepts": [
            "classic",
            "retro",
            "modern",
            "contemporary",
        ],
        "output_file": f"era_steering_results_{DATASET}.csv",
    },

    "genre": {
        "amplify": GENRE_AMPLIFY,
        "suppress": GENRE_SUPPRESS,
        "all_concepts": GENRE_ALL,
        "output_file": f"genre_steering_results_{DATASET}.csv",
    },

    "style": {
        "amplify": GENRE_AMPLIFY,
        "suppress": GENRE_SUPPRESS,
        "all_concepts": GENRE_ALL,
        "output_file": f"style_steering_results_{DATASET}.csv",
    },
    "abv": {
        "amplify": ["abv_high", "abv_med"],
        "suppress": ["abv_low"],
        "all_concepts": ["abv_low", "abv_med", "abv_high"],
        "output_file": f"abv_steering_results_{DATASET}.csv",
    },
}


# ============================================================
# PATCH MODEL FOR MULTIPLE INTERVENTIONS
# ============================================================

def patched_forward(self, item_seq, item_seq_len):
    h = self._encode(item_seq, item_seq_len)

    logits = self.concept_predictor(h)
    c_hat = torch.sigmoid(logits)

    #z = self.latent_predictor(h)

    if getattr(self, "_interventions", None) and not self.training:
        c_hat = c_hat.clone()

        for idx, scale in self._interventions.items():
            c_hat[:, idx] *= scale

    #bottleneck = torch.cat([c_hat, z], dim=-1)
    bottleneck = c_hat
    h_hat = self.reconstructor(bottleneck)

    #return h, c_hat, z, h_hat
    return h, c_hat, h_hat



CBM_model.forward = types.MethodType(
    patched_forward,
    CBM_model,
)

CBM_model._interventions = {}
CBM_model.eval()


# ============================================================
# EVALUATOR
# ============================================================

trainer = Trainer(config, CBM_model)

trainer.eval_collector = Collector(config)
trainer.evaluator = Evaluator(config)
trainer.eval_collector.data_collect(train_data)


def evaluate(data):
    return trainer.evaluate(
        data,
        load_best_model=False,
        show_progress=False,
    )




# ============================================================
# BASELINES
# ============================================================

CBM_model._interventions = {}

print("\nComputing validation baseline...")
valid_baseline = evaluate(valid_data)

print(
    f"Validation NDCG@{K}: "
    f"{valid_baseline[NDCG_KEY]:.6f}"
)

print("\nComputing test baseline...")
test_baseline = evaluate(test_data)

print(
    f"Test NDCG@{K}: "
    f"{test_baseline[NDCG_KEY]:.6f}"
)


# ============================================================
# METRIC NAMES
# ============================================================

def build_metric_keys(result, group_name, concepts):

    if group_name == "popularity":
        keys = {
            "popularity": f"pop_rate@{K}",
            "mid": f"mid_rate@{K}",
            "niche": f"niche_rate@{K}",
        }

    elif group_name == "era":
        keys = {
            c: f"era_{c}_rate@{K}"
            for c in concepts
        }

    elif group_name == "genre":
        keys = {
            c: f"genre_{c}_rate@{K}"
            for c in concepts
        }

    elif group_name == "abv":
            keys = {
                c: f"abv_{c}_rate@{K}"
                for c in concepts
            }
    
    elif group_name == "style":
        keys = {
            c: f"style_{c.replace(' ', '_')}_rate@{K}"
            for c in concepts
        }
    
    else:
        raise ValueError(
            f"Unknown concept group: {group_name}"
        )

    for concept in concepts:
        key = keys[concept]

        if key not in result:
            raise KeyError(
                f"Missing metric '{key}' for '{concept}'.\n"
                f"Available metrics:\n{list(result.keys())}"
            )

    return keys


# ============================================================
# STEERING SCORE
# ============================================================



def steering_stats(
    result,
    baseline,
    amp_concept,
    sup_concept,
    metric_keys,
):

    amp_key = metric_keys[amp_concept]
    sup_key = metric_keys[sup_concept]

    amp_delta = 100.0 * (
        result[amp_key] - baseline[amp_key]
    )

    sup_delta = 100.0 * (
        result[sup_key] - baseline[sup_key]
    )
    balance_gap = abs(
        abs(amp_delta) - abs(sup_delta)
    )


    ndcg_change = 100.0 * (
        result[NDCG_KEY] - baseline[NDCG_KEY]
    ) / (baseline[NDCG_KEY] + 1e-12)

    ndcg_loss = max(0.0, -ndcg_change)

    mean_target = (
        amp_delta + abs(sup_delta)
    ) / 2.0

    non_target_deltas = []

    for concept, key in metric_keys.items():

        if concept in {amp_concept, sup_concept}:
            continue

        delta = 100.0 * (
            result[key] - baseline[key]
        )

        non_target_deltas.append(abs(delta))

    max_non_target = (
        max(non_target_deltas)
        if non_target_deltas
        else 0.0
    )

    score = mean_target - max_non_target

    return {
        "Amplify Delta (pp)": amp_delta,
        "Suppress Delta (pp)": sup_delta,
        "Mean Target (pp)": mean_target,
        "Max Non-target (pp)": max_non_target,
        "Balance Gap": balance_gap,
        "NDCG Loss (%)": ndcg_loss,
        "Target-Non-target": score,
    }


def valid_configuration(stats):
    return (
        
        stats["NDCG Loss (%)"] <= MAX_NDCG_LOSS    )    



# All exposure metrics reported by the evaluator,
# including popularity, genre/style, era and ABV,
# depending on the dataset and configured metrics.
EXPOSURE_KEYS = sorted(
    key for key in valid_baseline
    if key.endswith(f"rate@{K}")
    and key in test_baseline
)

print("Logging exposure metrics:", EXPOSURE_KEYS)


def save_combination(
    split,
    group_name,
    amp_concept,
    sup_concept,
    amp_scale,
    sup_scale,
    result,
    baseline,
    stats,
    valid_on_validation,
):

    base_ndcg = float(baseline[NDCG_KEY])
    new_ndcg = float(result[NDCG_KEY])

    ndcg_change = (
        100.0 * (new_ndcg - base_ndcg)
        / (base_ndcg + 1e-12)
    )

    row = {
        "Dataset": DATASET,
        "Split": split,
        "Group": group_name,
        "Amplify": amp_concept,
        "Suppress": sup_concept,
        "Amplify Scale": float(amp_scale),
        "Suppress Scale": float(sup_scale),
        "Valid on Validation": bool(valid_on_validation),

        "Baseline NDCG": base_ndcg,
        "Steered NDCG": new_ndcg,
        "NDCG Change (%)": ndcg_change,

        **stats,
    }

    # Record every exposure metric, not only
    # the amplification/suppression targets.
    for key in EXPOSURE_KEYS:

        before = float(baseline[key])
        after = float(result[key])

        row[f"{key} Baseline"] = before
        row[f"{key} Steered"] = after

        # Percentage-point change
        row[f"{key} Delta (pp)"] = (
            100.0 * (after - before)
        )

    # Append each evaluation immediately
    # so results are preserved during long runs.
    with GRID_CSV.open("a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(row.keys())
        )

        if f.tell() == 0:
            writer.writeheader()

        writer.writerow(row)



# ============================================================
# FIND BEST SCALES ON VALIDATION
# ============================================================

def find_best_scales(
    group_name,

    amp_concept,
    sup_concept,
    metric_keys,
):

    amp_idx = CBM_model.concept_names.index(
        amp_concept
    )

    sup_idx = CBM_model.concept_names.index(
        sup_concept
    )

    best = None

    for amp_scale in AMPLIFY_SCALES:

        for sup_scale in SUPPRESS_SCALES:

            CBM_model._interventions = {
                amp_idx: float(amp_scale),
                sup_idx: float(sup_scale),
            }

            result = evaluate(valid_data)

            stats = steering_stats(
                result,
                valid_baseline,
                amp_concept,
                sup_concept,
                metric_keys,
            )
            is_valid = valid_configuration(stats)
            # Save EVERY validation configuration
            save_combination(
                split="validation",
                group_name=group_name,
                amp_concept=amp_concept,
                sup_concept=sup_concept,
                amp_scale=amp_scale,
                sup_scale=sup_scale,
                result=result,
                baseline=valid_baseline,
                stats=stats,
                valid_on_validation=is_valid,
            )
            # Optionally evaluate and save EVERY test
            # configuration (including invalid ones).
            if LOG_ALL_TEST_COMBINATIONS:

                test_result = evaluate(test_data)

                test_stats = steering_stats(
                    test_result,
                    test_baseline,
                    amp_concept,
                    sup_concept,
                    metric_keys,
                )

                save_combination(
                    split="test",
                    group_name=group_name,
                    amp_concept=amp_concept,
                    sup_concept=sup_concept,
                    amp_scale=amp_scale,
                    sup_scale=sup_scale,
                    result=test_result,
                    baseline=test_baseline,
                    stats=test_stats,
                    valid_on_validation=is_valid,
                )

            if not is_valid:
                continue

            if (
                best is None
                or stats["Target-Non-target"]
                > best["Target-Non-target"]
            ):
                best = {
                    "Amplify Scale": float(amp_scale),
                    "Suppress Scale": float(sup_scale),
                    **stats,
                }

    CBM_model._interventions = {}

    return best


# ============================================================
# TEST THE SELECTED SCALES
# ============================================================

def evaluate_on_test(
    amp_concept,
    sup_concept,
    amp_scale,
    sup_scale,
    metric_keys,
):

    amp_idx = CBM_model.concept_names.index(
        amp_concept
    )

    sup_idx = CBM_model.concept_names.index(
        sup_concept
    )

    CBM_model._interventions = {
        amp_idx: float(amp_scale),
        sup_idx: float(sup_scale),
    }

    result = evaluate(test_data)

    stats = steering_stats(
        result,
        test_baseline,
        amp_concept,
        sup_concept,
        metric_keys,
    )

    CBM_model._interventions = {}

    return stats


# ============================================================
# RUN POPULARITY + ERA + GENRE
# ============================================================

all_results = []
summary_rows = []

for group_name in CONCEPT_GROUPS:

    group = GROUP_CONFIG[group_name]

    amplify_concepts = group["amplify"]
    suppress_concepts = group["suppress"]
    all_concepts = group["all_concepts"]

    print("\n" + "=" * 60)
    print(f"{group_name.upper()} STEERING")
    print("=" * 60)

    # Check steering concepts
    for concept in (
        amplify_concepts
        + suppress_concepts
    ):
        if concept not in CBM_model.concept_names:
            raise ValueError(
                f"Concept '{concept}' is not in CBM_model.concept_names."
            )

    metric_keys = build_metric_keys(
        valid_baseline,
        group_name,
        all_concepts,
    )

    pairs = list(
        product(
            amplify_concepts,
            suppress_concepts,
        )
    )

    group_rows = []

    print(
        f"Searching {len(pairs)} concept pairs..."
    )

    for i, (amp_concept, sup_concept) in enumerate(
        pairs,
        start=1,
    ):

        print(
            f"\n[{i}/{len(pairs)}] "
            f"Amplify {amp_concept} / "
            f"Suppress {sup_concept}"
        )

        best = find_best_scales(
            group_name,
            amp_concept,
            sup_concept,
            metric_keys,
        )

        if best is None:
            print(
                "  No valid validation configuration. "
                "Counting this pair as 0."
            )

            group_rows.append({
                "Group": group_name,
                "Amplify": amp_concept,
                "Suppress": sup_concept,
                "Valid": False,
                "Amplify Scale": np.nan,
                "Suppress Scale": np.nan,
                "Val Target-Non-target": 0.0,
                "Test Amplify Delta (pp)": 0.0,
                "Test Suppress Delta (pp)": 0.0,
                "Test Mean Target (pp)": 0.0,
                "Test Max Non-target (pp)": 0.0,
                "Test Balance Gap": 0.0,
                "Test NDCG Loss (%)": 0.0,
                "Test Target-Non-target": 0.0,
            })

            continue

        test_stats = evaluate_on_test(
            amp_concept,
            sup_concept,
            best["Amplify Scale"],
            best["Suppress Scale"],
            metric_keys,
        )

        row = {
            "Group": group_name,
            "Amplify": amp_concept,
            "Suppress": sup_concept,
            "Valid": True,
            "Amplify Scale":
                best["Amplify Scale"],
            "Suppress Scale":
                best["Suppress Scale"],
            "Val Target-Non-target":
                best["Target-Non-target"],
            "Test Amplify Delta (pp)":
                test_stats["Amplify Delta (pp)"],
            "Test Suppress Delta (pp)":
                test_stats["Suppress Delta (pp)"],
            "Test Mean Target (pp)":
                test_stats["Mean Target (pp)"],
            "Test Max Non-target (pp)":
                test_stats["Max Non-target (pp)"],
            "Test Balance Gap":
                test_stats["Balance Gap"],
            "Test NDCG Loss (%)":
                test_stats["NDCG Loss (%)"],
            "Test Target-Non-target":
                test_stats["Target-Non-target"],
        }

        group_rows.append(row)

        print(
            f"  Best scales: "
            f"amp={best['Amplify Scale']:.2f}, "
            f"sup={best['Suppress Scale']:.2f}"
        )

        print(
            f"  Test score: "
            f"{test_stats['Target-Non-target']:.3f}"
        )

    group_df = pd.DataFrame(group_rows)

    if group_df.empty:
        group_score = None

        print(
            f"\nNo valid validation configurations "
            f"for {group_name}."
        )

    else:
        group_score = group_df[
            "Test Target-Non-target"
        ].mean()

        group_df.to_csv(
            group["output_file"],
            index=False,
        )

        all_results.append(group_df)

        print(
            f"\nFINAL {group_name.upper()} "
            f"TEST SCORE = {group_score:.3f}"
        )
        valid_pairs = int(group_df["Valid"].sum())
        failed_pairs = len(group_df) - valid_pairs
        print(
            f"Valid pairs = {valid_pairs}/{len(group_df)}"
        )

        print(
        f"Failed pairs counted as 0 = {failed_pairs}"
        )

        print(
            f"Saved to: "
            f"{group['output_file']}"
        )

    summary_rows.append(
        {
            "Group": group_name,
            "Test Score": group_score,
            "Valid Pairs": valid_pairs,
            "Failed Pairs": failed_pairs,
            "Total Pairs": len(pairs),
        }
    )


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_df = pd.DataFrame(summary_rows)

print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(summary_df.to_string(index=False))

summary_df.to_csv(
    f"steering_summary_{DATASET}.csv",
    index=False,
)

if all_results:

    final_df = pd.concat(
        all_results,
        ignore_index=True,
    )

    final_df.to_csv(
        f"all_steering_results_{DATASET}.csv",
        index=False,
    )

    print(
        f"\nSaved combined results to: "
        f"all_steering_results_{DATASET}.csv"
    )
