"""Train LightGCN_CBM using a pretrained RecBole LightGCN checkpoint.

Requirements:
    1. recbole/model/general_recommender/lightgcn_cbm.py contains LightGCN_CBM
    2. recbole/utils/build_concepts.py contains build_cache and seq_to_concepts
    3. BASE_MODEL_PATH points to the pretrained LightGCN checkpoint, not a CBM

Run:
    python train_lightgcn_cbm_final.py

Output:
    The best LightGCN_CBM checkpoint in ./saved, ready for paired steering.
"""

import copy
import inspect
from pathlib import Path

import torch

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.general_recommender.lightgcn_cbm import LightGCN_CBM
from recbole.trainer import Trainer
from recbole.utils import init_seed


# ---------------------------------------------------------------------------
# EDIT THIS PATH: a fully trained LightGCN checkpoint (NOT LightGCN_CBM).
# ---------------------------------------------------------------------------
BASE_MODEL_PATH = "saved/LightGCN-Oct-19-2026_08-45-31.pth"

# These match your LightGCN_CBM defaults. Tune on validation as needed.
CBM_OVERRIDES = {
    "n_free": 64,
    "lambda_concept": 1.0,
    "lambda_recon": 2.0,
    "lambda_distill": 1.0,
    "lambda_z_l2": 1e-2,
    "dedupe_users": True,
    "collect_eval_concepts": False,
    "learning_rate": 1e-3,
    "epochs": 3,
    "eval_step": 5,
    "stopping_step": 10,
    "metrics": ["Recall", "NDCG"],
    "topk": [10],
    "valid_metric": "NDCG@10",
    "checkpoint_dir": "saved",
}




import time
import statistics

def sync_gpu(device):
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

def measure_time(fn, device):
    sync_gpu(device)
    start = time.perf_counter()

    result = fn()

    sync_gpu(device)
    elapsed = time.perf_counter() - start

    return result, elapsed

def benchmark_eval(fn, device, repeats=3):
    times = []
    result = None

    for _ in range(repeats):
        result, elapsed = measure_time(fn, device)
        times.append(elapsed)

    return result, statistics.mean(times), statistics.stdev(times)




def load_checkpoint(path):
    """Load a trusted, local RecBole checkpoint across PyTorch versions."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def verify_pretrained_backbone(model, base_state):
    """Ensure training will use the actual pretrained LightGCN embeddings."""
    for name, param in (
        ("user_embedding.weight", model.user_embedding.weight),
        ("item_embedding.weight", model.item_embedding.weight),
    ):
        if name not in base_state:
            raise RuntimeError(f"Pretrained checkpoint lacks {name}")
        expected = base_state[name].detach().cpu()
        actual = param.detach().cpu()
        if actual.shape != expected.shape:
            raise RuntimeError(
                f"Shape mismatch for {name}: "
                f"model={tuple(actual.shape)}, checkpoint={tuple(expected.shape)}"
            )
        if not torch.allclose(
            actual, expected.to(dtype=actual.dtype), rtol=1e-6, atol=1e-6
        ):
            raise RuntimeError(
                f"{name} does not match the pretrained checkpoint. "
                "Verify the installed LightGCN_CBM model and checkpoint."
            )


def freeze_backbone_train_heads(model):
    """Train only the post-hoc CBM, never the pretrained graph backbone."""
    model.requires_grad_(False)
    trainable_heads = (
        "concept_predictor", "latent_predictor", "reconstructor"
    )
    for name in trainable_heads:
        head = getattr(model, name)
        head.requires_grad_(True)

    if model.user_embedding.weight.requires_grad:
        raise RuntimeError("User embedding unexpectedly trainable")
    if model.item_embedding.weight.requires_grad:
        raise RuntimeError("Item embedding unexpectedly trainable")

    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters found")
    if any(not name.startswith(tuple(h + "." for h in trainable_heads))
           for name in trainable):
        raise RuntimeError(f"Unexpected trainable parameters: {trainable}")

    print("Frozen backbone: user/item embeddings")
    print("Trainable heads:", ", ".join(trainable_heads))
    print("Trainable parameter count:",
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")


def main():
    base_path = Path(BASE_MODEL_PATH).expanduser().resolve()
    if not base_path.is_file():
        raise FileNotFoundError(
            f"Pretrained LightGCN checkpoint not found: {base_path}. "
            "Set BASE_MODEL_PATH at the top of this script."
        )

    checkpoint = load_checkpoint(base_path)
    base_config = checkpoint["config"]
    base_state = checkpoint["state_dict"]

    model_name = base_config["model"]
    if not isinstance(model_name, str):
        model_name = getattr(model_name, "__name__", str(model_name))
    if model_name.lower() != "lightgcn":
        raise ValueError(
            f"Expected pretrained LightGCN, got {model_name!r}. "
            "Do not pass a CBM checkpoint as the teacher."
        )

    dataset_name = base_config["dataset"]
    config_values = copy.deepcopy(base_config.final_config_dict)

    # Avoid the RecBole 'offset' KeyError caused by stale DDP settings.
    # These fields must be generated for the current process, not copied.
    for name in (
        "local_rank", "world_size", "nproc", "offset",
        "ip", "port", "single_spec", "device",
    ):
        config_values.pop(name, None)

    # RecBole appends the dataset name to data_path during Config setup.
    # Saved configs may already contain this dataset-specific suffix.
    original_data_path = config_values.get("data_path")
    if original_data_path is None:
        raise KeyError("Pretrained config does not contain 'data_path'")
    path = Path(original_data_path)
    if path.name == dataset_name:
        config_values["data_path"] = str(path.parent)

    # Preserve pretrained data fields, filtering, splits and graph settings.
    config_values.update(CBM_OVERRIDES)
    config_values.update({
        "model": "LightGCN_CBM",
        "dataset": dataset_name,
        "base_path": str(base_path),
    })

    config = Config(
        model="LightGCN_CBM",
        dataset=dataset_name,
        config_dict=config_values,
    )
    init_seed(config["seed"], config["reproducibility"])

    print("Dataset:", dataset_name)
    print("Pretrained LightGCN:", base_path)
    print("CBM class imported from:", inspect.getfile(LightGCN_CBM))
    print("Device:", config["device"])

    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # Use TRAIN-SPLIT data for the LightGCN graph and concept supervision.
    model = LightGCN_CBM(config, train_data._dataset).to(config["device"])

    # Verify checkpoint loading *before* freezing; never freeze random weights.
    verify_pretrained_backbone(model, base_state)
    freeze_backbone_train_heads(model)

    # Confirm that the model can compute a finite post-hoc training loss.
    trainer = Trainer(config, model)
    best_score, best_valid = trainer.fit(
        train_data, valid_data, saved=True, show_progress=True
    )

    print("\nBest validation score:", best_score)
    print("Best validation metrics:", best_valid)
    print("Saved LightGCN-CBM checkpoint:", trainer.saved_model_file)

    # Evaluation reloads the best checkpoint (rather than last-epoch weights).
    #test_result = trainer.evaluate(
     #   test_data, load_best_model=True, show_progress=False
    #)
    
    # Load best checkpoint and warm up
    trainer.evaluate(
    test_data,
    load_best_model=True,
    show_progress=False
)

    # Measure evaluation 3 times
    test_result, eval_time, eval_std = benchmark_eval(
    lambda: trainer.evaluate(
        test_data,
        load_best_model=False,
        show_progress=False
    ),
     config["device"],
    repeats=3
)

    print(f"LightGCN_TopK_SAE evaluation: {eval_time:.6f} +/- {eval_std:.6f} seconds")
    print("Best-checkpoint test results:", test_result)

    print("Best-checkpoint test results:", test_result)
    print("Test metrics (best validation checkpoint):", test_result)
    print("\nFor your paired-steering script, use:")
    print(f"MODEL_FILE = {trainer.saved_model_file!r}")
    print(f"BASE_MODEL_PATH = {str(base_path)!r}")


if __name__ == "__main__":
    main()
