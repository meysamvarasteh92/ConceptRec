

"""Train LightGCN_TopK_SAE from a PRETRAINED RecBole LightGCN checkpoint.

1. Install lightgcn_topk_sae.py under recbole/model/general_recommender/.
2. Update BASE_MODEL_PATH below, then run: python train_lightgcn_topk_sae.py
3. Use resulting checkpoint for SAE steering and fidelity experiments.
"""

import copy
import inspect
from pathlib import Path
import torch
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer
from recbole.utils import init_seed
from recbole.model.general_recommender.lightgcn_topk_sae import LightGCN_TopK_SAE

BASE_MODEL_PATH = "saved/LightGCN-Oct-19-2026_08-45-31.pth"

SAE_OVERRIDES = {
    "sae_k": 32,
    "sae_scale_size": 32,
    "lambda_recon": 1.0,
    "lambda_pred": 0.0,             # same pure SAE default as SASRec
    "sae_num_sampled_items": 256,
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



def main():
    path = Path(BASE_MODEL_PATH).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Pretrained LightGCN checkpoint not found: {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    base_config = checkpoint["config"]
    model_name = base_config["model"]
    if not isinstance(model_name, str):
        model_name = getattr(model_name, "__name__", str(model_name))
    if model_name.lower() != "lightgcn":
        raise ValueError(f"Expected a LightGCN base checkpoint, got {model_name}")

    dataset_name = base_config["dataset"]
    values = copy.deepcopy(base_config.final_config_dict)
    for key in ("local_rank", "world_size", "nproc", "offset", "ip", "port", "single_spec", "device"):
        values.pop(key, None)
    data_path = Path(values["data_path"])
    if data_path.name == dataset_name:
        values["data_path"] = str(data_path.parent)
    values.update(SAE_OVERRIDES)
    values.update({
        "model": "LightGCN_TopK_SAE",
        "dataset": dataset_name,
        "base_path": str(path),
    })
    config = Config(
        model="LightGCN_TopK_SAE",
        dataset=dataset_name,
        config_dict=values,
    )
    init_seed(config["seed"], config["reproducibility"])
    print("Using SAE implementation:", inspect.getfile(LightGCN_TopK_SAE))
    print("Data:", dataset_name, "Device:", config["device"])

    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    # CRITICAL: use train-split graph, not unsplit dataset.
    model = LightGCN_TopK_SAE(config, train_data._dataset).to(config["device"])
    model.invalidate_cache()

    for key, p in (
        ("user_embedding.weight", model.user_embedding.weight),
        ("item_embedding.weight", model.item_embedding.weight),
    ):
        expected = checkpoint["state_dict"][key].to(device=p.device, dtype=p.dtype)
        if not torch.allclose(p, expected, rtol=1e-6, atol=1e-6):
            raise RuntimeError(f"Pretrained {key} failed verification")
        assert not p.requires_grad, f"{key} should be frozen"

    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    if not trainable or any(not name.startswith("sae.") for name in trainable):
        raise RuntimeError(f"Expected ONLY SAE parameters trainable; got {trainable}")
    print("Trainable SAE parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    trainer = Trainer(config, model)
    best_score, best_valid = trainer.fit(
        train_data, valid_data, saved=True, show_progress=True
    )
    print("Best validation score:", best_score)
    print("Best validation metrics:", best_valid)
    print("Saved model:", trainer.saved_model_file)
    #test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=False)

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


if __name__ == "__main__":
    main()




