from logging import config

import torch

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer, trainer
from recbole.utils import init_seed, init_logger
from recbole.model.sequential_recommender.sasrec_mon_sae import SASRec_Mon_SAE





import time
import statistics
import torch

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

    # ============================================================
    # 1. Configuration
    # ============================================================

    config = Config(
        model=SASRec_Mon_SAE,
        dataset="ml-1m",
        config_file_list=[
            "YAML_files/sasrec_mon_sae.yaml"
        ],
    )

    # ============================================================
    # 2. Reproducibility
    # ============================================================

    init_seed(
        config["seed"],
        config["reproducibility"],
    )

    # ============================================================
    # 3. Logger
    # ============================================================

    init_logger(config)

    # ============================================================
    # 4. Build dataset
    # ============================================================

    dataset = create_dataset(config)

    print("\n======================================")
    print("DATASET")
    print("======================================")
    print(dataset)

    # ============================================================
    # 5. Train / validation / test loaders
    # ============================================================

    train_data, valid_data, test_data = data_preparation(
        config,
        dataset,
    )

    # ============================================================
    # 6. Create SAE model
    #
    # Inside SASRec_Mon_SAE.__init__:
    #
    #   - SASRec is constructed
    #   - pretrained SASRec checkpoint is loaded from base_path
    #   - SASRec is frozen
    #   - SAE is initialized
    #
    # ============================================================

    model = SASRec_Mon_SAE(
        config,
        train_data.dataset,
    ).to(config["device"])

    print("\n======================================")
    print("MODEL")
    print("======================================")
    print(model)

    # ============================================================
    # 7. Verify trainable parameters
    #
    # ONLY the SAE should be trainable.
    # ============================================================

    print("\n======================================")
    print("TRAINABLE PARAMETERS")
    print("======================================")

    total_trainable = 0

    for name, parameter in model.named_parameters():

        if parameter.requires_grad:

            n_params = parameter.numel()
            total_trainable += n_params

            print(
                f"{name:50s} "
                f"{str(tuple(parameter.shape)):20s} "
                f"{n_params:,}"
            )

    print(
        f"\nTotal trainable parameters: "
        f"{total_trainable:,}"
    )

    # ============================================================
    # 8. Sanity check
    #
    # SASRec backbone should NOT appear in trainable parameters.
    # ============================================================

    frozen_names = []

    for name, parameter in model.named_parameters():

        if not parameter.requires_grad:
            frozen_names.append(name)

    print(
        f"Frozen parameter tensors: "
        f"{len(frozen_names)}"
    )

    # ============================================================
    # 9. Create standard RecBole trainer
    # ============================================================

    trainer = Trainer(
        config,
        model,
    )

    # ============================================================
    # 10. Train SAE
    #
    # Trainer automatically calls:
    #
    #       model.calculate_loss(interaction)
    #
    # for every batch.
    #
    # Therefore your SAE objective:
    #
    # lambda_emb  * L_emb
    # + lambda_pred * L_pred
    # + lambda_l1   * L1
    # + lambda_kl   * KL
    #
    # is optimized here.
    # ============================================================

    best_valid_score, best_valid_result = trainer.fit(
        train_data,
        valid_data,
        saved=True,
        show_progress=True,
    )

    print("\n======================================")
    print("TRAINING FINISHED")
    print("======================================")

    print(
        "Best validation score:",
        best_valid_score,
    )

    print(
        "Best validation result:",
        best_valid_result,
    )

    # ============================================================
    # 11. Evaluate best saved SAE checkpoint on test set
    # ============================================================
    # Warm up and load best SAE checkpoint
    trainer.evaluate(
    test_data,
    load_best_model=True,
    show_progress=False
                
                )
    # Evaluate 3 times and calculate average
    test_result, eval_time, eval_std = benchmark_eval(
    lambda: trainer.evaluate(
        test_data,
        load_best_model=False,
        show_progress=False
    ),
    config['device'],
    repeats=3
    )

    print(f"SAE evaluation: {eval_time:.3f} +/- {eval_std:.3f} seconds")
    
    #test_result = trainer.evaluate(
     #   test_data,
      #  load_best_model=True,
       # show_progress=True,
    #)

    print("\n======================================")
    print("TEST RESULT")
    print("======================================")

    print(test_result)

    # ============================================================
    # 12. Show last SAE losses
    # ============================================================

    if hasattr(model, "_last_losses"):

        print("\n======================================")
        print("LAST SAE LOSSES")
        print("======================================")

        for key, value in model._last_losses.items():
            print(
                f"{key:30s}: {value:.6f}"
            )


if __name__ == "__main__":
    main()