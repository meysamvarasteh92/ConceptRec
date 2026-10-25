
# ConceptRec

**Steerable and Interpretable Recommendation via Post-Hoc Concept Bottlenecks**

This repository contains the code for **ConceptRec**, a post-hoc framework for interpreting and steering pretrained recommender systems. ConceptRec combines metadata-supervised concepts with an unsupervised free channel to preserve recommendation quality while enabling targeted concept amplification and suppression.

## Framework

<p align="center">
  <img src="conceptrec.png" width="850" alt="ConceptRec Framework">
</p>

<p align="center">
  <em>Overview of the ConceptRec framework.</em>
</p>


## Environment Setup

Create and activate a Python virtual environment:

```bash
python3 -m venv conceptrec-env
source conceptrec-env/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

To deactivate the environment:

```bash
deactivate
```

## Experiments

- **Datasets:** MovieLens 1M (ML-1M), BeerAdvocate, and LastFM.
- **Backbones:** SASRec and LightGCN.
- **Baselines:** IaSAE and RecSAE.
- **Evaluation:** nDCG@10, RBO@10, representation similarity, concept alignment, and Push-Pull steering.

## Training and Evaluation

### Training ConceptRec

To train ConceptRec, first train the backbone recommender, then run:

```bash
python run_conceptrec.py \
    --dataset ml-1m \
    --config ConceptRec_config.yaml \
    --lambda_concept 1.0 \
    --lambda_recon 2.0 \
    --epochs 3000
```

### Evaluation and Steering

After training both the backbone and ConceptRec:

- **RQ1 & RQ2:** Run `Steering.ipynb` to evaluate reconstruction fidelity and concept alignment.
- **RQ3:** Run `steering_concept_all.py` to evaluate concept steering effectiveness.

