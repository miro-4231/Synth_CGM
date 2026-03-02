# Synth_CGM

A research project for generating **synthetic Continuous Glucose Monitoring (CGM) signals** using four different generative model families. The goal is to mitigate data imbalance and availability constraints in T1D glucose prediction tasks.

---

## Motivation

CGM datasets (e.g., OhioT1DM) are small and imbalanced — hypoglycemic events are rare but clinically critical. This project benchmarks four generative architectures for their ability to produce realistic 128-step CGM sequences that can be used to augment downstream classification and regression models.

---

## Models

| Model | Architecture | Source |
|---|---|---|
| **VAE** | Convolutional encoder-decoder (DCGAN-inspired), β-VAE loss | `src/VAE_src.py` |
| **DCGAN** | 1D transposed-conv generator + discriminator | `src/GAN_src.py` |
| **Normalizing Flow** | RealNVP with alternating coupling layers | `src/NF_src.py` |
| **DDPM** | 1D U-Net with sinusoidal time embeddings + attention | `src/DDPM_src.py` |

All models are trained on 128-step CGM windows (5-min resolution → ~10 hours) and output signals in the physiological range [40, 400] mg/dL.

---

## Project Structure

```
Synth_CGM/
├── src/
│   ├── data_loader.py      # Data pipeline: OhioT1DM XML, CSV datasets, segmentation
│   ├── VAE_src.py          # VAE model, trainer, loss
│   ├── GAN_src.py          # DCGAN generator, discriminator, trainer
│   ├── NF_src.py           # Normalizing Flow model and trainer
│   └── DDPM_src.py         # Denoising Diffusion Probabilistic Model and trainer
├── notebooks/
│   ├── VAEs.ipynb          # VAE training experiments
│   ├── GANs.ipynb          # GAN training experiments
│   ├── NFs.ipynb           # Normalizing Flow experiments
│   ├── DDPM.ipynb          # DDPM training experiments
│   ├── data_analysis.ipynb # Dataset exploration and statistics
│   └── FID_like_exper.ipynb# Evaluation metrics experiments
├── models/                 # Saved model checkpoints (.pt)
├── data/
│   ├── raw/                # Raw OhioT1DM XML files
│   ├── processed/          # Processed intermediate data
│   └── generated/          # Synthetic samples output
├── tests/
│   ├── test_api.py         # FastAPI endpoint tests (no GPU required)
│   └── test_data_pipeline.py # Data loading and preprocessing tests
├── sample_ddpm.py          # Standalone DDPM sampling script
├── sample_vae.py           # Standalone VAE sampling script
├── sample_nf.py            # Standalone NF sampling script
├── sample_gan.py           # Standalone GAN sampling script
└── serve.py                # FastAPI server exposing all 4 models
```

---

## Setup

### 1. Create and activate the environment

```bash
conda create -n synth_cgm python=3.10
conda activate synth_cgm
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `torch==2.10.0+cu126` requires CUDA 12.6. For CPU-only, replace with:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> ```

### 3. Add the OhioT1DM dataset

Place the OhioT1DM XML files under `data/raw/OhioT1DM/` following the original dataset structure:
```
data/raw/OhioT1DM/
├── train/
│   ├── 559-ws-training.xml
│   └── ...
└── test/
    ├── 559-ws-testing.xml
    └── ...
```

---

## Training

Open and run the relevant notebook in `notebooks/`:

```bash
jupyter notebook notebooks/VAEs.ipynb
```

All trainers log metrics, reconstructions, and model checkpoints via **MLflow**. To view the experiment dashboard:

```bash
mlflow ui
```

---

## Generating Synthetic Samples

Run a sampling script directly to generate and save synthetic signals:

```bash
# From the project root
python sample_ddpm.py   # saves to data/generated/synth_ddpm.pt
python sample_vae.py    # saves to data/generated/synth_vaes.pt
python sample_nf.py     # saves to data/generated/synth_nf.pt
python sample_gan.py    # saves to data/generated/synth_gan.pt
```

---

## API

Start the FastAPI server to expose all 4 models over HTTP:

```bash
uvicorn serve:app --reload
```

### Endpoints

| Endpoint | Description |
|---|---|
| `GET /ddpm/{n}` | Generate `n` samples using DDPM (max 100) |
| `GET /vae/{n}` | Generate `n` samples using VAE (max 100) |
| `GET /nf/{n}` | Generate `n` samples using Normalizing Flow (max 100) |
| `GET /gan/{n}` | Generate `n` samples using DCGAN (max 100) |

Returns a JSON list of shape `[n, 1, 128]`.

Interactive docs available at: `http://127.0.0.1:8000/docs`

---

## Tests

Run tests from the **project root**:

```bash
# Data pipeline tests
python -m unittest tests.test_data_pipeline -v

# API tests (no GPU or model files required)
python -m unittest tests.test_api -v
```

---

## Dataset

This project uses the **[OhioT1DM Dataset](http://smarthealth.cs.ohio.edu/OhioT1DM-dataset.html)** (Marling & Bunescu, 2020), which requires registration. The dataset is not included in this repository.

---

## License

See [LICENSE](LICENSE) for details.
