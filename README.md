# LEO - Language Extraction and Optimization

![LEO Logo](images/LEO_logo05.svg)

## Multilingual NMT Fine-Tuning with LoRA & NLLB

This repository contains a modular, production-ready pipeline for fine-tuning the **NLLB-200 (No Language Left Behind)** model using **PyTorch Lightning** and **LoRA (Low-Rank Adaptation)**.

It is designed to run efficiently on consumer, single-gpu hardware (e.g., NVIDIA RTX 4090) by leveraging **4-bit quantization (QLoRA)** and Mixed Precision training.

## Features

- **Modular Architecture**: Code is organized into clear modules (`dataset`, `model`, `config`) using PyTorch Lightning.
- **Efficient Fine-Tuning**: Uses PEFT/LoRA to fine-tune only a small fraction of parameters (`q_proj`, `v_proj`, `k_proj`, `o_proj`).
- **4-Bit Quantization**: Loads the 1.3B parameter model in 4-bit precision using `bitsandbytes` to minimize VRAM usage.
- **Multilingual Support**: dynamic handling of source and target languages using NLLB's special tokens.
- **Custom Tokenization**: Correctly handles `src_lang` and `forced_bos_token_id` for many-to-many translation tasks.
- **Experiment Tracking**: Integrated with TensorBoard.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/MaxDvTn/LEO.git
   cd LEO
   ```
2. **Create a Conda environment**:
   ```bash
   conda create -n LEO python=3.10
   conda activate LEO
   ```

3. **Ceck the GPU and install the correct version of PyTorch**:
   ```bash
   nvidia-smi
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

4. **Install Dependencies**:
   ```bash
   pip3 install -r requirements.txt

   ```




## Data Preparation

Prepare a CSV file (e.g., `data.csv`) with the following columns:

| source_text | target_text | source_lang | target_lang |
|-------------|-------------|-------------|-------------|
| Hello world | Ciao mondo  | eng_Latn    | ita_Latn    |
| ...         | ...         | ...         | ...         |

*Note: Ensure language codes match the [NLLB supported languages](https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200) (e.g., `eng_Latn`, `ita_Latn`, `fra_Latn`, `spa_Latn`).*

## Usage

### Training

To start fine-tuning, run the `train.py` script. You can specify the path to your data and the output directory.

```bash
python train.py --data_path data/my_data.csv --output_dir checkpoints
```

**Key Hyperparameters** (editable in `config.py`):
- `batch_size`: 8
- `accumulate_grad_batches`: 4
- `learning_rate`: 2e-4
- `max_epochs`: 3
- `precision`: "bf16-mixed" (Optimized for Ampere+ GPUs)

### Inference

To translate text using a trained checkpoint:

```bash
python inference.py \
    --checkpoint checkpoints/nllb-finetuned-epoch=02-val_loss=0.15.ckpt \
    --src_lang eng_Latn \
    --tgt_lang ita_Latn \
    --text "This is a test sentence."
```

## Project Structure

- `config.py`: Central configuration for model parameters and paths.
- `dataset_module.py`: `LightningDataModule` handling data loading, splitting, and tokenization.
- `model_module.py`: `LightningModule` wrapping the NLLB model with LoRA and defining training steps.
- `train.py`: Main entry point for training.
- `inference.py`: Script for loading adapters and generating translations.

## License

[MIT](LICENSE)
