**Options / CLI Flags — Reference**

This document describes the main command-line options used by this repository (`options/*.py`) and how they affect training and testing. Use these flags when running `train.py` and `test.py`.

1) Where options are defined
- **`options/base_options.py`**: base flags shared by training and testing (dataset paths, image sizes, device settings, logging). `BaseOptions.parse()` returns an `opt` namespace used throughout the code.
- **`options/train_options.py`**: training-specific flags (learning rates, epochs, optimizer choices, additional loss weights like `--lambda_style` and `--lambda_OCR`).
- **`options/test_options.py`**: test/inference flags (which epoch to load, results directory, how many test images to run).

2) Important common options (summary)
- `--dataroot`: path to dataset root (expects CycleGAN layout with `trainA`, `trainB`, `testA`, `testB`).
- `--name`: experiment name (checkpoint directory `checkpoints/<name>/`).
- `--model`: model type (default `cycle_gan` unless changed). Kept for compatibility with original code.
- `--gpu_ids`: comma-separated GPU ids (e.g. `0` or `0,1`); use `-1` for CPU. `BaseOptions` will set `opt.device` accordingly.
- `--batch_size`: batch size (CycleGAN commonly uses `1`).
- `--load_size`: resize shorter side before cropping (we set 512 in current defaults for better style fidelity).
- `--crop_size`: final crop size fed to network (512 to match generator). Keep `load_size >= crop_size`.

3) Training-specific flags (from `train_options.py`)
- `--n_epochs`: number of epochs with initial (fixed) learning rate.
- `--n_epochs_decay`: epochs to linearly decay the LR to zero after `--n_epochs`.
- `--lr`: initial learning rate for optimizers (default 0.0002 commonly used).
- `--lr_policy`: learning rate schedule (`linear`, `step`, `cosine`, `plateau`).
- `--beta1` / `--beta2`: Adam betas.
- `--netG` / `--netD`: architecture names used by `networks.define_G` / `define_D` (e.g., `resnet_9blocks`).
- `--embed_dim`: dimension of writer/style embedding; set >0 to enable style conditioning. When enabled, the generator expects an extra `embed_dim` channels concatenated to input.
- `--lambda_style`: weight applied to style loss term. Controls stylistic strength (start around `1.0` and tune).
- `--lambda_OCR`: weight applied to OCR consistency loss. Lower values (0.01–0.1) help preserve readability without overpowering style.
- `--continue_train`: continue training from latest checkpoint if present.

4) Test / inference flags
- `--epoch`: which checkpoint epoch to load (e.g., `latest` or a number like `50`).
- `--results_dir`: directory where test outputs are saved (defaults to `checkpoints/<name>/results/`).
- `--how_many`: max number of test images to run.
- `--ref_image` / `--ref_dir`: (project-specific) path to reference/style images used by `test_custom_style.py` and `test_paragraph.py`.

5) Device and reproducibility
- Set `--gpu_ids 0` to use the first GPU. `BaseOptions.parse()` will populate `opt.device` so code can call `to(opt.device)`.
- For CPU-only runs use `--gpu_ids -1`. Expect much slower execution and OCR may still require CPU-only mode for EasyOCR.
- To reproduce runs, fix random seeds in your training script and ensure deterministic flags in PyTorch if needed.

6) How to enable style conditioning and OCR loss (concrete steps)
- Style conditioning:
  1. Add `--embed_dim 128` to your `train.py` command (or set in options). This creates a generator that concatenates a tiled style vector.
  2. Provide a `StyleEncoder` instance to the model (this is wired in `cycle_gan_model.py` by default if `embed_dim>0`).
  3. Ensure parameters of `StyleEncoder` are included in the generator optimizer (`model.optimizer_G` includes `netstyle_encoder` parameters).

- OCR loss:
  1. Install EasyOCR: `pip install easyocr` and its dependencies.
  2. Set `--lambda_OCR 0.05` (or an initial small value) and monitor validation readability.
  3. If training slows too much, compute OCR loss every K iterations or only on a validation subset.

7) Example commands
PowerShell examples (copyable):
```powershell
# Train with style conditioning (embed_dim=128), OCR loss small weight
python train.py --dataroot C:\data\iam --name writer_ocr_style_512 --gpu_ids 0 --embed_dim 128 --lambda_style 1.0 --lambda_OCR 0.05 --load_size 512 --crop_size 512

# Continue training from latest checkpoint
python train.py --dataroot C:\data\iam --name writer_ocr_style_512 --continue_train --gpu_ids 0

# Test paragraph generation using latest checkpoint
python test_paragraph.py --name writer_ocr_style_512 --epoch latest --gpu_ids 0

# Custom style test using a single reference image
python test_custom_style.py --name writer_ocr_style_512 --epoch latest --ref_image C:\refs\writer1.png --gpu_ids 0
```

8) Debugging tips for common option issues
- Mismatched input channels / size errors:
  - Confirm `--embed_dim` used at training matches the generator definition; `actual_input_nc = input_nc + embed_dim`.
  - Ensure `--load_size` >= `--crop_size` and both match what generators/discriminators expect.
- Checkpoint loading errors:
  - Use `--continue_train` to resume; verify the checkpoint directory `checkpoints/<name>/` contains expected `latest_net_G_A.pth`, etc.
- OCR loss not applied:
  - Confirm `easyocr` installed and `OCRLoss` initialization prints success; otherwise loss returns zero.

This file mirrors `options/*.py` behavior and ties flags to where they influence the model and training loop. If you want, I can also insert short examples directly into `train_options.py` or add `docs/EXAMPLES.md` with reproducible experiment commands and tuned hyperparameters we used.
