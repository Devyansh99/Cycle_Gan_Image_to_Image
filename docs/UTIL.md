**Util Folder — Detailed Reference**

This document explains the helper utilities in `util/`, their APIs, expected inputs/outputs, and practical tips for debugging and extending them.

- **`util.py`**
  - Purpose: small helpers used across training and testing.
  - Key functions:
    - `tensor2im(input_image, imtype=np.uint8)`
      - Converts a PyTorch tensor (or numpy array) to a numpy image array suitable for saving or logging.
      - Input: usually a tensor of shape `[B, C, H, W]` (function uses `image_tensor[0]`), range `[-1, 1]`.
      - Output: numpy array shape `[H, W, 3]` with values `[0,255]` and dtype `imtype`.
      - Notes: This function expects a batch and takes the first element; for single-image tensors ensure an extra batch dim (e.g., `unsqueeze(0)`) or pass a numpy array.

    - `diagnose_network(net, name="network")`
      - Prints mean absolute gradient across parameters for quick debugging (helps detect vanishing/zero grads).

    - `init_ddp()` / `cleanup_ddp()`
      - Helpers to initialize/disconnect PyTorch DDP. `init_ddp()` chooses device from `LOCAL_RANK` / `WORLD_SIZE` env vars or falls back to `cuda:0` or CPU.

    - `save_image(image_numpy, image_path, aspect_ratio=1.0)`
      - Save a numpy image (`H,W,3`) to disk using PIL. Optionally resizes by `aspect_ratio`.
      - Note: `image_numpy` must be `uint8` (or convertible); typical usage is `util.save_image(util.tensor2im(tensor), path)`.

    - `print_numpy(x, val=True, shp=False)` — prints summary stats of a numpy array.

    - `mkdirs(paths)` / `mkdir(path)` — robust directory creation helpers.

  - Practical tips:
    - Always ensure tensors fed to `tensor2im()` are in `[-1,1]`; else the scaling will be incorrect.
    - If you see a grayscale result, `tensor2im` automatically tiles single-channel tensors into 3 channels.

- **`image_pool.py`**
  - Purpose: implement a historical image buffer that stores previously generated images for discriminator updates.
  - Class: `ImagePool(pool_size)`
    - `query(images)` accepts a batch (iterable) of images and returns a batch where each returned image is either the current image or a sampled previously stored one (probabilistic mixing).
    - Use-case: stabilizes adversarial training by exposing the discriminator to a wider history of fake examples.
  - Notes:
    - `pool_size=0` disables the buffer and returns inputs directly.
    - The pool stores torch tensors with an added leading batch dim; `query()` returns a single tensor concatenated along batch dimension.
    - If you change image dtype/device outside the pool, ensure stored images remain on the same device before using in discriminator.

- **`metrics.py`**
  - Purpose: simple epoch-level metrics logger that writes `metrics.json` under checkpoint dir.
  - Class: `MetricsLogger(checkpoint_dir)`
    - `log_epoch(epoch, losses)` where `losses` is a dict of name->value; it appends values and writes `metrics.json`.
    - `print_summary(epoch)` prints latest and average per-loss values.
  - Practical tips:
    - Call `log_epoch` at the end of each epoch with a concise loss dict (e.g., `{"G": 1.234, "D": 0.456, "style": 0.06}`).
    - `metrics.json` is small and human-readable; integrate into external dashboards or CI easily.

- **`get_data.py`**
  - Purpose: small convenience downloader for standard CycleGAN/pix2pix datasets.
  - Class: `GetData(technique='cyclegan', verbose=True)`
    - `get(save_path, dataset=None)` will either present options scraped from dataset index pages or download the specified `dataset` archive and unpack it.
  - Notes:
    - Requires network access and `beautifulsoup4` (`bs4`) and `requests` to function.
    - Use only when you need to fetch public demo datasets; for local IAM dataset you usually already have, skip this.

- **`html.py`**
  - Purpose: create simple image web pages to visualize results during training/testing.
  - Class: `HTML(web_dir, title, refresh=0)`
    - `add_header(text)`, `add_images(ims, txts, links, width=400)`, `save()` to write `index.html` into `web_dir`.
    - `get_image_dir()` returns `web_dir/images` where images are saved.
  - Notes:
    - Depends on `dominate` library; install via `pip install dominate` if missing.
    - Useful for quick inspection: Visualizer saves epoch images and updates an HTML gallery view.

- **`visualizer.py`**
  - Purpose: orchestrates logging, HTML generation, and optional wandb logging.
  - Key functions / class:
    - `save_images(webpage, visuals, image_path, aspect_ratio=1.0, width=256)` — saves OrderedDict `visuals` (label→tensor) into `webpage` using `util.tensor2im` and `util.save_image`.
    - `Visualizer(opt)` sets up logging, optional `wandb`, and `HTML` web folder under `checkpoints/<name>/web`.
    - `display_current_results(visuals, epoch, total_iters, save_result=False)` — logs images to wandb and/or writes them to HTML (only on main process rank 0 for DDP).
    - `plot_current_losses(total_iters, losses)` — logs losses to wandb if enabled.
    - `print_current_losses(epoch, iters, losses, t_comp, t_data)` — prints and appends losses to `loss_log.txt` in checkpoint dir.
  - Important details:
    - Visualizer respects distributed training: only main process writes to disk and wandb to prevent duplicates.
    - `util.tensor2im` is called for each visual; ensure visuals are tensors with appropriate shape & range.
    - Wandb integration is optional; set `opt.use_wandb=True` and provide `opt.wandb_project_name` if you want remote logging.

**Quick practical checklist**
- When saving visuals from training loops, call `visualizer.display_current_results(...)` only on rank 0 or rely on `Visualizer`'s built-in guards.
- For saving images manually, convert with `util.tensor2im(tensor)` then `util.save_image(numpy_img, path)`.
- If using `ImagePool`, be aware that returned images are clones of stored tensors — ensure their `device` matches the discriminator's device.

If you'd like, I can also add small code snippets into `docs/UTIL.md` showing exact calls (e.g., minimal example saving a visual dict, or how to initialize wandb). Do you want examples included? 
