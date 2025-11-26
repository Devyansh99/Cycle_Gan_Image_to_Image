**Models Folder — Detailed Reference**

This document explains each source file in `models/`, the main classes/functions, expected tensor shapes, and practical notes for training/debugging.

- **`__init__.py`**
  - Marks `models` as a Python package. No runtime logic.

- **`base_model.py`**
  - Purpose: common training/inference scaffolding used by model classes (e.g., `CycleGANModel`).
  - Key responsibilities:
    - Device placement and `to(device)` logic for all networks.
    - Checkpoint save/load conventions (names like `latest_net_G_A.pth`).
    - Optimizer & scheduler setup and stepping.
    - Utility helpers: `set_input()`, `get_current_losses()`, `save_networks()`, `load_networks()`, and `print_networks()`.
  - Typical usage: subclass `BaseModel` and implement `forward()`, `optimize_parameters()`, `update_learning_rate()`.
  - Debug tips: check checkpoint folder and naming if loads fail; inspect `print_networks()` to verify `embed_dim` was applied.

- **`cycle_gan_model.py`**
  - Purpose: training and inference logic for the CycleGAN variant used here (includes style conditioning and extra losses).
  - Main components:
    - `netG_A`, `netG_B` (generators) and `netD_A`, `netD_B` (discriminators).
    - Optional `netstyle_encoder` (instance of `StyleEncoder`) and legacy `netwriter_encoder`.
    - Loss components: GAN, cycle-consistency (L1), identity (optional), OCR loss, and style loss.
  - Important methods:
    - `set_input(data)`: prepares `real_A`, `real_B` tensors and any metadata (writer ids / reference images).
    - `forward()`: runs generators to produce `fake_B` and `fake_A` (passes `writer_style` when present).
    - `backward_G()`: composes generator losses — GAN + cycle + identity + OCR + style (each multiplied by its lambda hyperparameter).
    - `backward_D_basic(netD, real, fake)`: standard discriminator update helper.
    - `optimize_parameters()`: runs forward, backward (G and D) and optimizer steps.
  - Shapes & conventions:
    - Images: `[B, C, H, W]` (B usually 1 in CycleGAN experiments here).
    - Style vector from `StyleEncoder`: `[B, embed_dim]`.
    - Generator signature: `netG(input_image, writer_style=None)`.
  - Practical tips:
    - Ensure `StyleEncoder` parameters are added to the generator optimizer so they learn jointly.
    - If OCR loss is expensive, compute it every K iterations or set `--lambda_OCR` small.

- **`networks.py`**
  - Purpose: contains the concrete network architectures, initializers, schedulers, and loss wrappers.
  - Key classes and functions:
    - `ResnetGenerator(input_nc, output_nc, ngf, norm_layer, n_blocks, embed_dim=0)`
      - If `embed_dim>0`, the generator concatenates a tiled style tensor to the input channels.
      - Forward signature: `forward(input, writer_style=None)` where `input` is `[B,C,H,W]` and `writer_style` is `[B,embed_dim]`.
      - Implementation detail: `writer_style` is expanded to `[B, embed_dim, H, W]` and `torch.cat`'d to input channels before the first conv.
    - `ResnetBlock` — residual block used by the generator.
    - `NLayerDiscriminator` — PatchGAN classifier returning a prediction map; input `[B, C, H, W]`.
    - `PixelDiscriminator` — 1×1 discriminator variant.
    - `GANLoss(gan_mode)` — wrapper supporting `lsgan`, `vanilla`, and `wgangp`.
    - `get_scheduler(optimizer, opt)` and `init_weights(net, init_type)` helpers.
  - Debug tips:
    - If `RuntimeError: size mismatch` occurs, check `input_nc` and whether `embed_dim` was counted into `actual_input_nc`.
    - Use `print_networks()` (from `BaseModel`) to confirm network parameter counts and shapes.

- **`style_encoder.py`**
  - Purpose: extract a compact style embedding from one or multiple reference handwriting images.
  - API:
    - `StyleEncoder(embed_dim=128)` constructs encoder.
    - `forward(reference_images)` accepts either `[B, C, H, W]` or `[B, N, C, H, W]` (N references per sample). Returns `[B, embed_dim]`.
  - Architecture: a small conv stack (downsampling) → `AdaptiveAvgPool2d(1)` → FC layers → `embed_dim` vector.
  - Practical notes:
    - Averaging multiple references reduces variance and is recommended (3–5 refs per user).
    - Ensure input size used at inference matches the size used when training the style encoder (resize/crop consistency).

- **`writer_encoder.py`**
  - Purpose: a simple `nn.Embedding` lookup for datasets where each image has a writer ID.
  - API: `WriterEmbedding(num_writers, embed_dim)` and `forward(writer_ids)` -> `[B, embed_dim]`.
  - When to use: if your dataset provides writer IDs and you want a fixed embedding per writer.
  - When to avoid: if you need to support unseen writers at inference — prefer `StyleEncoder`.

- **`ocr_loss.py`**
  - Purpose: measure OCR consistency between generated & real images to preserve text readability.
  - How it works (current lightweight implementation):
    - Wraps EasyOCR (if available) to extract recognized text strings per image.
    - Computes a simple scalar feature per image (total character count) and returns normalized L1 between fake and real counts.
  - API:
    - Construct: `OCRLoss(device='cuda', lang_list=['en'])`.
    - Call: `compute_loss(fake_images, real_images)` -> scalar `torch.Tensor`.
  - Practical notes:
    - EasyOCR must be installed (`pip install easyocr`) to enable this loss; otherwise computation returns zero.
    - OCR is expensive: compute less frequently or on a small validation set if training cost is critical.

- **`style_loss.py`**
  - Purpose: enforce stylistic resemblance to a reference style image.
  - Current implementation: pixel-level L1 (`F.l1_loss(generated, style_reference)`) — stable and cheap.
  - API: `StyleLoss.compute_loss(generated, style_reference)` -> scalar.
  - Extension ideas: replace with a VGG-based perceptual Gram matrix loss for richer texture/style capture.

**Quick practical checklist**
- To enable style-conditioned training:
  - set `embed_dim` when creating generator (`--embed_dim 128`).
  - instantiate and pass `StyleEncoder` to the model so `netstyle_encoder` is available.
  - include `netstyle_encoder` parameters in the generator optimizer.
- To enable OCR loss: install `easyocr` and adjust `--lambda_OCR` and compute frequency.
- To debug style influence: try toggling the style vector (zeros vs. reference) and observe generator outputs.

This updated file provides concrete signatures, expected shapes, and practical tips. Let me know if you want the same level of detail for `options/` and `util/`, or if you'd like examples showing how to enable each feature in `train.py`.
