**Architecture Overview — CycleGAN (writer-aware) with OCR & Style Loss**

This document summarizes the project's high-level architecture, data flow, and training/inference loop. It focuses on the modifications that make the model writer-aware and preserve text readability.

1) High-level components
- Generator (ResnetGenerator): ResNet-based image-to-image network (default `resnet_9blocks`). When `--embed_dim>0` the generator concatenates a tiled style embedding to the input channels before the first convolution. Input: `[B, C, H, W]` (+ style channels). Output: `[B, C, H, W]` (Tanh in [-1,1]).
- Discriminator (NLayerDiscriminator / PixelDiscriminator): PatchGAN-style discriminators that output a prediction map for real/fake patches.
- StyleEncoder: CNN that converts 1 or N reference images into a per-sample style vector `[B, embed_dim]`. Multiple reference images are averaged to form a stable style.
- WriterEmbedding (legacy): `nn.Embedding` for datasets with fixed writer IDs (useful when labels exist; replaced by `StyleEncoder` for unseen writers).
- OCRLoss: lightweight OCR consistency loss using EasyOCR (if available) — currently uses character-count proxy features and normalized L1 between fake/real.
- StyleLoss: currently a pixel-level L1 between generated image and style reference (cheap and stable; can be swapped for VGG/Gram-based losses).
- ImagePool: history buffer for fake images used to stabilize discriminator updates.
- Visualizer / HTML / WandB: logging, image gallery and remote experiment tracking.

2) Dataflow (training iteration)
1. `set_input(data)` in `cycle_gan_model.py` loads `real_A`, `real_B` and optional `ref_images` or `writer_ids`.
2. If `embed_dim>0`, `netstyle_encoder(reference_images)` -> `style_vector` with shape `[B, embed_dim]`.
3. Forward pass:
   - `fake_B = netG_A(real_A, writer_style=style_vector)` (A→B conditional on style)
   - `fake_A = netG_B(real_B, writer_style=style_vector_other)` (B→A)
4. Cycle: `rec_A = netG_B(fake_B, writer_style_of_A)` and `rec_B = netG_A(fake_A, writer_style_of_B)`.
5. Compute losses:
   - GAN loss: `GANLoss(netD(fake), True/False)`
   - Cycle-consistency: `L1(rec_A, real_A)` + `L1(rec_B, real_B)`
   - Identity (optional): `L1(netG_A(real_B), real_B)`
   - Style loss: `style_loss.compute_loss(generated, style_reference) * lambda_style`
   - OCR loss: `ocr_loss.compute_loss(fake, real) * lambda_OCR` (may be computed every K steps)
6. Backprop: `backward_G()` (sum of above), update G & style encoder params; then `backward_D()` updates discriminators (possibly using `ImagePool.query` for fakes).

3) Key design choices & rationale
- Conditioning by concatenation: The style vector is tiled spatially and concatenated to the input channels. This is simple and effective for global style signals like handwriting stroke width and slant.
- Dynamic StyleEncoder vs. fixed embeddings: `StyleEncoder` supports unseen writers at test time by extracting style from reference images; `WriterEmbedding` is only useful when writer IDs are known and fixed.
- OCR consistency: adds an explicit readability objective to prevent style transfer from distorting characters; implemented as a lightweight proxy to avoid full feature matching overhead.
- Style loss L1 (pixel-space): chosen for stability; Gram/VGG-based alternatives give richer texture transfer but require more compute and tuning.

4) Checkpoints & outputs
- Checkpoints saved under `checkpoints/<name>/` with files like `latest_net_G_A.pth`, `latest_net_style_encoder.pth`.
- Visual outputs and HTML saved under `checkpoints/<name>/web/` and `checkpoints/<name>/paragraph_test/` from test scripts.
- Metrics JSON (`metrics.json`) is written by `util/metrics.py` in the checkpoint folder.

5) Training tips & recommended hyperparameters (starting point)
- Image size: `--load_size 512 --crop_size 512` for better style fidelity.
- Batch size: `1` (typical for image-to-image CycleGAN runs).
- LR: `0.0002` with Adam (`beta1=0.5`, `beta2=0.999`).
- `--embed_dim 128` for style vector dimension.
- Loss weights: `--lambda_style 1.0`, `--lambda_OCR 0.05` (start small for OCR so readability remains strong).
- If OOM at high resolution: reduce load/crop size or use mixed precision (AMP) and/or gradient accumulation.

6) Inference / test-time
- Supply one or more reference images to `StyleEncoder` to extract writer style.
- Call `netG_A(real_A, writer_style=style_vector)` to produce stylized output.
- Test scripts `test_paragraph.py`, `test_custom_style.py`, and `test_three_lines.py` show minimal examples of generating paragraphs and custom-style images.

7) Extension points
- Replace `StyleLoss` with a VGG-feature Gram loss for richer texture transfer.
- Replace OCR proxy with feature-based OCR loss (e.g., compare intermediate OCR network features) for better semantics.
- Experiment with conditional normalization (AdaIN) instead of channel concatenation to inject style.

References (where to look in code)
- Generator/Discriminator: `models/networks.py` (look at `ResnetGenerator` and `NLayerDiscriminator`).
- Main logic: `models/cycle_gan_model.py` (forward, backward_G, optimize_parameters).
- Style extraction: `models/style_encoder.py`.
- Loss helpers: `models/style_loss.py` and `models/ocr_loss.py`.
- Utilities: `util/visualizer.py`, `util/metrics.py`, `util/image_pool.py`.

This overview is meant to give you a focused mental map of the repository and where to make future changes. If you want, I can add a small ASCII dataflow diagram or a runnable minimal example that performs a single forward+loss computation with dummy tensors to validate shapes. Which would you prefer? 
