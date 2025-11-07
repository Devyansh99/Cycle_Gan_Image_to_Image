# Writer-Aware CycleGAN - Handwriting Style Transfer

Train CycleGAN to generate handwriting in ANY person's style using StyleEncoder.

## 🚀 Training

```bash
python train.py \
  --dataroot /path/to/iam_cyclegan \
  --name writer_ocr_cyclegan \
  --model cycle_gan \
  --dataset_mode unaligned \
  --batch_size 1 \
  --n_epochs 10 \
  --n_epochs_decay 5 \
  --lambda_OCR 0.1 \
  --embed_dim 128 \
  --save_epoch_freq 1 \
  --no_dropout
```
<!-- python train.py --dataroot /home/studentiotlab/image_to_image/data/iam_cyclegan --name writer_ocr_cyclegan --model cycle_gan --dataset_mode unaligned --batch_size 1 --n_epochs 15 --n_epochs_decay 10 --lambda_OCR 0.1 --embed_dim 128 --display_freq 100 --print_freq 100 --save_epoch_freq 5 --save_latest_freq 500 --no_dropout --lr 0.0002 -->

**Outputs:**
- `checkpoints/{name}/` - Model checkpoints
- `checkpoints/{name}/metrics.json` - Training metrics
- `checkpoints/{name}/web/` - Training visualizations

## 🎨 Testing (After Training)

Generate a paragraph in 3 different handwriting styles:

```bash
python test_paragraph.py \
  --checkpoints_dir ./checkpoints \
  --name writer_ocr_cyclegan \
  --dataroot /path/to/iam_cyclegan
```

**Outputs:**
```
checkpoints/writer_ocr_cyclegan/paragraph_test/
├── writer_1/
│   ├── reference_style.png          # Original handwriting sample
│   └── generated_paragraph.png      # Paragraph in this style
├── writer_2/
│   ├── reference_style.png
│   └── generated_paragraph.png
└── writer_3/
    ├── reference_style.png
    └── generated_paragraph.png
```

## 📊 View Metrics

```bash
cat checkpoints/writer_ocr_cyclegan/metrics.json
```

## 🎯 For iPad App

The trained model works with ANY handwriting style:
1. User writes calibration samples
2. StyleEncoder extracts their style
3. Generate text in user's handwriting

## 📁 Project Structure

```
.
├── train.py                   # Training script
├── test_paragraph.py          # Paragraph generation test
├── models/
│   ├── cycle_gan_model.py     # CycleGAN with StyleEncoder
│   ├── style_encoder.py       # Extract style from ANY handwriting
│   ├── networks.py            # Generator/Discriminator
│   └── ocr_loss.py            # OCR consistency loss
├── data/
│   └── unaligned_dataset.py   # Dataset loader
├── options/
│   ├── train_options.py       # Training options
│   └── test_options.py        # Testing options
└── util/
    └── metrics.py             # Metrics tracking
```
