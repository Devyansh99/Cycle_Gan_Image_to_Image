"""Test script to generate text in a specific handwriting style"""
import torch
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
from options.test_options import TestOptions
from models import create_model
from util.util import tensor2im
import os

def create_text_image(text, size=(512, 512)):
    """Create a simple text image"""
    img = Image.new('RGB', size, color='white')
    draw = ImageDraw.Draw(img)
    # Use default font, centered
    bbox = draw.textbbox((0, 0), text)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (size[0] - text_width) // 2
    y = (size[1] - text_height) // 2
    draw.text((x, y), text, fill='black')
    return img

def test_custom_style(style_image_path, output_dir):
    """Generate paragraph in a custom handwriting style"""
    
    # Setup options
    opt = TestOptions().parse()
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.isTrain = False
    
    # Load model
    model = create_model(opt)
    model.setup(opt)
    model.eval()
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load style reference image
    style_img = Image.open(style_image_path).convert('RGB')
    style_tensor = transform(style_img).unsqueeze(0).to(model.device)
    
    # Create text input
    paragraph = "The quick brown fox jumps over the lazy dog. This is a test of handwriting style transfer using CycleGAN. Machine learning can generate personalized handwriting."
    text_img = create_text_image(paragraph, size=(512, 512))
    text_tensor = transform(text_img).unsqueeze(0).to(model.device)
    
    # Generate with style
    with torch.no_grad():
        if hasattr(model, 'netstyle_encoder'):
            style_vector = model.netstyle_encoder(style_tensor)
            generated = model.netG_A(text_tensor, writer_style=style_vector)
        else:
            generated = model.netG_A(text_tensor)
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    # Save reference style
    style_img_save = tensor2im(style_tensor)
    style_pil = Image.fromarray(style_img_save)
    style_pil.save(os.path.join(output_dir, 'reference_style.png'))
    
    # Save generated paragraph
    generated_img = tensor2im(generated)
    generated_pil = Image.fromarray(generated_img)
    generated_pil.save(os.path.join(output_dir, 'generated_paragraph.png'))
    
    print(f"\n✅ Generated paragraph in custom style!")
    print(f"📁 Results saved to: {output_dir}/")
    print(f"   - reference_style.png")
    print(f"   - generated_paragraph.png\n")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--style_image', type=str, required=True, help='Path to style reference image')
    parser.add_argument('--output_dir', type=str, default='./custom_style_output', help='Output directory')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='Checkpoints directory')
    parser.add_argument('--name', type=str, default='writer_ocr_cyclegan_512', help='Model name')
    parser.add_argument('--dataroot', type=str, required=True, help='Dataset root (required by model)')
    
    args = parser.parse_args()
    
    # Set test options
    import sys
    sys.argv = [
        sys.argv[0],
        '--checkpoints_dir', args.checkpoints_dir,
        '--name', args.name,
        '--dataroot', args.dataroot,
        '--model', 'cycle_gan',
        '--dataset_mode', 'unaligned',
        '--no_dropout',
        '--embed_dim', '128'
    ]
    
    test_custom_style(args.style_image, args.output_dir)
