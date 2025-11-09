"""Test script to generate 3 lines of text in a handwriting style"""
import torch
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
from options.test_options import TestOptions
from models import create_model
from util.util import tensor2im
import os

def create_three_lines_image(size=(512, 512)):
    """Create a simple 3-line text image"""
    img = Image.new('RGB', size, color='white')
    draw = ImageDraw.Draw(img)
    
    # Three simple lines
    lines = [
        "The quick brown fox",
        "jumps over the lazy dog.",
        "Machine learning works!"
    ]
    
    y_start = 150
    line_spacing = 80
    
    for i, line in enumerate(lines):
        y = y_start + (i * line_spacing)
        draw.text((50, y), line, fill='black')
    
    return img

def test_three_lines(style_image_path, output_dir):
    """Generate 3 lines in a custom handwriting style"""
    
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
    
    # Create 3-line text input
    text_img = create_three_lines_image(size=(512, 512))
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
    
    # Save input text
    text_img.save(os.path.join(output_dir, 'input_text.png'))
    
    # Save generated 3 lines
    generated_img = tensor2im(generated)
    generated_pil = Image.fromarray(generated_img)
    generated_pil.save(os.path.join(output_dir, 'generated_3lines.png'))
    
    print(f"\n✅ Generated 3 lines in custom style!")
    print(f"📁 Results saved to: {output_dir}/")
    print(f"   - reference_style.png (handwriting reference)")
    print(f"   - input_text.png (plain text input)")
    print(f"   - generated_3lines.png (style transferred output)\n")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--style_image', type=str, required=True, help='Path to style reference image')
    parser.add_argument('--output_dir', type=str, default='./three_lines_test', help='Output directory')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='Checkpoints directory')
    parser.add_argument('--name', type=str, default='writer_ocr_cyclegan_512', help='Model name')
    parser.add_argument('--dataroot', type=str, required=True, help='Dataset root')
    
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
    
    test_three_lines(args.style_image, args.output_dir)
