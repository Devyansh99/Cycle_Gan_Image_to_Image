"""
Test trained model by generating a paragraph in 3 different handwriting styles.
Run this AFTER training is complete.

Usage:
    python test_paragraph.py --checkpoints_dir ./checkpoints --name writer_ocr_cyclegan --dataroot /path/to/data
"""

import torch
import os
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
from options.test_options import TestOptions
from models import create_model
from util.util import tensor2im
import glob


class ParagraphTester:
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = create_model(opt)
        self.model.setup(opt)
        self.model.eval()
        
        # Create output directory
        self.output_dir = Path(opt.checkpoints_dir) / opt.name / "paragraph_test"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # Test paragraph
        self.paragraph = (
            "The quick brown fox jumps over the lazy dog. "
            "This is a test of handwriting style transfer using CycleGAN. "
            "Machine learning can generate personalized handwriting."
        )
    
    def create_text_image(self, text, size=(512, 256)):
        """Create a simple text image as input"""
        img = Image.new('RGB', size, color='white')
        draw = ImageDraw.Draw(img)
        
        # Try to use a font, fallback to default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        # Draw text with word wrapping
        margin = 20
        y = margin
        for line in self._wrap_text(text, size[0] - 2*margin, font, draw):
            draw.text((margin, y), line, fill='black', font=font)
            y += 30
        
        return img
    
    def _wrap_text(self, text, max_width, font, draw):
        """Wrap text to fit within max_width"""
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            bbox = draw.textbbox((0, 0), test_line, font=font)
            width = bbox[2] - bbox[0]
            
            if width <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines
    
    def get_random_style_images(self, n=3):
        """Get n random images from dataset to extract styles"""
        # Try testA first, then trainA
        patterns = [
            os.path.join(self.opt.dataroot, "testA", "*.png"),
            os.path.join(self.opt.dataroot, "trainA", "*.png"),
        ]
        
        all_images = []
        for pattern in patterns:
            all_images.extend(glob.glob(pattern))
            if len(all_images) >= n:
                break
        
        if len(all_images) < n:
            print(f"Warning: Only found {len(all_images)} images, need {n}")
        
        return random.sample(all_images, min(n, len(all_images)))
    
    def generate_paragraph_with_style(self, style_img_path, style_name):
        """Generate paragraph in given handwriting style"""
        print(f"\n📝 Generating paragraph in {style_name}...")
        
        # Load and process style reference image
        style_img = Image.open(style_img_path).convert('RGB')
        style_tensor = self.transform(style_img).unsqueeze(0).to(self.device)
        
        # Create text image as input
        text_img = self.create_text_image(self.paragraph)
        text_tensor = self.transform(text_img).unsqueeze(0).to(self.device)
        
        # Extract style and generate
        with torch.no_grad():
            if hasattr(self.model, 'netstyle_encoder'):
                style_vector = self.model.netstyle_encoder(style_tensor)
                generated = self.model.netG_A(text_tensor, writer_style=style_vector)
            else:
                generated = self.model.netG_A(text_tensor)
        
        # Convert tensors to images (tensor2im handles batch dimension internally)
        style_img_save = tensor2im(style_tensor)
        generated_img = tensor2im(generated)
        
        return style_img_save, generated_img
    
    def run_test(self):
        """Main testing function"""
        print("\n" + "="*70)
        print("🎨 PARAGRAPH HANDWRITING STYLE TRANSFER TEST")
        print("="*70)
        print(f"\nTest Paragraph:\n\"{self.paragraph}\"\n")
        
        # Get 3 random style images
        style_images = self.get_random_style_images(n=3)
        
        if not style_images:
            print("❌ No images found in dataset!")
            return
        
        # Generate for each style
        for idx, style_img_path in enumerate(style_images, 1):
            style_name = f"writer_{idx}"
            
            # Create style folder
            style_folder = self.output_dir / style_name
            style_folder.mkdir(exist_ok=True)
            
            # Generate
            style_ref, generated_paragraph = self.generate_paragraph_with_style(
                style_img_path, style_name
            )
            
            # Save images
            style_ref_path = style_folder / "reference_style.png"
            generated_path = style_folder / "generated_paragraph.png"
            
            Image.fromarray(style_ref).save(style_ref_path)
            Image.fromarray(generated_paragraph).save(generated_path)
            
            print(f"  ✅ Saved to: {style_folder}/")
            print(f"     - reference_style.png (original handwriting sample)")
            print(f"     - generated_paragraph.png (paragraph in this style)")
        
        print(f"\n{'='*70}")
        print(f"✨ Done! Check results in: {self.output_dir}/")
        print(f"{'='*70}\n")


if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.isTrain = False
    
    tester = ParagraphTester(opt)
    tester.run_test()
