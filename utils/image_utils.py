"""
Image handling utilities for the Meme Generator.
"""

from PIL import Image, ImageDraw, ImageFont
import os
import numpy as np

def overlay_text_on_image(image_path, text, output_path=None):
    """Overlay text on an image to create a meme."""
    try:
        # Open the image
        img = Image.open(image_path)
        
        # Create a drawing context
        draw = ImageDraw.Draw(img)
        
        # Determine font size based on image dimensions
        width, height = img.size
        font_size = int(min(width, height) * 0.08)  # 8% of the smaller dimension
        
        # Try to use a bold font if available, otherwise use default
        try:
            # For different operating systems, try different font paths
            font_paths = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux
                "/Library/Fonts/Arial Bold.ttf",  # macOS
                "C:\\Windows\\Fonts\\arialbd.ttf",  # Windows
                "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf"  # Some Linux distributions
            ]
            
            font = None
            for path in font_paths:
                if os.path.exists(path):
                    font = ImageFont.truetype(path, font_size)
                    break
            
            if font is None:
                font = ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()
        
        # Determine text position (center top of image)
        text_width, text_height = draw.textsize(text, font=font)
        position = ((width - text_width) / 2, height * 0.1)  # Top center
        
        # Add text with black outline for readability
        # This simulates the classic meme text style
        outline_width = max(1, font_size // 15)
        
        # Draw text outline
        for offset_x in range(-outline_width, outline_width + 1):
            for offset_y in range(-outline_width, outline_width + 1):
                draw.text(
                    (position[0] + offset_x, position[1] + offset_y),
                    text,
                    font=font,
                    fill="black"
                )
        
        # Draw text in white
        draw.text(position, text, font=font, fill="white")
        
        # Save or return the image
        if output_path:
            img.save(output_path)
            return output_path
        else:
            return img
        
    except Exception as e:
        raise Exception(f"Failed to overlay text on image: {str(e)}")

def get_meme_image_dimensions(image_path):
    """Get the width and height of a meme template image."""
    try:
        with Image.open(image_path) as img:
            return img.size
    except Exception as e:
        raise Exception(f"Failed to get image dimensions: {str(e)}")