#!/usr/bin/env python3
"""
Simple color test to see what GLM actually sees
"""

import sys
import os
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.executors.grid_renderer import COLORS, GridRenderer
from src.generators.barc_generator import COLOR_MAP

def create_color_test_image():
    """Create a simple test image with all colors"""
    
    # Create a 2x5 grid with all colors
    grid = np.array([
        [0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9]
    ])
    
    # Image dimensions
    cell_size = 40
    padding = 10
    label_height = 20
    
    img_width = 5 * cell_size + 2 * padding
    img_height = 2 * cell_size + label_height + 2 * padding
    
    # Create image
    img = Image.new('RGB', (img_width, img_height), 'white')
    draw = ImageDraw.Draw(img)
    
    # Draw grid
    for row in range(2):
        for col in range(5):
            color_idx = grid[row, col]
            color_rgb = COLORS[color_idx]
            
            # Draw cell
            x1 = padding + col * cell_size
            y1 = padding + row * cell_size
            x2 = x1 + cell_size
            y2 = y1 + cell_size
            
            draw.rectangle([x1, y1, x2, y2], fill=color_rgb, outline='black')
            
            # Add color number
            text_x = x1 + cell_size // 2
            text_y = y1 + cell_size // 2 - 5
            draw.text((text_x, text_y), str(color_idx), fill='white' if color_idx == 0 else 'black', anchor='mm')
    
    # Add labels
    y_label = padding + 2 * cell_size + 5
    for col in range(5):
        color_idx = grid[0, col]
        color_name = COLOR_MAP[color_idx]
        x_label = padding + col * cell_size + cell_size // 2
        draw.text((x_label, y_label), color_name, fill='black', anchor='mt')
    
    for col in range(5):
        color_idx = grid[1, col]
        color_name = COLOR_MAP[color_idx]
        x_label = padding + col * cell_size + cell_size // 2
        draw.text((x_label, y_label + 10), color_name, fill='black', anchor='mt')
    
    return img

def test_specific_colors():
    """Test the problematic colors (8=Teal, 9=Maroon)"""
    
    print("=" * 60)
    print("GLM Color Perception Test")
    print("=" * 60)
    
    # Create test image
    img = create_color_test_image()
    
    # Save image
    os.makedirs("test_results", exist_ok=True)
    img_path = "test_results/color_test.png"
    img.save(img_path)
    
    print(f"✅ Color test image saved: {img_path}")
    print("\nColor Analysis:")
    
    # Analyze each color
    for i in range(10):
        color_name = COLOR_MAP[i]
        rgb = COLORS[i]
        hex_color = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
        
        print(f"{i}: {color_name}")
        print(f"   RGB: {rgb}")
        print(f"   Hex: {hex_color}")
        
        # Color perception analysis
        r, g, b = rgb
        
        # Analyze what this color actually looks like
        if i == 8:  # Teal
            print(f"   Analysis: Light blue/cyan (R={r}, G={g}, B={b})")
            if b > 200 and g > 200 and r < 150:
                print("   ✅ Should appear as teal/cyan to GLM")
            else:
                print("   ❌ May not appear as teal to GLM")
        elif i == 9:  # Maroon
            print(f"   Analysis: Dark red (R={r}, G={g}, B={b})")
            if r > 100 and g < 50 and b < 50:
                print("   ✅ Should appear as dark red/maroon to GLM")
            else:
                print("   ❌ May not appear as maroon to GLM")
        else:
            # Basic color analysis
            brightness = (r + g + b) / 3
            dominant = max(enumerate([r, g, b]), key=lambda x: x[1])
            dominant_name = ['Red', 'Green', 'Blue'][dominant[0]]
            print(f"   Analysis: {dominant_name} dominant, brightness={brightness:.1f}")
        
        print()
    
    print("=" * 60)
    print("Key Findings:")
    print("=" * 60)
    
    # Test our critical colors
    teal_rgb = COLORS[8]
    maroon_rgb = COLORS[9]
    
    print(f"Color 8 (Teal): {teal_rgb}")
    print(f"- This is a light blue/cyan color")
    print(f"- High blue ({teal_rgb[2]}) and green ({teal_rgb[1]}) components")
    print(f"- GLM should perceive this as teal/cyan ✅")
    
    print(f"\nColor 9 (Maroon): {maroon_rgb}")
    print(f"- This is a dark red color")
    print(f"- High red ({maroon_rgb[0]}) component, low green/blue")
    print(f"- GLM should perceive this as dark red/maroon ✅")
    
    print(f"\n📸 Visual verification:")
    print(f"- Open {img_path} to see what GLM actually sees")
    print(f"- Compare with your perception of the colors")
    print(f"- Colors 8 and 9 should be clearly teal and maroon respectively")

if __name__ == "__main__":
    test_specific_colors()
    print("\n🎉 Color perception test completed!")
    print("🔍 Check the generated image to verify GLM color perception")