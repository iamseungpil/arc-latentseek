#!/usr/bin/env python3
"""
Test GLM color perception vs our color mapping
"""

import sys
import os
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.executors.grid_renderer import COLORS, GridRenderer
from src.generators.barc_generator import COLOR_MAP
from src.executors.common_utils import Color

def create_color_test_grid():
    """Create a test grid with all 10 colors"""
    # Create a 2x5 grid with all colors 0-9
    grid = np.array([
        [0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9]
    ])
    return grid

def test_glm_color_perception():
    """Test what GLM actually sees vs what we think it sees"""
    
    print("=" * 60)
    print("GLM Color Perception Test")
    print("=" * 60)
    
    # Create test grid
    test_grid = create_color_test_grid()
    
    print("\n1. Test Grid Layout:")
    print("   Row 1: [0=Black, 1=Blue, 2=Red, 3=Green, 4=Yellow]")
    print("   Row 2: [5=Gray, 6=Pink, 7=Orange, 8=Teal, 9=Maroon]")
    
    # Check our color mappings
    print("\n2. Our Color Mappings:")
    print("   BARC Generator COLOR_MAP:")
    for i in range(10):
        color_name = COLOR_MAP.get(i, "MISSING")
        print(f"   {i}: {color_name}")
    
    print("\n   Grid Renderer COLORS (RGB):")
    for i in range(10):
        rgb = COLORS[i] if i < len(COLORS) else "MISSING"
        print(f"   {i}: {rgb}")
    
    # Render the test grid
    print("\n3. Rendering Test Grid...")
    renderer = GridRenderer("test_results")
    
    # Create a simple test problem structure
    class TestProblem:
        def __init__(self):
            self.uid = "color_test"
            self.train_pairs = []
    
    # Render the grid
    try:
        result = renderer._render_grid(test_grid, title="Color Test Grid")
        print(f"   ✅ Grid rendered successfully")
        print(f"   📁 Saved as: {result.image_path}")
        
        # Show what each color looks like
        print("\n4. Visual Color Analysis:")
        for i in range(10):
            rgb = COLORS[i]
            hex_color = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
            color_name = COLOR_MAP[i]
            
            # Basic color analysis
            brightness = (rgb[0] + rgb[1] + rgb[2]) / 3
            is_dark = brightness < 128
            dominant_channel = max(enumerate(rgb), key=lambda x: x[1])
            
            print(f"   {i}: {color_name}")
            print(f"      RGB: {rgb} | Hex: {hex_color}")
            print(f"      Brightness: {brightness:.1f} ({'Dark' if is_dark else 'Light'})")
            print(f"      Dominant: {'Red' if dominant_channel[0] == 0 else 'Green' if dominant_channel[0] == 1 else 'Blue'}")
            print()
        
        # Test specific problematic colors
        print("5. Critical Color Analysis (8=Teal, 9=Maroon):")
        
        # Color 8 (Teal)
        teal_rgb = COLORS[8]
        print(f"   Color 8 (Teal): {teal_rgb}")
        print(f"   - Should appear as light blue/cyan")
        print(f"   - RGB analysis: R={teal_rgb[0]}, G={teal_rgb[1]}, B={teal_rgb[2]}")
        if teal_rgb[2] > teal_rgb[0] and teal_rgb[1] > teal_rgb[0]:
            print("   ✅ Appears cyan/teal-like (high blue + green)")
        else:
            print("   ❌ May not appear teal-like")
        
        # Color 9 (Maroon)
        maroon_rgb = COLORS[9]
        print(f"   Color 9 (Maroon): {maroon_rgb}")
        print(f"   - Should appear as dark red")
        print(f"   - RGB analysis: R={maroon_rgb[0]}, G={maroon_rgb[1]}, B={maroon_rgb[2]}")
        if maroon_rgb[0] > maroon_rgb[1] and maroon_rgb[0] > maroon_rgb[2]:
            print("   ✅ Appears reddish (high red component)")
        else:
            print("   ❌ May not appear maroon-like")
            
        return True
        
    except Exception as e:
        print(f"   ❌ Error rendering grid: {e}")
        return False

def analyze_color_names_vs_perception():
    """Analyze if color names match visual perception"""
    
    print("\n" + "=" * 60)
    print("Color Name vs Visual Perception Analysis")
    print("=" * 60)
    
    # Define what colors should look like
    expected_appearance = {
        0: "Black - should be completely black",
        1: "Blue - should be clearly blue", 
        2: "Red - should be clearly red",
        3: "Green - should be clearly green",
        4: "Yellow - should be clearly yellow",
        5: "Gray - should be neutral gray",
        6: "Pink - should be pinkish/magenta",
        7: "Orange - should be orange",
        8: "Teal - should be blue-green/cyan",
        9: "Maroon - should be dark red"
    }
    
    print("\nExpected vs Actual Analysis:")
    for i in range(10):
        color_name = COLOR_MAP[i]
        rgb = COLORS[i]
        expected = expected_appearance[i]
        
        print(f"\n{i}. {color_name}")
        print(f"   Expected: {expected}")
        print(f"   RGB: {rgb}")
        
        # Analyze if RGB matches expectation
        r, g, b = rgb
        
        if i == 0:  # Black
            matches = r == 0 and g == 0 and b == 0
        elif i == 1:  # Blue
            matches = b > r and b > g
        elif i == 2:  # Red
            matches = r > g and r > b
        elif i == 3:  # Green
            matches = g > r and g > b
        elif i == 4:  # Yellow
            matches = r > 200 and g > 200 and b < 100
        elif i == 5:  # Gray
            matches = abs(r - g) < 30 and abs(g - b) < 30 and abs(r - b) < 30
        elif i == 6:  # Pink
            matches = r > 200 and b > 150 and g < 100
        elif i == 7:  # Orange
            matches = r > 200 and g > 100 and b < 50
        elif i == 8:  # Teal
            matches = b > 200 and g > 200 and r < 150
        elif i == 9:  # Maroon
            matches = r > 100 and g < 50 and b < 50
        else:
            matches = False
            
        status = "✅ Matches expectation" if matches else "❌ May not match expectation"
        print(f"   Analysis: {status}")

if __name__ == "__main__":
    print("Testing GLM color perception...")
    
    # Test 1: Basic color rendering
    success = test_glm_color_perception()
    
    # Test 2: Color name vs perception analysis
    analyze_color_names_vs_perception()
    
    if success:
        print("\n🎉 Color perception test completed!")
        print("📄 Check the rendered image to see what GLM actually sees")
        print("🔍 Compare with the analysis above")
    else:
        print("\n❌ Color perception test failed!")
        sys.exit(1)