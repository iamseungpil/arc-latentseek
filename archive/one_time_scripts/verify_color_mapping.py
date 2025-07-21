#!/usr/bin/env python3
"""
Verify color mapping consistency across all components
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.generators.barc_generator import COLOR_MAP
from src.executors.common import Color
from src.executors.grid_renderer import COLORS

def verify_color_mapping():
    """Verify that all color mappings are consistent"""
    
    print("=" * 60)
    print("Color Mapping Verification")
    print("=" * 60)
    
    # Expected standard ARC colors
    expected_colors = {
        0: "Black", 1: "Blue", 2: "Red", 3: "Green", 4: "Yellow",
        5: "Gray", 6: "Pink", 7: "Orange", 8: "Teal", 9: "Maroon"
    }
    
    # Check BARC generator color mapping
    print("\n1. BARC Generator Color Mapping:")
    print("   Index -> Name")
    for i in range(10):
        barc_color = COLOR_MAP.get(i, "MISSING")
        expected_color = expected_colors.get(i, "UNKNOWN")
        status = "✅" if barc_color == expected_color else "❌"
        print(f"   {i}: {barc_color} {status}")
    
    # Check Common.py Color constants
    print("\n2. Common.py Color Constants:")
    print("   Name -> Value")
    color_constants = {
        'BLACK': Color.BLACK,
        'BLUE': Color.BLUE,
        'RED': Color.RED,
        'GREEN': Color.GREEN,
        'YELLOW': Color.YELLOW,
        'GREY': Color.GREY,
        'PINK': Color.PINK,
        'ORANGE': Color.ORANGE,
        'TEAL': Color.TEAL,
        'MAROON': Color.MAROON
    }
    
    for name, value in color_constants.items():
        print(f"   Color.{name}: {value}")
    
    # Check Grid Renderer colors
    print("\n3. Grid Renderer Visual Colors:")
    print("   Index -> RGB")
    for i, rgb in enumerate(COLORS):
        print(f"   {i}: {rgb}")
    
    # Verify consistency
    print("\n4. Consistency Check:")
    issues = []
    
    # Check BARC -> Common.py consistency
    for i in range(10):
        barc_name = COLOR_MAP.get(i, "MISSING")
        expected_name = expected_colors.get(i, "UNKNOWN")
        
        if barc_name != expected_name:
            issues.append(f"BARC color {i}: expected '{expected_name}', got '{barc_name}'")
    
    # Check that we have the right number of visual colors
    if len(COLORS) != 10:
        issues.append(f"Grid renderer has {len(COLORS)} colors, expected 10")
    
    # Check specific problematic colors (8 and 9)
    if COLOR_MAP.get(8) != "Teal":
        issues.append(f"Color 8 should be 'Teal', got '{COLOR_MAP.get(8)}'")
    
    if COLOR_MAP.get(9) != "Maroon":
        issues.append(f"Color 9 should be 'Maroon', got '{COLOR_MAP.get(9)}'")
    
    if Color.TEAL != 8:
        issues.append(f"Color.TEAL should be 8, got {Color.TEAL}")
    
    if Color.MAROON != 9:
        issues.append(f"Color.MAROON should be 9, got {Color.MAROON}")
    
    if issues:
        print("\n❌ Issues found:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    else:
        print("\n✅ All color mappings are consistent!")
        return True

if __name__ == "__main__":
    success = verify_color_mapping()
    if not success:
        sys.exit(1)
    else:
        print("\n🎉 Color mapping verification passed!")