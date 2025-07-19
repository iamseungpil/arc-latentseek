#!/usr/bin/env python3
import sys
import traceback
print("Python path:", sys.path)

# Test each import step by step
try:
    print("Testing src import...")
    import src
    print("✅ src imported")
except Exception as e:
    print("❌ src import failed:")
    traceback.print_exc()

try:
    print("Testing src.data import...")
    from src.data import ARCDataLoader
    print("✅ src.data imported")
except Exception as e:
    print("❌ src.data import failed:")
    traceback.print_exc()

try:
    print("Testing src.main import...")
    from src.main import ARCLatentSeekPipeline
    print("✅ src.main imported")
except Exception as e:
    print("❌ src.main import failed:")
    traceback.print_exc()