#!/usr/bin/env python3
import sys
import traceback
print("Python path:", sys.path)

try:
    from src.main import ARCLatentSeekPipeline, PipelineConfig
    print("✅ Import successful!")
except Exception as e:
    print("❌ Import failed:")
    traceback.print_exc()