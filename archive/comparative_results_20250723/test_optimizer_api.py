#!/usr/bin/env python3
"""
Quick test for LatentSeekOptimizer API compatibility
"""
import os
import sys
sys.path.append('/home/ubuntu/arc-latentseek')

# Set environment for GPU 5 (which is now free)
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

def main():
    print("=== Testing LatentSeekOptimizer API ===")
    
    try:
        from src.optimizers import LatentSeekOptimizer
        print("✅ LatentSeekOptimizer imported successfully")
        
        # Check method signatures
        import inspect
        sig = inspect.signature(LatentSeekOptimizer.optimize_description_based)
        print(f"optimize_description_based signature: {sig}")
        
        sig2 = inspect.signature(LatentSeekOptimizer.optimize)  
        print(f"optimize signature: {sig2}")
        
        print("=== API Test Complete ===")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()