import pickle
import numpy as np

# Load the fresh results from the WIPV run
print("Loading BOBE results from the fresh WIPV run...")

try:
    with open('banana_results.pkl', 'rb') as f:
        results = pickle.load(f)
    print(f"Successfully loaded results: {type(results)}")
    
    # Examine the structure
    print(f"\nResults keys: {list(results.keys()) if isinstance(results, dict) else 'Not a dict'}")
    
    if isinstance(results, dict):
        for key, value in results.items():
            if hasattr(value, '__len__') and not isinstance(value, str):
                print(f"  {key}: {type(value)} with length {len(value) if hasattr(value, '__len__') else 'N/A'}")
            else:
                print(f"  {key}: {type(value)} = {value}")
        
        # Check if we have GP data
        if 'gp' in results:
            gp = results['gp']
            print(f"\nGP object: {type(gp)}")
            if hasattr(gp, '__dict__'):
                print(f"GP attributes: {list(gp.__dict__.keys())}")
        
        # Check for samples
        if 'samples' in results:
            samples = results['samples']
            print(f"\nSamples: {type(samples)}")
            if hasattr(samples, 'shape'):
                print(f"Samples shape: {samples.shape}")
                
        # Look for any tracking data
        for key in results.keys():
            if 'track' in key.lower() or 'history' in key.lower():
                print(f"\nFound tracking key: {key}")
                print(f"  Type: {type(results[key])}")
                if hasattr(results[key], '__len__'):
                    print(f"  Length: {len(results[key])}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
