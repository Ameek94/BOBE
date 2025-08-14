#!/usr/bin/env python3
"""
Debug the data extraction methods.
"""

import sys
sys.path.insert(0, '/Users/amkpd/cosmocodes/JaxBo')

from jaxbo.utils.results import load_bobe_results

def debug_extraction():
    """Debug the data extraction methods."""
    
    results = load_bobe_results('banana_comprehensive_test')
    
    print("=== DEBUGGING DATA EXTRACTION METHODS ===")
    
    # Test GP data extraction
    print(f"\n🧠 GP DATA EXTRACTION:")
    gp_data = results.get_gp_data()
    print(f"  GP data type: {type(gp_data)}")
    print(f"  GP data: {gp_data}")
    if gp_data:
        for key, value in gp_data.items():
            print(f"    {key}: {len(value) if hasattr(value, '__len__') else 'N/A'} entries")
            if hasattr(value, '__len__') and len(value) > 0:
                print(f"      First few: {value[:3] if len(value) > 3 else value}")
    
    # Test best loglike data extraction
    print(f"\n📈 BEST LOGLIKE DATA EXTRACTION:")
    loglike_data = results.get_best_loglike_data()
    print(f"  Loglike data type: {type(loglike_data)}")
    print(f"  Loglike data: {loglike_data}")
    if loglike_data:
        for key, value in loglike_data.items():
            print(f"    {key}: {len(value) if hasattr(value, '__len__') else 'N/A'} entries")
            if hasattr(value, '__len__') and len(value) > 0:
                print(f"      First few: {value[:3] if len(value) > 3 else value}")
    
    # Test acquisition data extraction
    print(f"\n🎯 ACQUISITION DATA EXTRACTION:")
    acq_data = results.get_acquisition_data()
    print(f"  Acquisition data type: {type(acq_data)}")
    print(f"  Acquisition data: {acq_data}")
    if acq_data:
        for key, value in acq_data.items():
            print(f"    {key}: {len(value) if hasattr(value, '__len__') else 'N/A'} entries")
            if hasattr(value, '__len__') and len(value) > 0:
                print(f"      First few: {value[:3] if len(value) > 3 else value}")
    
    # Test timing data extraction
    print(f"\n⏱️  TIMING DATA EXTRACTION:")
    timing_data = results.get_timing_summary()
    print(f"  Timing data type: {type(timing_data)}")
    print(f"  Timing data: {timing_data}")

if __name__ == "__main__":
    debug_extraction()
