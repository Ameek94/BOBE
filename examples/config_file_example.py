#!/usr/bin/env python3
"""
Example showing how to save and load JaxBo settings to/from configuration files.

This demonstrates how to use JSON and YAML files to store and retrieve
JaxBo configurations for reproducible runs.
"""

import json
import os
from pathlib import Path

# For YAML support (optional)
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    print("YAML support not available. Install PyYAML for YAML configuration files.")

from jaxbo.settings import (
    JaxBoSettings, BOBESettings, GPSettings, 
    get_high_dimensional_settings, get_fast_settings
)


def save_settings_json(settings: JaxBoSettings, filename: str):
    """Save settings to a JSON file."""
    config_dict = settings.to_dict()
    
    with open(filename, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"Settings saved to {filename}")


def load_settings_json(filename: str) -> JaxBoSettings:
    """Load settings from a JSON file."""
    with open(filename, 'r') as f:
        config_dict = json.load(f)
    
    settings = JaxBoSettings.from_dict(config_dict)
    print(f"Settings loaded from {filename}")
    return settings


def save_settings_yaml(settings: JaxBoSettings, filename: str):
    """Save settings to a YAML file (requires PyYAML)."""
    if not HAS_YAML:
        print("YAML support not available. Install PyYAML to use this function.")
        return
    
    config_dict = settings.to_dict()
    
    with open(filename, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    print(f"Settings saved to {filename}")


def load_settings_yaml(filename: str) -> JaxBoSettings:
    """Load settings from a YAML file (requires PyYAML)."""
    if not HAS_YAML:
        print("YAML support not available. Install PyYAML to use this function.")
        return JaxBoSettings()
    
    with open(filename, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    settings = JaxBoSettings.from_dict(config_dict)
    print(f"Settings loaded from {filename}")
    return settings


def create_example_configurations():
    """Create and save example configuration files."""
    print("=== Creating Example Configuration Files ===")
    
    # Create output directory
    config_dir = Path("example_configs")
    config_dir.mkdir(exist_ok=True)
    
    # 1. Default configuration
    default_settings = JaxBoSettings()
    save_settings_json(default_settings, config_dir / "default_config.json")
    
    if HAS_YAML:
        save_settings_yaml(default_settings, config_dir / "default_config.yaml")
    
    # 2. High-dimensional problem configuration
    high_dim_settings = get_high_dimensional_settings()
    save_settings_json(high_dim_settings, config_dir / "high_dimensional_config.json")
    
    # 3. Fast testing configuration
    fast_settings = get_fast_settings()
    save_settings_json(fast_settings, config_dir / "fast_testing_config.json")
    
    # 4. Custom cosmology configuration
    cosmo_settings = JaxBoSettings()
    
    # Customize for cosmological likelihoods
    cosmo_settings.bobe = BOBESettings(
        maxiters=2000,
        use_clf=True,
        clf_type='svm',
        clf_threshold=300,
        lengthscale_priors='SAAS',
        mc_points_method='NUTS',
        logz_threshold=0.5
    )
    
    cosmo_settings.gp = GPSettings(
        kernel='rbf',
        fit_maxiter=200,
        noise=1e-8,
        lengthscale_bounds=[-2, 2]
    )
    
    save_settings_json(cosmo_settings, config_dir / "cosmology_config.json")
    
    print(f"Configuration files created in {config_dir}/")
    return config_dir


def demonstrate_config_loading():
    """Demonstrate loading and using configuration files."""
    print("\n=== Loading and Using Configuration Files ===")
    
    config_dir = Path("example_configs")
    
    if not config_dir.exists():
        print("Configuration directory not found. Creating examples first...")
        create_example_configurations()
    
    # Load different configurations
    configs = {
        "default": load_settings_json(config_dir / "default_config.json"),
        "fast": load_settings_json(config_dir / "fast_testing_config.json"),
        "cosmology": load_settings_json(config_dir / "cosmology_config.json")
    }
    
    # Compare settings
    print("\nConfiguration comparison:")
    for name, settings in configs.items():
        print(f"\n{name.capitalize()} configuration:")
        print(f"  BOBE maxiters: {settings.bobe.maxiters}")
        print(f"  BOBE use_clf: {settings.bobe.use_clf}")
        print(f"  GP kernel: {settings.gp.kernel}")
        print(f"  GP fit_maxiter: {settings.gp.fit_maxiter}")
        print(f"  NS dlogz: {settings.nested_sampling.dlogz}")


def demonstrate_partial_config_override():
    """Show how to load a base config and override specific settings."""
    print("\n=== Partial Configuration Override ===")
    
    config_dir = Path("example_configs")
    
    # Load base configuration
    base_settings = load_settings_json(config_dir / "default_config.json")
    print(f"Base maxiters: {base_settings.bobe.maxiters}")
    
    # Override specific settings
    base_settings.bobe.maxiters = 3000
    base_settings.bobe.use_clf = False
    base_settings.gp.kernel = 'matern'
    
    print(f"Modified maxiters: {base_settings.bobe.maxiters}")
    print(f"Modified use_clf: {base_settings.bobe.use_clf}")
    print(f"Modified kernel: {base_settings.gp.kernel}")
    
    # Save the modified configuration
    save_settings_json(base_settings, config_dir / "modified_config.json")


def demonstrate_environment_based_config():
    """Show how to use different configs based on environment variables."""
    print("\n=== Environment-Based Configuration ===")
    
    config_dir = Path("example_configs")
    
    # Check environment variable for configuration choice
    config_type = os.environ.get('JAXBO_CONFIG', 'default')
    
    config_files = {
        'default': 'default_config.json',
        'fast': 'fast_testing_config.json',
        'cosmology': 'cosmology_config.json',
        'high_dim': 'high_dimensional_config.json'
    }
    
    config_file = config_files.get(config_type, 'default_config.json')
    
    print(f"Using configuration type: {config_type}")
    print(f"Loading configuration file: {config_file}")
    
    if (config_dir / config_file).exists():
        settings = load_settings_json(config_dir / config_file)
        print(f"Configuration loaded successfully!")
        print(f"  Max iterations: {settings.bobe.maxiters}")
        print(f"  Use classifier: {settings.bobe.use_clf}")
    else:
        print(f"Configuration file not found: {config_dir / config_file}")
        print("Using default settings instead.")
        settings = JaxBoSettings()
    
    return settings


def create_run_specific_config():
    """Create a configuration for a specific run with metadata."""
    print("\n=== Run-Specific Configuration ===")
    
    # Create a configuration with metadata
    config = {
        'metadata': {
            'description': 'Configuration for Planck+DESI CPL analysis',
            'created_by': 'user_name',
            'created_date': '2025-01-08',
            'problem_type': 'cosmological_parameter_estimation',
            'dimensions': 7,
            'expected_runtime_hours': 24
        },
        'bobe': BOBESettings(
            maxiters=2500,
            use_clf=True,
            clf_type='svm',
            lengthscale_priors='SAAS'
        ).to_dict(),
        'gp': GPSettings(
            kernel='rbf',
            fit_maxiter=250
        ).to_dict(),
        'nested_sampling': {
            'dlogz': 0.1,
            'maxcall': 10000000
        }
    }
    
    # Save with metadata
    config_file = Path("example_configs") / "planck_desi_cpl_config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Run-specific configuration saved to {config_file}")
    
    # Show how to load and use it
    with open(config_file, 'r') as f:
        loaded_config = json.load(f)
    
    print("\nMetadata:")
    for key, value in loaded_config['metadata'].items():
        print(f"  {key}: {value}")
    
    # Extract just the settings part
    settings_dict = {k: v for k, v in loaded_config.items() if k != 'metadata'}
    settings = JaxBoSettings.from_dict(settings_dict)
    
    return settings


if __name__ == "__main__":
    # Create example configurations
    create_example_configurations()
    
    # Demonstrate loading
    demonstrate_config_loading()
    
    # Show partial override
    demonstrate_partial_config_override()
    
    # Environment-based configuration
    demonstrate_environment_based_config()
    
    # Run-specific configuration
    create_run_specific_config()
    
    print("\n=== Summary ===")
    print("Configuration file features:")
    print("• Save/load settings to JSON or YAML files")
    print("• Support for partial configuration override")
    print("• Environment-based configuration selection")
    print("• Metadata support for run documentation")
    print("• Easy integration with existing workflows")
    print("\nUsage examples:")
    print("• export JAXBO_CONFIG=fast && python my_script.py")
    print("• settings = load_settings_json('my_config.json')")
    print("• bobe = BOBE(loglikelihood=ll, **settings.bobe.to_dict())")
