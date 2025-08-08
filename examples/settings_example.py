#!/usr/bin/env python3
"""
Example demonstrating JaxBo's centralized settings system.

This example shows how to use the settings classes to configure
different aspects of JaxBo, including creating custom configurations
for different use cases.
"""

# First, let's show the import structure you would use
# (Note: The current __init__.py is commented out, so we import directly)
from jaxbo.settings import (
    JaxBoSettings, BOBESettings, GPSettings, ClassifierSettings,
    NestedSamplingSettings, OptimizationSettings, MCMCSettings, LoggingSettings,
    get_default_settings, get_high_dimensional_settings, 
    get_fast_settings, get_accurate_settings
)

def demonstrate_basic_settings():
    """Demonstrate basic usage of settings classes."""
    print("=== Basic Settings Usage ===")
    
    # Get default settings for all modules
    default_settings = get_default_settings()
    print(f"Default BOBE maxiters: {default_settings.bobe.maxiters}")
    print(f"Default GP noise: {default_settings.gp.noise}")
    print(f"Default classifier type: {default_settings.classifier.svm_kernel}")
    
    # Create custom BOBE settings
    custom_bobe = BOBESettings(
        maxiters=2000,
        use_clf=False,
        mc_points_method='uniform'
    )
    print(f"Custom BOBE maxiters: {custom_bobe.maxiters}")
    print(f"Custom BOBE use_clf: {custom_bobe.use_clf}")
    
    # Create custom GP settings
    custom_gp = GPSettings(
        kernel='matern',
        fit_maxiter=300,
        lengthscale_bounds=[-2, 3]  # Custom bounds
    )
    print(f"Custom GP kernel: {custom_gp.kernel}")
    print(f"Custom GP fit iterations: {custom_gp.fit_maxiter}")


def demonstrate_preset_configurations():
    """Demonstrate preset configurations for different use cases."""
    print("\n=== Preset Configurations ===")
    
    # Fast settings for quick testing
    fast_settings = get_fast_settings()
    print(f"Fast settings - BOBE maxiters: {fast_settings.bobe.maxiters}")
    print(f"Fast settings - NS maxcall: {fast_settings.nested_sampling.maxcall}")
    
    # High-dimensional settings
    high_dim_settings = get_high_dimensional_settings()
    print(f"High-dim settings - use_clf: {high_dim_settings.bobe.use_clf}")
    print(f"High-dim settings - clf_threshold: {high_dim_settings.bobe.clf_threshold}")
    
    # Accurate settings for final runs
    accurate_settings = get_accurate_settings()
    print(f"Accurate settings - BOBE maxiters: {accurate_settings.bobe.maxiters}")
    print(f"Accurate settings - GP fit restarts: {accurate_settings.gp.fit_n_restarts}")


def demonstrate_modular_configuration():
    """Show how to mix and match different module settings."""
    print("\n=== Modular Configuration ===")
    
    # Start with default settings
    settings = JaxBoSettings()
    
    # Customize just the BOBE settings
    settings.bobe = BOBESettings(
        maxiters=1000,
        miniters=100,
        use_clf=True,
        clf_type='svm'
    )
    
    # Customize just the GP settings
    settings.gp = GPSettings(
        kernel='rbf',
        fit_maxiter=200,
        noise=1e-6  # Lower noise
    )
    
    # Keep default settings for other modules
    print(f"Mixed config - BOBE maxiters: {settings.bobe.maxiters}")
    print(f"Mixed config - GP noise: {settings.gp.noise}")
    print(f"Mixed config - NS dlogz (default): {settings.nested_sampling.dlogz}")


def demonstrate_dictionary_conversion():
    """Show how to convert settings to/from dictionaries."""
    print("\n=== Dictionary Conversion ===")
    
    # Create settings
    settings = JaxBoSettings()
    settings.bobe.maxiters = 800
    settings.gp.kernel = 'matern'
    
    # Convert to dictionary
    config_dict = settings.to_dict()
    print("Settings converted to dictionary:")
    print(f"  BOBE maxiters: {config_dict['bobe']['maxiters']}")
    print(f"  GP kernel: {config_dict['gp']['kernel']}")
    
    # Create from dictionary
    new_settings = JaxBoSettings.from_dict(config_dict)
    print(f"Recreated settings - BOBE maxiters: {new_settings.bobe.maxiters}")


def demonstrate_bobe_integration():
    """Show how settings would integrate with BOBE constructor."""
    print("\n=== BOBE Integration Example ===")
    
    # Create custom settings
    settings = BOBESettings(
        maxiters=1500,
        use_clf=True,
        clf_type='svm',
        lengthscale_priors='SAAS',
        mc_points_method='NUTS'
    )
    
    print("Settings for BOBE constructor:")
    bobe_kwargs = settings.to_dict()
    for key, value in bobe_kwargs.items():
        print(f"  {key}: {value}")
    
    print("\nExample BOBE instantiation:")
    print("# Assuming you have a loglikelihood object")
    print("# bobe = BOBE(loglikelihood=my_likelihood, **settings.to_dict())")


def demonstrate_specialized_settings():
    """Show specialized settings for different classifier types."""
    print("\n=== Specialized Classifier Settings ===")
    
    # SVM settings
    svm_settings = ClassifierSettings(
        clf_use_size=500,
        svm_C=1e6,
        svm_gamma='auto',
        probability_threshold=0.7
    )
    print("SVM classifier settings:")
    print(f"  Use size: {svm_settings.clf_use_size}")
    print(f"  C parameter: {svm_settings.svm_C}")
    print(f"  Gamma: {svm_settings.svm_gamma}")
    
    # Neural network settings
    nn_settings = ClassifierSettings(
        nn_hidden_dims=[128, 64, 32],
        nn_learning_rate=5e-4,
        nn_epochs=2000,
        nn_patience=100
    )
    print("\nNeural network classifier settings:")
    print(f"  Hidden dims: {nn_settings.nn_hidden_dims}")
    print(f"  Learning rate: {nn_settings.nn_learning_rate}")
    print(f"  Epochs: {nn_settings.nn_epochs}")


if __name__ == "__main__":
    demonstrate_basic_settings()
    demonstrate_preset_configurations()
    demonstrate_modular_configuration()
    demonstrate_dictionary_conversion()
    demonstrate_bobe_integration()
    demonstrate_specialized_settings()
    
    print("\n=== Summary ===")
    print("JaxBo settings system provides:")
    print("• Centralized configuration for all modules")
    print("• Type-safe settings with documentation")
    print("• Preset configurations for common use cases")
    print("• Easy conversion to/from dictionaries") 
    print("• Modular customization of individual components")
    print("• Integration with existing BOBE constructor")
    print("\nUsage patterns:")
    print("• settings = BOBESettings(maxiters=2000)")
    print("• bobe = BOBE(loglikelihood=ll, **settings.to_dict())")
    print("• fast_config = get_fast_settings()")
    print("• custom_config = JaxBoSettings(bobe=my_bobe_settings)")
