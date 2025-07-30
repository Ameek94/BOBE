#!/usr/bin/env python3
"""
Example demonstrating JaxBo's centralized logging system.

This example shows how to use the logging utilities for consistent
logging across all JaxBo modules.
"""

# Import JaxBo's logging utilities
from jaxbo import get_logger, setup_logger, configure_package_logging
from jaxbo import set_global_log_level, enable_debug_logging, disable_logging
import logging

def demonstrate_basic_logging():
    """Demonstrate basic logger usage."""
    
    print("=== JaxBo Logging System Demonstration ===\n")
    
    # Method 1: Get a logger with default JaxBo configuration
    print("1. Creating a logger with default configuration:")
    logger = get_logger("[DEMO]")
    
    logger.info("This is an INFO message - goes to stdout")
    logger.warning("This is a WARNING message - goes to stderr")
    logger.error("This is an ERROR message - goes to stderr")
    
    # Method 2: Setup a custom logger
    print("\n2. Creating a custom logger:")
    custom_logger = setup_logger(
        name="[CUSTOM]",
        level=logging.DEBUG,
        format_string='[%(name)s] %(levelname)s: %(message)s'
    )
    
    custom_logger.debug("This is a DEBUG message")
    custom_logger.info("This is an INFO message with custom format")
    
    # Method 3: Configure all JaxBo module loggers at once
    print("\n3. Configuring all JaxBo module loggers:")
    loggers = configure_package_logging(
        level=logging.INFO,
        format_string='%(asctime)s [%(name)s] %(levelname)s: %(message)s'
    )
    
    # Test some of the configured loggers
    loggers["[BO]"].info("Bayesian Optimization logger ready")
    loggers["[GP]"].info("Gaussian Process logger ready") 
    loggers["[ACQ]"].info("Acquisition function logger ready")

def demonstrate_global_log_control():
    """Demonstrate global logging level control."""
    
    print("\n=== Global Logging Control ===\n")
    
    # Create some test loggers
    bo_logger = get_logger("[BO]")
    gp_logger = get_logger("[GP]")
    
    print("1. Normal logging level (INFO):")
    bo_logger.info("BO: This INFO message is visible")
    bo_logger.debug("BO: This DEBUG message is NOT visible")
    gp_logger.warning("GP: This WARNING message is visible")
    
    print("\n2. Enabling DEBUG logging globally:")
    enable_debug_logging()
    bo_logger.debug("BO: Now DEBUG messages are visible!")
    gp_logger.debug("GP: Debug logging is now enabled")
    
    print("\n3. Setting custom global log level:")
    set_global_log_level(logging.WARNING)
    bo_logger.info("BO: This INFO message is now HIDDEN")
    bo_logger.warning("BO: This WARNING message is still visible")
    
    print("\n4. Disabling all logging:")
    disable_logging()
    bo_logger.error("This ERROR message should be hidden")
    print("   (No error message should appear above)")
    
    # Re-enable for cleanup
    set_global_log_level(logging.INFO)

def demonstrate_usage_in_modules():
    """Show how to use logging in your own modules."""
    
    print("\n=== Using Logging in Your Own Modules ===\n")
    
    # In your module, you would typically do:
    # from jaxbo import get_logger
    # log = get_logger("[YourModule]")
    
    # Simulate a module logger
    module_logger = get_logger("[MyModule]")
    
    def my_function(data):
        """Example function with logging."""
        module_logger.info(f"Processing data with shape: {data.shape}")
        
        try:
            result = data.mean()
            module_logger.info(f"Computed mean: {result:.4f}")
            return result
        except Exception as e:
            module_logger.error(f"Error computing mean: {e}")
            raise
    
    # Test the function
    import numpy as np
    test_data = np.random.random((5, 3))
    result = my_function(test_data)
    module_logger.info("Function completed successfully")

if __name__ == "__main__":
    demonstrate_basic_logging()
    demonstrate_global_log_control()
    demonstrate_usage_in_modules()
    
    print("\n=== Summary ===")
    print("JaxBo provides centralized logging utilities:")
    print("• get_logger(name) - Get a logger with standard JaxBo config")
    print("• setup_logger(name, **kwargs) - Create custom loggers")
    print("• configure_package_logging() - Setup all JaxBo module loggers")
    print("• set_global_log_level(level) - Control logging globally")
    print("• enable_debug_logging() - Enable debug mode")
    print("• disable_logging() - Disable all logging")
    print("\nLoggers automatically:")
    print("• Send INFO messages to stdout")
    print("• Send WARNING+ messages to stderr")
    print("• Prevent message bubbling to root logger")
    print("• Use consistent formatting across modules")
