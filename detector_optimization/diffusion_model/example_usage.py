
"""
Example usage script for TamboDiffusionGenerator class
"""

from detector_optimization.diffusion_model.tambo_diffusion_generator import TamboDiffusionGenerator

# ============================================================================
# Example 1: Complete pipeline with default settings
# ============================================================================

def example_full_pipeline():
    """Run the complete generation pipeline in one call."""

    generator = TamboDiffusionGenerator(
        checkpoint_path="/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_ckpts/diffusion/ckpt_epoch=1999.ckpt",
        output_dir="diffusion_model/run_3",
        tambo_optimization_path="/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_optimization",
        device="cuda:0",  # or "cpu", or None for auto-detection
    )

    # Run everything with one call
    generator.run_full_pipeline(
        num_samples=1000,     # Generate 1000 samples per condition
        num_conditions=10,    # Use 10 test conditions
        chunk_size=200,       # Process 200 at a time to avoid OOM
        plot_dpi=300          # High quality plots
    )


# ============================================================================
# Example 2: Step-by-step control
# ============================================================================

def example_step_by_step():
    """Run the pipeline with manual control over each step."""

    # Initialize with custom parameters
    generator = TamboDiffusionGenerator(
        checkpoint_path="/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_ckpts/diffusion/ckpt_epoch=1999.ckpt",
        output_dir="diffusion_model/run_4",
        tambo_optimization_path="/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_optimization",
        device="cuda:0",
        ddim_steps=100,       # DDIM sampling steps
        ddim_eta=0.0,         # Deterministic sampling
        batch_size=64,
        num_workers=4,
        seed=42,
    )

    # Step 1: Load model
    generator.load_model()

    # Step 2: Setup data
    generator.setup_data()

    # Step 3: Extract test samples
    generator.extract_test_samples(num_conditions=5)

    # Step 4: Generate samples
    generator.generate_samples(
        num_samples=500,   # Fewer samples for faster testing
        chunk_size=100,    # Smaller chunks if GPU memory is limited
    )

    # Step 5: Save results
    generator.save_results()

    # Step 6: Create plots
    generator.plot_results(num_conditions=5, dpi=150)


# ============================================================================
# Example 3: Generate for specific conditions only
# ============================================================================

def example_specific_conditions():
    """Extract many conditions but generate for only a subset."""

    generator = TamboDiffusionGenerator(
        checkpoint_path="/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_ckpts/diffusion/ckpt_epoch=1999.ckpt",
        output_dir="diffusion_model/run_5",
        tambo_optimization_path="/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_optimization",
    )

    generator.load_model()
    generator.setup_data()

    # Extract 20 conditions
    generator.extract_test_samples(num_conditions=20)

    # But only generate for the first 5
    generator.generate_samples(
        num_samples=200,
        num_conditions=5,  # Only process first 5
        chunk_size=50
    )

    generator.save_results()
    generator.plot_results()


# ============================================================================
# Example 4: Quick test run
# ============================================================================

def example_quick_test():
    """Quick test with minimal samples."""

    generator = TamboDiffusionGenerator(
        checkpoint_path="/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_ckpts/diffusion/ckpt_epoch=1999.ckpt",
        output_dir="diffusion_model/test_run",
        tambo_optimization_path="/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_optimization",
    )

    generator.run_full_pipeline(
        num_samples=50,      # Just 50 samples for testing
        num_conditions=2,    # Only 2 conditions
        chunk_size=25,       # Small chunks
        plot_dpi=100         # Lower DPI for speed
    )


# ============================================================================
# Run examples
# ============================================================================

if __name__ == "__main__":
    # Choose which example to run

    # Full pipeline (recommended for production)
    # example_full_pipeline()

    # Step-by-step (for debugging or custom workflows)
    # example_step_by_step()

    # Specific conditions (to save time)
    # example_specific_conditions()

    # Quick test (for validation)
    example_quick_test()
