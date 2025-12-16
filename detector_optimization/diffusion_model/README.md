# Tambo Diffusion Generator - Refactored Class

This refactoring converts your original script into a reusable, importable class structure.

## Files Created

1. **tambo_diffusion_generator.py** - Main class file
2. **example_usage.py** - Usage examples
3. **README.md** - This documentation

## Key Improvements

### 1. **Reusability**
- Import and use from any script
- No code duplication
- Easy to maintain

### 2. **Configurability**
All parameters are now configurable via constructor:
- Checkpoint path
- Output directory
- Device selection
- DDIM parameters (steps, eta)
- Data splitting ratios
- Batch size and workers
- Random seed

### 3. **Modularity**
Separate methods for each task:
- `load_model()` - Load checkpoint and sampler
- `setup_data()` - Initialize data module
- `extract_test_samples()` - Get test conditions
- `generate_samples()` - Generate images
- `save_results()` - Save to disk
- `plot_results()` - Create visualizations

### 4. **Flexibility**
Two usage modes:
- **Full pipeline**: One call does everything
- **Step-by-step**: Fine-grained control

### 5. **Better Error Handling**
- Validates state before operations
- Clear error messages
- Automatic directory creation

## Quick Start

### Simple Usage (One-Liner Pipeline)

```python
from tambo_diffusion_generator import TamboDiffusionGenerator

generator = TamboDiffusionGenerator(
    checkpoint_path="/path/to/checkpoint.ckpt",
    output_dir="output/run_1",
    tambo_optimization_path="/path/to/tambo_optimization"
)

generator.run_full_pipeline(
    num_samples=1000,
    num_conditions=10,
    chunk_size=200
)
```

### Advanced Usage (Step-by-Step)

```python
from tambo_diffusion_generator import TamboDiffusionGenerator

# Initialize with custom parameters
generator = TamboDiffusionGenerator(
    checkpoint_path="/path/to/checkpoint.ckpt",
    output_dir="output/run_2",
    tambo_optimization_path="/path/to/tambo_optimization",
    device="cuda:0",
    ddim_steps=100,
    ddim_eta=0.0,
    batch_size=64,
    seed=42
)

# Run each step manually
generator.load_model()
generator.setup_data()
generator.extract_test_samples(num_conditions=20)
generator.generate_samples(num_samples=500, num_conditions=10, chunk_size=100)
generator.save_results()
generator.plot_results(num_conditions=10, dpi=300)
```

## Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `checkpoint_path` | str | Required | Path to trained model checkpoint |
| `output_dir` | str | Required | Directory for outputs |
| `device` | str | Auto | Device ('cuda:0', 'cpu', etc.) |
| `ddim_steps` | int | 100 | DDIM sampling steps |
| `ddim_eta` | float | 0.0 | DDIM eta (0=deterministic) |
| `batch_size` | int | 64 | Data loading batch size |
| `train_ratio` | float | 0.85 | Training data ratio |
| `val_ratio` | float | 0.10 | Validation data ratio |
| `test_ratio` | float | 0.05 | Test data ratio |
| `num_workers` | int | 4 | Data loading workers |
| `seed` | int | 42 | Random seed |
| `tambo_optimization_path` | str | None | Path to add to sys.path |

## Methods

### `load_model()`
Loads the checkpoint and creates the DDIM sampler.

### `setup_data()`
Initializes the data module and test dataloader.

### `extract_test_samples(num_conditions=10)`
Extracts test images and their conditioning vectors.

**Parameters:**
- `num_conditions` (int): Number of test samples to extract

### `generate_samples(num_samples=1000, num_conditions=None, chunk_size=200)`
Generates samples for each extracted condition.

**Parameters:**
- `num_samples` (int): Samples per condition
- `num_conditions` (int): Number of conditions to process (None = all)
- `chunk_size` (int): Batch size for generation (prevents OOM)

### `save_results()`
Saves generated images and conditions as compressed numpy bundles.

**Output files:**
- `condition_1.npz`, `condition_2.npz`, ... - Per-condition bundles
- `summary.npz` - Summary of all conditions

### `plot_results(num_conditions=None, dpi=300)`
Creates comparison plots for ground truth vs generated samples.

**Parameters:**
- `num_conditions` (int): Number of plots to create (None = all)
- `dpi` (int): Plot resolution

### `run_full_pipeline(num_samples=1000, num_conditions=10, chunk_size=200, plot_dpi=300)`
Executes the complete workflow in one call.

## Output Structure

```
output_dir/
├── condition_1.npz      # Bundle for condition 1
│   ├── input           # Condition vector (5,)
│   ├── target          # Ground truth image (3,32,32)
│   ├── output          # Generated images (N,3,32,32)
│   └── meta            # Metadata dict
├── condition_2.npz
├── ...
├── summary.npz          # Summary file
│   └── summary
│       ├── all_conditions
│       ├── total_images
│       └── num_conditions
├── condition_1.png      # Comparison plot
├── condition_2.png
└── ...
```

## Loading Saved Results

```python
import numpy as np

# Load a specific condition bundle
data = np.load("output_dir/condition_1.npz", allow_pickle=True)
bundle = data['bundle'].item()

condition = bundle['input']        # (5,) condition vector
ground_truth = bundle['target']    # (3,32,32) GT image
generated = bundle['output']       # (N,3,32,32) generated images
metadata = bundle['meta']          # Dict with info

# Load summary
summary_data = np.load("output_dir/summary.npz", allow_pickle=True)
summary = summary_data['summary'].item()

all_conditions = summary['all_conditions']
total_images = summary['total_images']
```

## Migration from Original Script

### Before (Original Script)
```python
# Hard-coded paths and parameters
ckpt_path = "/n/holylfs05/.../ckpt_epoch=1999.ckpt"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# ... 200+ lines of procedural code ...
```

### After (Class-Based)
```python
# Configurable and reusable
generator = TamboDiffusionGenerator(
    checkpoint_path="/path/to/ckpt.ckpt",
    output_dir="my_output",
    device="cuda:0"
)
generator.run_full_pipeline(num_samples=1000, num_conditions=10)
```

## Tips

1. **Memory Management**: Adjust `chunk_size` based on GPU memory
   - 4GB GPU: chunk_size=50-100
   - 8GB GPU: chunk_size=100-200
   - 16GB+ GPU: chunk_size=200-500

2. **Quick Testing**: Use small values first
   ```python
   generator.run_full_pipeline(
       num_samples=50,
       num_conditions=2,
       chunk_size=25
   )
   ```

3. **Production Runs**: Increase after testing
   ```python
   generator.run_full_pipeline(
       num_samples=10000,
       num_conditions=100,
       chunk_size=500
   )
   ```

4. **Debugging**: Use step-by-step mode to inspect intermediate results

## Requirements

Same as original script:
- PyTorch
- NumPy
- Matplotlib
- Your `diffusion_train` module
- Your `models.DiffusionCondition` module

## Support

For issues or questions, refer to:
- `example_usage.py` for practical examples
- Original script comments for algorithm details
- Class docstrings for method documentation
