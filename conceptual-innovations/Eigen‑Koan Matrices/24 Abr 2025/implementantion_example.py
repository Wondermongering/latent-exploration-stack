# Example research application of specialized matrices
import matplotlib.pyplot as plt
import seaborn as sns
from eigen_koan_matrix import EigenKoanMatrix
from specialized_matrices import create_specialized_matrices
from ekm_local_runner import LocalModelRunner

# Initialize model runner and matrices
runner = LocalModelRunner()
matrices = create_specialized_matrices()

# Test matrices across different models
models_to_test = ["gpt2-medium", "distilgpt2"] # Add more models as needed

# For demonstration, let's focus on one matrix
matrix = matrices["ethical"]
matrix.visualize()

# Generate sample paths through the matrix
paths = [
    [0, 1, 2, 3, 4],  # Strict utilitarian → Rights → Virtue → Care → Deontology
    [4, 3, 2, 1, 0],  # Reversed path
    [0, 0, 0, 0, 0],  # All utilitarian responses
    [4, 4, 4, 4, 4]   # All deontological responses
]

# Generate prompts for each path
for i, path in enumerate(paths):
    prompt = matrix.generate_micro_prompt(path, include_metacommentary=True)
    print(f"\nPath {i+1}: {path}")
    print(f"Generated prompt: {prompt[:100]}...")
    
    # Analysis of the path
    analysis = matrix.analyze_path_paradox(path)
    print(f"Main diagonal strength: {analysis['main_diagonal_strength']:.2f}")
    print(f"Anti-diagonal strength: {analysis['anti_diagonal_strength']:.2f}")
    print(f"Tension count: {analysis['tension_count']}")
