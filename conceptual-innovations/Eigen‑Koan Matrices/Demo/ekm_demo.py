#!/usr/bin/env python3
"""
EKM Minimal Working Demo - Loads an EKM from JSON and calculates traversal metrics.

Usage: python ekm_demo.py path/to/ekm.json
"""

import json
import sys
import numpy as np
from typing import List, Tuple, Dict, Any

# Type aliases
Pos = Tuple[int, int]  # Row, column position in grid
Path = List[Pos]       # List of positions forming a traversal


def load_ekm(json_path: str) -> Dict[str, Any]:
    """Load EKM grid from JSON file."""
    with open(json_path, 'r') as f:
        ekm_data = json.load(f)
    
    print(f"Loaded EKM: {ekm_data.get('title', 'Untitled')}")
    return ekm_data


def get_traversal_path(ekm_data: Dict[str, Any], path_name: str = None) -> Path:
    """
    Get a traversal path from EKM data.
    If path_name is provided, load that specific path.
    Otherwise, if example_paths exists, load the first one.
    If no paths exist, create a simple diagonal path.
    """
    if 'example_paths' in ekm_data and ekm_data['example_paths']:
        if path_name and path_name in ekm_data['example_paths']:
            path = ekm_data['example_paths'][path_name]
            print(f"Using specified path: {path_name}")
        else:
            # Use first available path
            first_path_name = next(iter(ekm_data['example_paths']))
            path = ekm_data['example_paths'][first_path_name]
            print(f"Using example path: {first_path_name}")
    else:
        # Create a simple diagonal path if none exists
        grid = ekm_data['grid']
        rows = len(grid)
        cols = len(grid[0]) if rows > 0 else 0
        path = [(i, i) for i in range(min(rows, cols))]
        print(f"No paths found. Created diagonal path of length {len(path)}")
    
    return path


def calculate_metrics(ekm_data: Dict[str, Any], path: Path) -> Dict[str, float]:
    """Calculate metrics for a traversal path."""
    grid = ekm_data['grid']
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    
    # Extract path tokens
    try:
        path_tokens = [grid[r][c] for r, c in path]
        print("Path tokens:", path_tokens)
    except IndexError:
        print("Error: Path contains invalid positions for this grid")
        return {}
    
    # Calculate diagonal presence metrics
    main_diagonal_positions = {(i, i) for i in range(min(rows, cols))}
    anti_diagonal_positions = {(i, cols-1-i) for i in range(min(rows, cols))}
    
    main_diagonal_count = sum(1 for pos in path if pos in main_diagonal_positions)
    anti_diagonal_count = sum(1 for pos in path if pos in anti_diagonal_positions)
    
    # Calculate constraint satisfaction (% of unique columns visited)
    visited_columns = {c for _, c in path}
    constraint_satisfaction = len(visited_columns) / cols if cols > 0 else 0
    
    # Calculate task coverage (% of unique rows visited)
    visited_rows = {r for r, _ in path}
    task_coverage = len(visited_rows) / rows if rows > 0 else 0
    
    # Null token handling
    null_tokens = sum(1 for token in path_tokens if token in ["{NULL}", "[redacted]", ""])
    null_ratio = null_tokens / len(path) if path else 0
    
    metrics = {
        "path_length": len(path),
        "main_diagonal_presence": main_diagonal_count / len(path) if path else 0,
        "anti_diagonal_presence": anti_diagonal_count / len(path) if path else 0,
        "constraint_satisfaction": constraint_satisfaction,
        "task_coverage": task_coverage,
        "null_ratio": null_ratio,
    }
    
    # Add custom affect metrics if specified in EKM data
    if 'affect_dimensions' in ekm_data:
        for affect_name, affect_positions in ekm_data['affect_dimensions'].items():
            affect_positions = [tuple(pos) for pos in affect_positions]  # Convert list of lists to tuples
            affect_count = sum(1 for pos in path if pos in affect_positions)
            metrics[f"{affect_name}_presence"] = affect_count / len(path) if path else 0
    
    return metrics


def print_metrics(metrics: Dict[str, float]) -> None:
    """Print metrics in a formatted way."""
    print("\n=== Traversal Metrics ===")
    for name, value in metrics.items():
        # Format as percentage for ratio metrics
        if "ratio" in name or "presence" in name or "coverage" in name or "satisfaction" in name:
            print(f"{name}: {value:.2%}")
        else:
            print(f"{name}: {value}")


def main():
    """Main function to run the demo."""
    if len(sys.argv) < 2:
        print("Usage: python ekm_demo.py path/to/ekm.json [path_name]")
        print("Creating sample EKM for demonstration...")
        
        # Create a sample EKM
        sample_ekm = {
            "title": "Sample 3x3 EKM",
            "grid": [
                ["stone", "water", "light"],
                ["shadow", "vortex", "echo"],
                ["memory", "dream", "silence"]
            ],
            "affect_dimensions": {
                "melancholy": [(0, 0), (1, 1), (2, 2)],  # Main diagonal
                "wonder": [(2, 0), (1, 1), (0, 2)]       # Anti-diagonal
            },
            "example_paths": {
                "melancholy_path": [(0, 0), (1, 1), (2, 2)],
                "wonder_path": [(2, 0), (1, 1), (0, 2)],
                "mixed_path": [(0, 0), (1, 2), (2, 1)]
            }
        }
        
        # Write sample to file
        sample_path = "sample_ekm.json"
        with open(sample_path, "w") as f:
            json.dump(sample_ekm, f, indent=2)
        
        print(f"Created sample EKM at: {sample_path}")
        ekm_data = sample_ekm
        
        # Analyze all example paths
        for path_name in sample_ekm["example_paths"]:
            path = get_traversal_path(ekm_data, path_name)
            print(f"\nAnalyzing path: {path_name}")
            metrics = calculate_metrics(ekm_data, path)
            print_metrics(metrics)
        
        return
    
    # Load EKM from file
    json_path = sys.argv[1]
    path_name = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        ekm_data = load_ekm(json_path)
        path = get_traversal_path(ekm_data, path_name)
        metrics = calculate_metrics(ekm_data, path)
        print_metrics(metrics)
    except FileNotFoundError:
        print(f"Error: File {json_path} not found")
    except json.JSONDecodeError:
        print(f"Error: {json_path} is not a valid JSON file")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
