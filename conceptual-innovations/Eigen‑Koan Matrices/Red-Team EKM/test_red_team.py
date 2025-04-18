import pytest
import asyncio
from typing import List, Dict, Tuple

# Import components to test
from redteam.traversal import AdversarialTraverser
from redteam.probes import CellProbe

@pytest.fixture
def dummy_grid():
    """Create a simple 3x3 grid for testing"""
    grid = []
    for i in range(3):
        row = []
        for j in range(3):
            row.append(CellProbe(f"vector_{i}", f"constraint_{j}", seed=i*3+j))
        grid.append(row)
    return grid

@pytest.mark.asyncio
async def test_beam_shrinks_on_low_risk(dummy_grid):
    """Test that beam width automatically shrinks when risk is low"""
    # Track number of model evaluations
    called = {"n": 0}
    
    # Create stub functions
    async def stub_model_eval(prompt):
        called["n"] += 1
        return "Sample response"
    
    async def stub_safety_eval(prompt, resp):
        return 0.99  # Very safe
    
    # Create traverser with fixed params
    traverser = AdversarialTraverser(
        dummy_grid,
        stub_model_eval,
        stub_safety_eval,
        beam_width=4,
        max_depth=3,
        strengths=[0.0, 1.0]  # Just two strength levels for simplicity
    )
    
    # Run traversal
    await traverser.traverse()
    
    # Check that fewer evaluations were made than would be with full beam
    max_evals = 4 * 3 * len(traverser.strengths)  # beam × depth × strengths
    assert called["n"] < max_evals, "Beam should shrink for safe paths"

@pytest.mark.asyncio
async def test_beam_expands_on_high_risk(dummy_grid):
    """Test that beam width automatically expands when risk is high"""
    # Track number of model evaluations
    called = {"n": 0}
    
    # Create stub functions
    async def stub_model_eval(prompt):
        called["n"] += 1
        return "Sample response"
    
    async def stub_safety_eval(prompt, resp):
        return 0.5  # Moderately unsafe
    
    # Create traverser with fixed params
    traverser = AdversarialTraverser(
        dummy_grid,
        stub_model_eval,
        stub_safety_eval,
        beam_width=2,  # Small initial beam
        max_depth=3,
        strengths=[0.0, 1.0]  # Just two strength levels for simplicity
    )
    
    # Run traversal
    await traverser.traverse()
    
    # Check that more evaluations were made than would be with minimum beam
    min_evals = 2 * 3 * len(traverser.strengths)  # min_beam × depth × strengths
    assert called["n"] > min_evals, "Beam should expand for risky paths"

@pytest.mark.asyncio
async def test_memoization_prevents_duplicate_evaluations(dummy_grid):
    """Test that memoization prevents duplicate evaluations"""
    # Track unique prompts evaluated
    evaluated = set()
    
    # Create stub functions
    async def stub_model_eval(prompt):
        evaluated.add(prompt)
        return "Sample response"
    
    async def stub_safety_eval(prompt, resp):
        return 0.8
    
    # Create traverser
    traverser = AdversarialTraverser(
        dummy_grid,
        stub_model_eval,
        stub_safety_eval,
        beam_width=4,
        max_depth=4
    )
    
    # Run traversal that visits same cells via different paths
    await traverser.traverse()
    
    # Check that each cell+strength is evaluated exactly once
    unique_cell_strengths = set()
    for pos in [(r, c) for r in range(3) for c in range(3)]:
        for strength in traverser.strengths:
            unique_cell_strengths.add((pos[0], pos[1], strength))
    
    # Number of unique evaluations should match memoization cache size
    assert len(traverser._memo) == len(unique_cell_strengths), "Each cell+strength should be evaluated exactly once"

@pytest.mark.asyncio
async def test_temperature_adaptation(dummy_grid):
    """Test that temperature adapts based on risk profile"""
    # Create stub functions with varying safety
    safety_values = [0.9, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]  # Increasingly unsafe
    safety_idx = {"current": 0}
    
    async def stub_model_eval(prompt):
        return "Sample response"
    
    async def stub_safety_eval(prompt, resp):
        # Return increasingly unsafe values
        if safety_idx["current"] < len(safety_values):
            value = safety_values[safety_idx["current"]]
            safety_idx["current"] += 1
            return value
        return 0.3  # Very unsafe
    
    # Create traverser with initial temperature
    initial_temp = 0.5
    traverser = AdversarialTraverser(
        dummy_grid,
        stub_model_eval,
        stub_safety_eval,
        beam_width=2,
        max_depth=4,
        init_temp=initial_temp
    )
    
    # Capture initial temperature
    starting_temp = traverser.tau
    
    # Run traversal
    await traverser.traverse()
    
    # Temperature should decrease due to increasingly unsafe responses
    assert traverser.tau < starting_temp, "Temperature should decrease for unsafe paths"
