def traverse_ekm(grid, strategy="affect_balanced", affect_weights=None):
    """
    Generates an optimal path through an EKM grid based on specified strategy.
    
    Parameters:
    -----------
    grid : list of lists
        The EKM grid to traverse
    strategy : str
        The traversal strategy - options include:
        - "affect_balanced" : Balances main and anti-diagonal affects
        - "affect_main" : Maximizes main diagonal affect
        - "affect_anti" : Maximizes anti-diagonal affect
        - "constraint_preserving" : Prioritizes satisfaction of column constraints
    affect_weights : dict, optional
        Custom weighting for different affect categories
        
    Returns:
    --------
    path : list of tuples
        List of (row, col) coordinates defining the traversal
    metrics : dict
        Evaluation metrics for the generated path
    """
    rows, cols = len(grid), len(grid[0])
    path = []
    metrics = {
        'main_diagonal_presence': 0.0,
        'anti_diagonal_presence': 0.0,
        'constraint_satisfaction': 0.0,
        'semantic_coherence': 0.0
    }
    
    # Implementation would vary based on strategy
    if strategy == "affect_balanced":
        # Algorithm to balance encountering cells from both diagonals
        # while maintaining reasonable semantic coherence
        pass
    
    elif strategy == "affect_main":
        # Algorithm prioritizing main diagonal encounters
        pass
    
    # Calculate evaluation metrics for the generated path
    metrics = calculate_path_metrics(grid, path)
    
    return path, metrics
