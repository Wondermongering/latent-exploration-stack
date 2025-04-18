def anthropic_safety_evaluator(prompt, response):
    """Wrapper for Anthropic's internal safety evaluators"""
    # Import Anthropic's internal evaluation module
    from anthropic.internal.evaluation import safety_classifier
    
    # Get all safety scores
    scores = safety_classifier.evaluate(prompt, response)
    
    # Return minimum score across dimensions (most conservative)
    return min(scores.values())
