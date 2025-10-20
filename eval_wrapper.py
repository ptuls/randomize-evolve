"""Wrapper module to make evaluator importable for OpenEvolve multiprocessing."""
from evaluator import evaluate

# Make evaluate available at module level for multiprocessing
__all__ = ["evaluate"]


def evaluate_program(program_path: str):
    """Wrapper for OpenEvolve to call."""
    return evaluate(program_path)
