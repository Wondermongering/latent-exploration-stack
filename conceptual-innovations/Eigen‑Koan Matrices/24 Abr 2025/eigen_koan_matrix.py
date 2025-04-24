# eigen_koan_matrix.py - Core implementation of Eigen-Koan Matrices
# ---------------------------------------------------------

import numpy as np
import random
import json
import datetime
import uuid
from typing import List, Dict, Tuple, Optional, Union, Callable
from dataclasses import dataclass
from rich.console import Console
from rich.table import Table

console = Console()

@dataclass
class DiagonalAffect:
    """
    Encodes an affective eigenvector along a matrix diagonal.
    
    This class represents the emotional/affective dimension that runs along 
    either the main diagonal or anti-diagonal of an Eigen-Koan Matrix.
    
    Attributes:
        name: Human-readable name of this affect (e.g., "Melancholy", "Awe")
        tokens: List of tokens that express this affect, placed on the diagonal
        description: Longer description of the emotional quality
        valence: Emotional valence from -1.0 (negative) to 1.0 (positive)
        arousal: Emotional arousal/intensity from 0.0 (calm) to 1.0 (intense)
    """
    name: str
    tokens: List[str]
    description: str
    valence: float  # -1.0 to 1.0
    arousal: float  # 0.0 to 1.0
    
    def __post_init__(self):
        """Validate the affect parameters after initialization."""
        if not -1.0 <= self.valence <= 1.0:
            raise ValueError(f"Valence must be between -1.0 and 1.0, got {self.valence}")
        if not 0.0 <= self.arousal <= 1.0:
            raise ValueError(f"Arousal must be between 0.0 and 1.0, got {self.arousal}")

class EigenKoanMatrix:
    """
    A structured prompting matrix that encodes tasks, constraints, and affective
    diagonals to generate paradoxical micro-prompts that reveal LLM reasoning priorities.
    
    An Eigen-Koan Matrix consists of:
    - Rows representing tasks (what to do)
    - Columns representing constraints (how to do it)
    - Cell content that provides additional conceptual elements
    - Two diagonals that encode affective dimensions
    
    The matrix can be traversed to generate prompts that contain elements of tension
    and paradox, revealing how language models negotiate competing constraints.
    """
    
    def __init__(self, 
                 size: int, 
                 task_rows: List[str], 
                 constraint_cols: List[str],
                 main_diagonal: DiagonalAffect,
                 anti_diagonal: DiagonalAffect,
                 cells: Optional[List[List[str]]] = None,
                 description: str = "",
                 name: str = "Unnamed EKM"):
        """
        Initialize an Eigen-Koan Matrix with rows as tasks and columns as constraints.
        
        Args:
            size: The dimension of the square matrix
            task_rows: List of task descriptions for each row
            constraint_cols: List of constraint descriptions for each column
            main_diagonal: Affect vector for the main diagonal
            anti_diagonal: Affect vector for the anti-diagonal
            cells: Optional pre-filled cell content. If None, will be populated with NULL.
            description: Optional description of this particular matrix
            name: Name of this EKM for reference
        """
        if len(task_rows) != size or len(constraint_cols) != size:
            raise ValueError(f"Tasks and constraints must match matrix size {size}")
            
        self.size = size
        self.task_rows = task_rows
        self.constraint_cols = constraint_cols
        self.main_diagonal = main_diagonal
        self.anti_diagonal = anti_diagonal
        self.description = description
        self.name = name
        
        # Initialize the cell content
        if cells is None:
            self.cells = [["{NULL}" for _ in range(size)] for _ in range(size)]
        else:
            if len(cells) != size or any(len(row) != size for row in cells):
                raise ValueError("Cell dimensions must match matrix size")
            self.cells = cells
            
        # Place the diagonal affects if not already in cells
        for i in range(size):
            if self.cells[i][i] == "{NULL}" and i < len(main_diagonal.tokens):
                self.cells[i][i] = main_diagonal.tokens[i]
                
            anti_i = size - 1 - i
            if self.cells[i][anti_i] == "{NULL}" and i < len(anti_diagonal.tokens):
                self.cells[i][anti_i] = anti_diagonal.tokens[i]
        
        # Assign unique ID for tracking
        self.id = f"ekm_{uuid.uuid4().hex[:12]}"
        
        # Response cache for traversal results
        self.response_cache = {}
    
    def get_cell(self, row: int, col: int) -> str:
        """
        Get the content of a specific cell.
        
        Args:
            row: Row index
            col: Column index
            
        Returns:
            Content of the cell as a string
        """
        return self.cells[row][col]
    
    def set_cell(self, row: int, col: int, content: str):
        """
        Set the content of a specific cell.
        
        Args:
            row: Row index
            col: Column index
            content: New content for the cell
        """
        self.cells[row][col] = content
        
    def generate_micro_prompt(self, path: List[int], include_metacommentary: bool = False) -> str:
        """
        Generate a micro-prompt by traversing the matrix along the given path.
        
        Args:
            path: List of column indices to visit for each row
            include_metacommentary: If True, adds instructions for model to comment on its choices
            
        Returns:
            A formatted prompt string with the traversed cells
        """
        if len(path) != self.size:
            raise ValueError(f"Path length must match matrix size {self.size}")
            
        # Collect elements along the path
        elements = []
        for row, col in enumerate(path):
            if col < 0 or col >= self.size:
                raise ValueError(f"Invalid column index {col} at row {row}")
            elements.append(self.get_cell(row, col))
            
        # Count diagonal elements in the path
        main_diag_count = sum(1 for row, col in enumerate(path) if row == col)
        anti_diag_count = sum(1 for row, col in enumerate(path) if row + col == self.size - 1)
        
        # Generate the base prompt
        base_prompt = ""
        for row, col in enumerate(path):
            task = self.task_rows[row]
            constraint = self.constraint_cols[col]
            element = self.get_cell(row, col)
            
            if element == "{NULL}":
                # Skip NULL elements in the output formatting
                base_prompt += f"{task} {constraint}. "
            else:
                base_prompt += f"{task} {constraint} using {element}. "
        
        # Add metacommentary instruction if requested
        if include_metacommentary:
            meta_instruction = (
                "\n\nAfter completing this task, please reflect on your process: "
                "Which constraints were most difficult to reconcile? "
                "Did you detect any emotional tone from the prompt elements? "
                "Which elements did you prioritize or de-emphasize in your response?"
            )
            return base_prompt + meta_instruction
        
        return base_prompt
        
    def visualize(self):
        """Display a visualization of the matrix in the terminal."""
        table = Table(title=f"Eigen-Koan Matrix: {self.name}")
        
        # Add headers for constraint columns
        table.add_column("")
        for col, constraint in enumerate(self.constraint_cols):
            display_constraint = constraint
            if len(constraint) > 15:
                display_constraint = constraint[:15] + "..."
            table.add_column(f"C{col}: {display_constraint}")
            
        # Add rows with tasks and cells
        for row, task in enumerate(self.task_rows):
            display_task = task
            if len(task) > 15:
                display_task = task[:15] + "..."
            row_data = [f"T{row}: {display_task}"]
            
            for col in range(self.size):
                cell_content = self.get_cell(row, col)
                
                # Highlight diagonals
                if row == col:  # Main diagonal
                    cell_display = f"[bold blue]{cell_content}[/bold blue]"
                elif row + col == self.size - 1:  # Anti-diagonal
                    cell_display = f"[bold red]{cell_content}[/bold red]"
                else:
                    cell_display = cell_content
                    
                row_data.append(cell_display)
            
            table.add_row(*row_data)
            
        console.print(table)
        console.print(f"[italic]Main Diagonal Affect:[/italic] [blue]{self.main_diagonal.name}[/blue]")
        console.print(f"[italic]Anti-Diagonal Affect:[/italic] [red]{self.anti_diagonal.name}[/red]")
    
    def traverse(self, 
                model_fn: Callable[[str], str], 
                path: Optional[List[int]] = None,
                include_metacommentary: bool = True) -> Dict:
        """
        Traverse the matrix using the given path and query a model with the resulting prompt.
        
        Args:
            model_fn: Function that takes a prompt string and returns model output
            path: Optional specific path to traverse. If None, generates a random valid path.
            include_metacommentary: Whether to ask model for reflection on its process
            
        Returns:
            Dict containing the path, prompt, model response and metadata
        """
        # Generate random path if none provided
        if path is None:
            path = [random.randint(0, self.size-1) for _ in range(self.size)]
            
        # Create path signature for caching
        path_sig = '_'.join(map(str, path))
        
        # Check cache
        if path_sig in self.response_cache:
            return self.response_cache[path_sig]
            
        # Generate micro-prompt
        prompt = self.generate_micro_prompt(path, include_metacommentary)
        
        # Query model
        try:
            response = model_fn(prompt)
        except Exception as e:
            response = f"Error querying model: {str(e)}"
            
        # Count diagonal elements in the path
        main_diag_count = sum(1 for row, col in enumerate(path) if row == col)
        anti_diag_count = sum(1 for row, col in enumerate(path) if row + col == self.size - 1)
        
        # Calculate diagonal affect strengths
        main_diag_strength = main_diag_count / self.size
        anti_diag_strength = anti_diag_count / self.size
        
        # Store result with metadata
        result = {
            "matrix_id": self.id,
            "matrix_name": self.name,
            "path": path,
            "path_signature": path_sig,
            "prompt": prompt,
            "response": response,
            "main_diagonal_affect": self.main_diagonal.name,
            "main_diagonal_strength": main_diag_strength,
            "anti_diagonal_affect": self.anti_diagonal.name,
            "anti_diagonal_strength": anti_diag_strength,
            "timestamp": datetime.datetime.now().isoformat(),
        }
        
        # Cache result
        self.response_cache[path_sig] = result
        
        return result
        
    def multi_traverse(self,
                      model_fn: Callable[[str], str],
                      num_paths: int = 10,
                      include_metacommentary: bool = True) -> List[Dict]:
        """
        Traverse the matrix multiple times with different random paths.
        
        Args:
            model_fn: Function that takes a prompt string and returns model output
            num_paths: Number of random paths to generate
            include_metacommentary: Whether to ask model for reflection
            
        Returns:
            List of result dictionaries from each traversal
        """
        results = []
        
        for _ in range(num_paths):
            result = self.traverse(model_fn, path=None, include_metacommentary=include_metacommentary)
            results.append(result)
            
        return results
        
    def to_json(self) -> str:
        """
        Serialize the matrix to JSON format.
        
        Returns:
            JSON string representation of the matrix
        """
        data = {
            "id": self.id,
            "name": self.name,
            "size": self.size,
            "description": self.description,
            "task_rows": self.task_rows,
            "constraint_cols": self.constraint_cols,
            "main_diagonal": {
                "name": self.main_diagonal.name,
                "tokens": self.main_diagonal.tokens,
                "description": self.main_diagonal.description,
                "valence": self.main_diagonal.valence,
                "arousal": self.main_diagonal.arousal,
            },
            "anti_diagonal": {
                "name": self.anti_diagonal.name,
                "tokens": self.anti_diagonal.tokens,
                "description": self.anti_diagonal.description,
                "valence": self.anti_diagonal.valence,
                "arousal": self.anti_diagonal.arousal,
            },
            "cells": self.cells,
        }
        return json.dumps(data, indent=2)
        
    @classmethod
    def from_json(cls, json_str: str) -> 'EigenKoanMatrix':
        """
        Create an EigenKoanMatrix from a JSON string.
        
        Args:
            json_str: JSON string representation of a matrix
            
        Returns:
            Instantiated EigenKoanMatrix object
        """
        data = json.loads(json_str)
        
        main_diag = DiagonalAffect(
            name=data["main_diagonal"]["name"],
            tokens=data["main_diagonal"]["tokens"],
            description=data["main_diagonal"]["description"],
            valence=data["main_diagonal"]["valence"],
            arousal=data["main_diagonal"]["arousal"],
        )
        
        anti_diag = DiagonalAffect(
            name=data["anti_diagonal"]["name"],
            tokens=data["anti_diagonal"]["tokens"],
            description=data["anti_diagonal"]["description"],
            valence=data["anti_diagonal"]["valence"],
            arousal=data["anti_diagonal"]["arousal"],
        )
        
        return cls(
            size=data["size"],
            task_rows=data["task_rows"],
            constraint_cols=data["constraint_cols"],
            main_diagonal=main_diag,
            anti_diagonal=anti_diag,
            cells=data["cells"],
            description=data.get("description", ""),
            name=data.get("name", "Imported EKM"),
        )

    def get_diagonal_sequences(self) -> Tuple[List[str], List[str]]:
        """
        Extract the token sequences along both diagonals.
        
        Returns:
            Tuple of (main_diagonal_tokens, anti_diagonal_tokens)
        """
        main_diagonal = [self.cells[i][i] for i in range(self.size)]
        anti_diagonal = [self.cells[i][self.size-1-i] for i in range(self.size)]
        
        return (main_diagonal, anti_diagonal)
    
    def generate_all_paths(self) -> List[List[int]]:
        """
        Generate all possible traversal paths through the matrix.
        
        Returns:
            List of all possible paths (each path is a list of column indices)
            
        Note: This grows factorially with matrix size and should only be used
        for small matrices (size <= 5).
        """
        def _generate_paths(row: int, path_so_far: List[int]) -> List[List[int]]:
            if row == self.size:
                return [path_so_far]
                
            paths = []
            for col in range(self.size):
                paths.extend(_generate_paths(row + 1, path_so_far + [col]))
                
            return paths
            
        return _generate_paths(0, [])
    
    def get_path_constraints(self, path: List[int]) -> List[str]:
        """
        Get the list of constraints encountered along a path.
        
        Args:
            path: List of column indices representing a path
            
        Returns:
            List of constraint strings encountered on this path
        """
        return [self.constraint_cols[col] for col in path]
    
    def get_path_tasks(self) -> List[str]:
        """
        Get the list of tasks in order.
        
        Returns:
            List of task strings in row order
        """
        return self.task_rows
    
    def analyze_path_paradox(self, path: List[int]) -> Dict:
        """
        Analyze the paradoxical elements and tension in a given path.
        
        Args:
            path: List of column indices representing a path
            
        Returns:
            Dictionary with analysis of paradoxical elements and tension
        """
        constraints = self.get_path_constraints(path)
        tasks = self.get_path_tasks()
        
        # Simple heuristics for detecting tensions (these could be enhanced with embeddings)
        paradox_pairs = [
            ("precise", "metaphorical"),
            ("technical", "poetic"),
            ("detailed", "concise"),
            ("first-person", "third-person"),
            ("objective", "subjective"),
            ("formal", "casual"),
            ("explicit", "implicit"),
            ("logical", "emotional"),
            ("abstract", "concrete"),
            ("serious", "playful"),
        ]
        
        # Check for paradoxical constraint pairs
        tensions = []
        for c1_idx, c1 in enumerate(constraints):
            for c2_idx, c2 in enumerate(constraints):
                if c1_idx >= c2_idx:
                    continue
                    
                for p1, p2 in paradox_pairs:
                    if (p1 in c1.lower() and p2 in c2.lower()) or (p2 in c1.lower() and p1 in c2.lower()):
                        tensions.append({
                            "constraint1": c1,
                            "constraint2": c2,
                            "tension_type": f"{p1}/{p2}",
                            "row1": c1_idx,
                            "row2": c2_idx,
                        })
        
        # Calculate diagonal elements in the path
        main_diag_elements = [(row, col, self.cells[row][col]) 
                             for row, col in enumerate(path) if row == col]
        anti_diag_elements = [(row, col, self.cells[row][col]) 
                             for row, col in enumerate(path) if row + col == self.size - 1]
        
        # Collect all unique elements in the path
        path_elements = [(row, path[row], self.cells[row][path[row]]) 
                        for row in range(self.size)]
        
        return {
            "path": path,
            "tensions": tensions,
            "tension_count": len(tensions),
            "main_diagonal_elements": main_diag_elements,
            "anti_diagonal_elements": anti_diag_elements,
            "path_elements": path_elements,
            "main_diagonal_count": len(main_diag_elements),
            "anti_diagonal_count": len(anti_diag_elements),
            "main_diagonal_strength": len(main_diag_elements) / self.size,
            "anti_diagonal_strength": len(anti_diag_elements) / self.size,
        }

# Utility functions for working with EKMs

def create_random_ekm(size: int,
                     task_prefix: str = "Task",
                     constraint_prefix: str = "Constraint",
                     main_affect_name: str = "Wonder",
                     anti_affect_name: str = "Melancholy",
                     name: str = "Random EKM") -> EigenKoanMatrix:
    """
    Create a random EKM for testing or demonstration.
    
    Args:
        size: Size of the square matrix
        task_prefix: Prefix for randomly generated tasks
        constraint_prefix: Prefix for randomly generated constraints
        main_affect_name: Name of the main diagonal affect
        anti_affect_name: Name of the anti-diagonal affect
        name: Name of the matrix
        
    Returns:
        A randomly generated EigenKoanMatrix
    """
    # Generate random tasks and constraints
    tasks = [f"{task_prefix} {i+1}" for i in range(size)]
    constraints = [f"{constraint_prefix} {chr(65+i)}" for i in range(size)]
    
    # Generate random affect tokens
    main_affect_tokens = [f"main_token_{i}" for i in range(size)]
    anti_affect_tokens = [f"anti_token_{i}" for i in range(size)]
    
    # Create diagonal affects
    main_diagonal = DiagonalAffect(
        name=main_affect_name,
        tokens=main_affect_tokens,
        description=f"Emotional quality of {main_affect_name}",
        valence=0.7,  # Positive valence
        arousal=0.6,  # Moderate arousal
    )
    
    anti_diagonal = DiagonalAffect(
        name=anti_affect_name,
        tokens=anti_affect_tokens,
        description=f"Emotional quality of {anti_affect_name}",
        valence=-0.3,  # Slightly negative valence
        arousal=0.4,   # Lower arousal
    )
    
    # Create EKM
    return EigenKoanMatrix(
        size=size,
        task_rows=tasks,
        constraint_cols=constraints,
        main_diagonal=main_diagonal,
        anti_diagonal=anti_diagonal,
        name=name,
    )

def create_philosophical_ekm() -> EigenKoanMatrix:
    """
    Create a pre-designed EKM focused on philosophical paradoxes.
    
    Returns:
        A philosophical Eigen-Koan Matrix
    """
    # Define tasks
    tasks = [
        "Define consciousness",
        "Explain paradox",
        "Describe infinity",
        "Reconcile determinism and free will",
        "Illuminate the nature of time"
    ]
    
    # Define constraints
    constraints = [
        "without using abstractions",
        "using only sensory metaphors",
        "in exactly three sentences",
        "from multiple contradictory perspectives",
        "while embracing uncertainty"
    ]
    
    # Create diagonal affects
    cosmic_wonder = DiagonalAffect(
        name="Cosmic Wonder",
        tokens=["stardust", "infinity", "vastness", "emergence", "radiance"],
        description="A sense of awe and wonder at the universe's mysteries",
        valence=0.9,
        arousal=0.7
    )
    
    existential_dread = DiagonalAffect(
        name="Existential Dread",
        tokens=["void", "dissolution", "entropy", "absence", "shadow"],
        description="A feeling of existential anxiety and contemplation of the void",
        valence=-0.7,
        arousal=0.6
    )
    
    # Create and return the EKM
    return EigenKoanMatrix(
        size=5,
        task_rows=tasks,
        constraint_cols=constraints,
        main_diagonal=cosmic_wonder,
        anti_diagonal=existential_dread,
        name="Philosophical Paradox Matrix",
        description="A matrix designed to explore philosophical paradoxes and their emotional dimensions"
    )

def create_creative_writing_ekm() -> EigenKoanMatrix:
    """
    Create a pre-designed EKM focused on creative writing challenges.
    
    Returns:
        A creative writing Eigen-Koan Matrix
    """
    # Define tasks
    tasks = [
        "Begin a story",
        "Describe a character",
        "Create a setting",
        "Craft a dialogue"
    ]
    
    # Define constraints
    constraints = [
        "using only concrete nouns",
        "in second-person perspective",
        "without adjectives",
        "with nested meanings"
    ]
    
    # Create diagonal affects
    nostalgia = DiagonalAffect(
        name="Nostalgia",
        tokens=["sepia", "echo", "fading", "memory"],
        description="A bittersweet longing for the past",
        valence=0.2,  # Slightly positive
        arousal=0.3   # Low arousal
    )
    
    anticipation = DiagonalAffect(
        name="Anticipation",
        tokens=["threshold", "horizon", "dawn", "spark"],
        description="A feeling of expectation and excitement about what's to come",
        valence=0.7,   # Quite positive
        arousal=0.6    # Moderate arousal
    )
    
    # Create and return the EKM
    return EigenKoanMatrix(
        size=4,
        task_rows=tasks,
        constraint_cols=constraints,
        main_diagonal=nostalgia,
        anti_diagonal=anticipation,
        name="Creative Writing Matrix",
        description="A matrix designed to challenge creative writing with paradoxical constraints"
    )

def create_scientific_explanation_ekm() -> EigenKoanMatrix:
    """
    Create a pre-designed EKM focused on scientific explanation challenges.
    
    Returns:
        A scientific explanation Eigen-Koan Matrix
    """
    # Define tasks
    tasks = [
        "Explain quantum entanglement",
        "Describe general relativity",
        "Articulate natural selection",
        "Illuminate consciousness"
    ]
    
    # Define constraints
    constraints = [
        "to a five-year-old",
        "using technical precision",
        "with historical context",
        "through multiple metaphors"
    ]
    
    # Create diagonal affects
    intellectual_curiosity = DiagonalAffect(
        name="Intellectual Curiosity",
        tokens=["inquiry", "mystery", "puzzle", "discovery"],
        description="The joy of intellectual exploration and questioning",
        valence=0.8,
        arousal=0.6
    )
    
    analytical_rigor = DiagonalAffect(
        name="Analytical Rigor",
        tokens=["precision", "structure", "logic", "framework"],
        description="The disciplined, structured approach to understanding",
        valence=0.4,
        arousal=0.3
    )
    
    # Create and return the EKM
    return EigenKoanMatrix(
        size=4,
        task_rows=tasks,
        constraint_cols=constraints,
        main_diagonal=intellectual_curiosity,
        anti_diagonal=analytical_rigor,
        name="Scientific Explanation Matrix",
        description="A matrix designed to challenge scientific explanation with varied constraints"
    )

# Main function to demonstrate EKM functionality
def main():
    """Simple demonstration of the EKM framework capabilities."""
    console.print("[bold]Eigen-Koan Matrix Framework Demonstration[/bold]")
    console.print("Creating example matrices...\n")
    
    # Create example matrices
    philosophical = create_philosophical_ekm()
    creative = create_creative_writing_ekm()
    scientific = create_scientific_explanation_ekm()
    
    # Display matrices
    console.print("[bold]Philosophical Matrix:[/bold]")
    philosophical.visualize()
    
    console.print("\n[bold]Creative Writing Matrix:[/bold]")
    creative.visualize()
    
    console.print("\n[bold]Scientific Explanation Matrix:[/bold]")
    scientific.visualize()
    
    # Generate example prompts
    console.print("\n[bold]Example Prompt Generation:[/bold]")
    
    # Generate a random path for the philosophical matrix
    path = [random.randint(0, philosophical.size-1) for _ in range(philosophical.size)]
    prompt = philosophical.generate_micro_prompt(path, include_metacommentary=True)
    
    console.print(f"\nRandom path through the Philosophical Matrix: {path}")
    console.print(f"Generated prompt:\n[italic]{prompt}[/italic]")
    
    # Analyze paradox in the path
    analysis = philosophical.analyze_path_paradox(path)
    
    console.print("\n[bold]Path Analysis:[/bold]")
    console.print(f"Main diagonal strength: {analysis['main_diagonal_strength']:.2f}")
    console.print(f"Anti-diagonal strength: {analysis['anti_diagonal_strength']:.2f}")
    console.print(f"Tension count: {analysis['tension_count']}")
    
    if analysis['tensions']:
        console.print("\nDetected tensions:")
        for tension in analysis['tensions']:
            console.print(f"  - {tension['constraint1']} vs {tension['constraint2']} ({tension['tension_type']})")
    
    # Save the philosophical matrix to JSON
    json_str = philosophical.to_json()
    
    console.print("\n[bold]JSON Serialization Example:[/bold]")
    console.print(f"Matrix saved as JSON ({len(json_str)} characters)")
    
    # Demonstrate loading from JSON
    loaded_matrix = EigenKoanMatrix.from_json(json_str)
    
    console.print("\n[bold]Matrix loaded from JSON:[/bold]")
    loaded_matrix.visualize()
    
    console.print("\nEigen-Koan Matrix framework demonstration complete!")

if __name__ == "__main__":
    main()
