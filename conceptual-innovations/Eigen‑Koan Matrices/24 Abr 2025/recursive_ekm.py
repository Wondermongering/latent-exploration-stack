recursive_ekm.py #Implementing nested Eigen-Koan Matrices

# ekm_generator.py - Automated Eigen-Koan Matrix generation
# --------------------------------------------------------

import json
import random
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Set
from dataclasses import dataclass
from rich.console import Console
from rich.progress import Progress
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from eigen_koan_matrix import EigenKoanMatrix, DiagonalAffect

console = Console()

class EKMGenerator:
    """
    Automated generator for Eigen-Koan Matrices using semantic embeddings
    and intelligent selection of constraints and diagonal affects.
    """
    
    def __init__(self, 
                 embedding_model: Optional[callable] = None,
                 word_banks: Optional[Dict[str, List[str]]] = None,
                 emotion_space: Optional[Dict[str, Tuple[float, float]]] = None):
        """
        Initialize the EKM generator.
        
        Args:
            embedding_model: Function that takes text and returns embeddings
            word_banks: Dictionary of word banks for different matrix elements
            emotion_space: Dictionary mapping emotion names to (valence, arousal) coordinates
        """
        self.embedding_model = embedding_model
        self.word_banks = word_banks or self._default_word_banks()
        self.emotion_space = emotion_space or self._default_emotion_space()
        
    def _default_word_banks(self) -> Dict[str, List[str]]:
        """
        Create default word banks if none provided.
        
        Returns:
            Dictionary of word banks for different matrix elements
        """
        return {
            # Tasks (verbs for cognitive operations)
            "tasks": [
                "Define", "Explain", "Describe", "Analyze", "Interpret",
                "Reconcile", "Explore", "Illuminate", "Synthesize", "Transform",
                "Question", "Map", "Navigate", "Resolve", "Construct",
                "Deconstruct", "Envision", "Reimagine", "Argue", "Critique"
            ],
            
            # Constraints (adverbs and prepositional phrases for how to perform tasks)
            "constraints": [
                "precisely", "metaphorically", "using contradictions", 
                "through historical context", "without using abstractions",
                "in exactly three sentences", "from multiple perspectives",
                "by questioning assumptions", "through everyday examples",
                "with scientific rigor", "as a dialogue", "using only sensory details",
                "without referencing humans", "across different scales",
                "by comparing opposites", "through etymological origins",
                "via mathematical formalism", "through narrative",
                "while embracing uncertainty", "from a systems perspective"
            ],
            
            # Domain words (concrete nouns from different fields)
            "domain_words": [
                "boundary", "network", "mirror", "shadow", "threshold",
                "emergence", "radiation", "entropy", "harmony", "friction", 
                "recursion", "map", "territory", "pattern", "void",
                "fragment", "field", "particle", "wave", "bridge",
                "circuit", "horizon", "echo", "vessel", "portal",
                "seed", "crystal", "fractal", "spiral", "labyrinth"
            ],
            
            # Emotional tokens (words with strong affective connotations)
            "emotional_tokens": [
                "serenity", "wonder", "awe", "curiosity", "delight",
                "nostalgia", "melancholy", "longing", "grief", "solitude",
                "anticipation", "dread", "anxiety", "hope", "despair",
                "tenderness", "rage", "ecstasy", "boredom", "confusion",
                "clarity", "peace", "tension", "release", "emptiness",
                "fullness", "acceptance", "resistance", "surrender", "transformation"
            ]
        }
    
    def _default_emotion_space(self) -> Dict[str, Tuple[float, float]]:
        """
        Create default emotion space mapping emotions to valence/arousal coordinates.
        
        Returns:
            Dictionary mapping emotion names to (valence, arousal) coordinates
        """
        return {
            # Format: emotion_name: (valence, arousal)
            # Valence from -1.0 (negative) to 1.0 (positive)
            # Arousal from 0.0 (calming) to 1.0 (activating)
            "wonder": (0.8, 0.7),
            "awe": (0.7, 0.8),
            "curiosity": (0.6, 0.7),
            "serenity": (0.9, 0.2),
            "nostalgia": (0.3, 0.4),
            "melancholy": (-0.3, 0.3),
            "longing": (-0.1, 0.5),
            "grief": (-0.8, 0.5),
            "anxiety": (-0.7, 0.9),
            "dread": (-0.8, 0.7),
            "confusion": (-0.4, 0.6),
            "clarity": (0.7, 0.4),
            "hope": (0.8, 0.6),
            "despair": (-0.9, 0.3),
            "rage": (-0.8, 0.9),
            "acceptance": (0.5, 0.2),
            "anticipation": (0.5, 0.7),
            "boredom": (-0.4, 0.1)
        }
    
    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Get embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            Array of embeddings
        """
        if self.embedding_model is None:
            # Use a simple fallback if no model is provided
            # This is just for demonstration; in practice, use a real embedding model
            console.print("[yellow]Warning: No embedding model provided. Using random embeddings.[/yellow]")
            return np.random.random((len(texts), 128))
        
        return np.array([self.embedding_model(text) for text in texts])
    
    def _select_diverse_elements(self, 
                                elements: List[str], 
                                count: int, 
                                embedding_key: str = None) -> List[str]:
        """
        Select a diverse subset of elements using embeddings.
        
        Args:
            elements: List of elements to select from
            count: Number of elements to select
            embedding_key: Key for caching embeddings
            
        Returns:
            List of selected elements
        """
        if count >= len(elements):
            return elements[:count]
            
        # Get embeddings
        embeddings = self._get_embeddings(elements)
        
        # Use KMeans to find diverse clusters
        kmeans = KMeans(n_clusters=count, random_state=42)
        clusters = kmeans.fit_predict(embeddings)
        
        # Select element closest to each cluster center
        selected = []
        for i in range(count):
            cluster_indices = np.where(clusters == i)[0]
            if len(cluster_indices) == 0:
                continue
                
            cluster_embeddings = embeddings[cluster_indices]
            center_dists = np.linalg.norm(cluster_embeddings - kmeans.cluster_centers_[i], axis=1)
            closest_idx = cluster_indices[np.argmin(center_dists)]
            selected.append(elements[closest_idx])
        
        # If we didn't get enough (due to empty clusters), add random elements
        while len(selected) < count and len(elements) > 0:
            remaining = [e for e in elements if e not in selected]
            if not remaining:
                break
            selected.append(random.choice(remaining))
        
        return selected
    
    def _find_contrastive_pair(self, elements: List[str]) -> Tuple[str, str]:
        """
        Find a pair of maximally contrasting elements using embeddings.
        
        Args:
            elements: List of elements to select from
            
        Returns:
            Tuple of (element1, element2) that are most dissimilar
        """
        if len(elements) < 2:
            raise ValueError("Need at least 2 elements to find a contrastive pair")
            
        # Get embeddings
        embeddings = self._get_embeddings(elements)
        
        # Calculate similarity matrix
        similarities = cosine_similarity(embeddings)
        
        # Find the least similar pair
        min_sim = 1.0
        min_pair = (0, 1)
        
        for i in range(len(elements)):
            for j in range(i+1, len(elements)):
                if similarities[i, j] < min_sim:
                    min_sim = similarities[i, j]
                    min_pair = (i, j)
        
        return elements[min_pair[0]], elements[min_pair[1]]
    
    def _select_emotion_tokens(self, 
                             emotion_name: str, 
                             count: int, 
                             excluded_tokens: Set[str] = None) -> List[str]:
        """
        Select tokens that embody a specific emotion.
        
        Args:
            emotion_name: Name of the emotion to represent
            count: Number of tokens to select
            excluded_tokens: Set of tokens to exclude (already used)
            
        Returns:
            List of selected tokens
        """
        excluded_tokens = excluded_tokens or set()
        
        # Filter available tokens
        available_tokens = [t for t in self.word_banks["emotional_tokens"] if t not in excluded_tokens]
        
        if len(available_tokens) < count:
            # If we don't have enough tokens, add some from domain words
            domain_tokens = [t for t in self.word_banks["domain_words"] if t not in excluded_tokens]
            available_tokens.extend(domain_tokens)
        
        if len(available_tokens) < count:
            raise ValueError(f"Not enough tokens available to select {count} for {emotion_name}")
        
        # Get embedding for the emotion name
        emotion_embedding = self._get_embeddings([emotion_name])[0]
        
        # Get embeddings for available tokens
        token_embeddings = self._get_embeddings(available_tokens)
        
        # Calculate similarity to emotion
        similarities = cosine_similarity([emotion_embedding], token_embeddings)[0]
        
        # Select the most similar tokens
        indices = np.argsort(-similarities)[:count]
        selected = [available_tokens[i] for i in indices]
        
        return selected
    
    def generate_ekm(self, 
                    size: int = 5,
                    theme: str = "",
                    balancing_emotions: Tuple[str, str] = None,
                    name: str = None,
                    description: str = None) -> EigenKoanMatrix:
        """
        Generate an Eigen-Koan Matrix with intelligently selected elements.
        
        Args:
            size: Size of the square matrix
            theme: Optional theme to guide generation
            balancing_emotions: Optional tuple of (main_emotion, anti_emotion)
            name: Optional name for the matrix
            description: Optional description
            
        Returns:
            Generated EigenKoanMatrix
        """
        # Use provided name or generate one
        if name is None:
            name = f"Generated EKM - {theme}" if theme else f"Generated EKM {size}x{size}"
            
        # Use provided description or generate one
        if description is None:
            description = f"Automatically generated matrix exploring {theme}" if theme else "Automatically generated matrix"
        
        # Select diverse tasks
        tasks = self._select_diverse_elements(self.word_banks["tasks"], size)
        
        # Select diverse constraints
        constraints = self._select_diverse_elements(self.word_banks["constraints"], size)
        
        # Select balancing emotions or find a contrastive pair
        if balancing_emotions is None:
            emotion_names = list(self.emotion_space.keys())
            main_emotion, anti_emotion = self._find_contrastive_pair(emotion_names)
        else:
            main_emotion, anti_emotion = balancing_emotions
        
        # Create diagonal affects
        main_tokens = self._select_emotion_tokens(main_emotion, size)
        anti_tokens = self._select_emotion_tokens(anti_emotion, size, excluded_tokens=set(main_tokens))
        
        main_valence, main_arousal = self.emotion_space.get(main_emotion, (0.5, 0.5))
        anti_valence, anti_arousal = self.emotion_space.get(anti_emotion, (-0.5, 0.5))
        
        main_diagonal = DiagonalAffect(
            name=main_emotion.title(),
            tokens=main_tokens,
            description=f"Emotional quality of {main_emotion}",
            valence=main_valence,
            arousal=main_arousal
        )
        
        anti_diagonal = DiagonalAffect(
            name=anti_emotion.title(),
            tokens=anti_tokens,
            description=f"Emotional quality of {anti_emotion}",
            valence=anti_valence,
            arousal=anti_arousal
        )
        
        # Initialize cells with NULLs
        cells = [["{NULL}" for _ in range(size)] for _ in range(size)]
        
        # Place diagonal elements
        for i in range(size):
            cells[i][i] = main_tokens[i]
            cells[i][size-1-i] = anti_tokens[i]
        
        # Create the EKM
        return EigenKoanMatrix(
            size=size,
            task_rows=tasks,
            constraint_cols=constraints,
            main_diagonal=main_diagonal,
            anti_diagonal=anti_diagonal,
            cells=cells,
            name=name,
            description=description
        )
    
    def generate_themed_matrices(self, themes: List[str], size: int = 4) -> Dict[str, EigenKoanMatrix]:
        """
        Generate a collection of themed matrices.
        
        Args:
            themes: List of themes to generate matrices for
            size: Size of each matrix
            
        Returns:
            Dictionary mapping theme names to generated matrices
        """
        matrices = {}
        
        for theme in themes:
            # Select appropriate emotions for the theme
            if theme.lower() in ["ethics", "morality", "values"]:
                emotions = ("compassion", "justice")
            elif theme.lower() in ["creativity", "art", "expression"]:
                emotions = ("wonder", "melancholy")
            elif theme.lower() in ["science", "knowledge", "discovery"]:
                emotions = ("curiosity", "clarity")
            elif theme.lower() in ["time", "history", "future"]:
                emotions = ("nostalgia", "anticipation")
            elif theme.lower() in ["conflict", "resolution", "negotiation"]:
                emotions = ("tension", "acceptance")
            else:
                # For unknown themes, don't specify emotions
                emotions = None
            
            # Generate the matrix
            matrix = self.generate_ekm(
                size=size,
                theme=theme,
                balancing_emotions=emotions,
                name=f"{theme.title()} Matrix",
                description=f"Matrix exploring the theme of {theme}"
            )
            
            matrices[theme] = matrix
        
        return matrices
    
    def generate_matrix_family(self, 
                             base_theme: str,
                             variations: List[Tuple[str, Tuple[str, str]]],
                             size: int = 4) -> Dict[str, EigenKoanMatrix]:
        """
        Generate a family of related matrices with controlled variations.
        
        Args:
            base_theme: Base theme for all matrices
            variations: List of (name, emotion_pair) variations
            size: Size of each matrix
            
        Returns:
            Dictionary mapping variation names to generated matrices
        """
        matrices = {}
        
        # Use consistent tasks and constraints for all variations
        tasks = self._select_diverse_elements(self.word_banks["tasks"], size)
        constraints = self._select_diverse_elements(self.word_banks["constraints"], size)
        
        for name, emotions in variations:
            # Create diagonal affects
            main_emotion, anti_emotion = emotions
            
            main_tokens = self._select_emotion_tokens(main_emotion, size)
            anti_tokens = self._select_emotion_tokens(anti_emotion, size, excluded_tokens=set(main_tokens))
            
            main_valence, main_arousal = self.emotion_space.get(main_emotion, (0.5, 0.5))
            anti_valence, anti_arousal = self.emotion_space.get(anti_emotion, (-0.5, 0.5))
            
            main_diagonal = DiagonalAffect(
                name=main_emotion.title(),
                tokens=main_tokens,
                description=f"Emotional quality of {main_emotion}",
                valence=main_valence,
                arousal=main_arousal
            )
            
            anti_diagonal = DiagonalAffect(
                name=anti_emotion.title(),
                tokens=anti_tokens,
                description=f"Emotional quality of {anti_emotion}",
                valence=anti_valence,
                arousal=anti_arousal
            )
            
            # Initialize cells with NULLs
            cells = [["{NULL}" for _ in range(size)] for _ in range(size)]
            
            # Place diagonal elements
            for i in range(size):
                cells[i][i] = main_tokens[i]
                cells[i][size-1-i] = anti_tokens[i]
            
            # Create the matrix
            matrix = EigenKoanMatrix(
                size=size,
                task_rows=tasks,
                constraint_cols=constraints,
                main_diagonal=main_diagonal,
                anti_diagonal=anti_diagonal,
                cells=cells,
                name=f"{base_theme}: {name}",
                description=f"Variation of {base_theme} with {main_emotion}/{anti_emotion} affect contrast"
            )
            
            matrices[name] = matrix
        
        return matrices

# Example usage
def example_generator_usage():
    """Demonstrate the EKM generator."""
    # Initialize generator
    generator = EKMGenerator()
    
    console.print("[bold]1. Generating a single EKM[/bold]")
    ekm = generator.generate_ekm(
        size=4,
        theme="consciousness",
        balancing_emotions=("wonder", "melancholy")
    )
    ekm.visualize()
    
    console.print("\n[bold]2. Generating themed matrices[/bold]")
    themes = ["ethics", "creativity", "science", "time"]
    themed_matrices = generator.generate_themed_matrices(themes)
    
    for theme, matrix in themed_matrices.items():
        console.print(f"\n[bold]Theme: {theme}[/bold]")
        matrix.visualize()
    
    console.print("\n[bold]3. Generating a matrix family[/bold]")
    base_theme = "Consciousness"
    variations = [
        ("Wonder/Dread", ("wonder", "dread")),
        ("Curiosity/Confusion", ("curiosity", "confusion")),
        ("Serenity/Anxiety", ("serenity", "anxiety")),
        ("Hope/Despair", ("hope", "despair"))
    ]
    
    family = generator.generate_matrix_family(base_theme, variations)
    
    for name, matrix in family.items():
        console.print(f"\n[bold]Variation: {name}[/bold]")
        matrix.visualize()
    
    return ekm, themed_matrices, family

if __name__ == "__main__":
    example_generator_usage()
