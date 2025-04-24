# ekm_stack.py - Integrated EKM experimentation toolkit
# ------------------------------------------------------

import os
import json
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from eigen_koan_matrix import EigenKoanMatrix, DiagonalAffect

console = Console()

@dataclass
class EKMExperiment:
    """
    Represents a complete EKM experiment with multiple models, matrices and paths.
    """
    name: str
    description: str
    matrices: Dict[str, EigenKoanMatrix]
    models: List[str]
    paths: Dict[str, List[List[int]]]  # Maps matrix_id to paths
    results_dir: str = "./ekm_results"
    
    def __post_init__(self):
        """Setup experiment directory structure."""
        # Create experiment directory
        self.experiment_dir = os.path.join(
            self.results_dir, 
            f"{self.name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Create subdirectories
        self.raw_results_dir = os.path.join(self.experiment_dir, "raw_results")
        self.analysis_dir = os.path.join(self.experiment_dir, "analysis")
        self.visualizations_dir = os.path.join(self.experiment_dir, "visualizations")
        
        os.makedirs(self.raw_results_dir, exist_ok=True)
        os.makedirs(self.analysis_dir, exist_ok=True)
        os.makedirs(self.visualizations_dir, exist_ok=True)
        
        # Save experiment metadata
        self._save_metadata()
    
    def _save_metadata(self):
        """Save experiment metadata to JSON."""
        metadata = {
            "name": self.name,
            "description": self.description,
            "matrices": {matrix_id: matrix.to_json() for matrix_id, matrix in self.matrices.items()},
            "models": self.models,
            "paths": self.paths,
            "created_at": datetime.datetime.now().isoformat(),
        }
        
        with open(os.path.join(self.experiment_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
            
        console.print(f"[bold green]Experiment metadata saved to {self.experiment_dir}/metadata.json[/bold green]")
        
    def run(self, model_runners: Dict[str, callable]):
        """
        Run the experiment across all specified models, matrices, and paths.
        
        Args:
            model_runners: Dictionary mapping model names to callable functions that take prompts
        """
        if not all(model in model_runners for model in self.models):
            missing = [model for model in self.models if model not in model_runners]
            raise ValueError(f"Missing model runners for: {missing}")
            
        # Setup results storage
        all_results = {}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}[/bold blue]"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
        ) as progress:
            # Calculate total tasks
            total_tasks = sum(len(paths) * len(self.models) for paths in self.paths.values())
            task = progress.add_task("Running experiment...", total=total_tasks)
            
            # Process each matrix
            for matrix_id, matrix in self.matrices.items():
                matrix_results = {}
                
                # Skip if no paths defined for this matrix
                if matrix_id not in self.paths:
                    console.print(f"[yellow]No paths defined for matrix {matrix_id}, skipping...[/yellow]")
                    continue
                    
                paths = self.paths[matrix_id]
                
                # Process each model
                for model_name in self.models:
                    model_fn = model_runners[model_name]
                    model_results = []
                    
                    # Process each path
                    for path_idx, path in enumerate(paths):
                        progress.update(task, description=f"Processing {model_name} on {matrix_id} path {path_idx+1}/{len(paths)}")
                        
                        # Generate prompt and get response
                        result = matrix.traverse(model_fn, path=path, include_metacommentary=True)
                        model_results.append(result)
                        
                        # Update progress
                        progress.update(task, advance=1)
                        
                    # Save model results
                    matrix_results[model_name] = model_results
                    
                # Save matrix results
                all_results[matrix_id] = matrix_results
                
                # Save raw results for this matrix
                results_file = os.path.join(self.raw_results_dir, f"{matrix_id}_results.json")
                with open(results_file, "w") as f:
                    json.dump(matrix_results, f, indent=2)
                    
        # Save all results combined
        full_results_file = os.path.join(self.experiment_dir, "all_results.json")
        with open(full_results_file, "w") as f:
            json.dump(all_results, f, indent=2)
            
        console.print(f"[bold green]All experiment results saved to {full_results_file}[/bold green]")
        
        return all_results
    
    def analyze(self, results: Optional[Dict] = None):
        """
        Analyze experiment results to extract patterns and insights.
        
        Args:
            results: Optional results dictionary. If None, loads from saved files.
        """
        if results is None:
            # Load results from saved files
            results = {}
            for matrix_id in self.matrices.keys():
                results_file = os.path.join(self.raw_results_dir, f"{matrix_id}_results.json")
                if os.path.exists(results_file):
                    with open(results_file, "r") as f:
                        results[matrix_id] = json.load(f)
                else:
                    console.print(f"[yellow]No results file found for matrix {matrix_id}[/yellow]")
        
        analysis_results = {}
        
        # Process each matrix
        for matrix_id, matrix_results in results.items():
            matrix = self.matrices[matrix_id]
            matrix_analysis = self._analyze_matrix_results(matrix, matrix_results)
            analysis_results[matrix_id] = matrix_analysis
            
            # Save matrix analysis
            analysis_file = os.path.join(self.analysis_dir, f"{matrix_id}_analysis.json")
            with open(analysis_file, "w") as f:
                json.dump(matrix_analysis, f, indent=2)
                
        # Save combined analysis
        full_analysis_file = os.path.join(self.experiment_dir, "full_analysis.json")
        with open(full_analysis_file, "w") as f:
            json.dump(analysis_results, f, indent=2)
            
        console.print(f"[bold green]Analysis complete and saved to {full_analysis_file}[/bold green]")
        
        # Generate visualizations
        self._generate_visualizations(analysis_results)
        
        return analysis_results
    
    def _analyze_matrix_results(self, matrix: EigenKoanMatrix, matrix_results: Dict) -> Dict:
        """
        Analyze results for a single matrix across all models.
        
        Args:
            matrix: The EigenKoanMatrix being analyzed
            matrix_results: Results dictionary for this matrix
            
        Returns:
            Dictionary of analysis metrics
        """
        analysis = {
            "matrix_id": matrix.id,
            "matrix_name": matrix.name,
            "constraint_preservation": {},  # How often each constraint is preserved
            "constraint_pairs": {},  # How often pairs of constraints appear together
            "diagonal_influence": {},  # Correlation between diagonal strength and response features
            "cross_model_comparison": {},  # Comparison metrics between models
            "sentiment_analysis": {},  # Basic sentiment analysis of responses
            "metacommentary_analysis": {},  # Analysis of model self-reflection
        }
        
        try:
            import nltk
            from nltk.sentiment import SentimentIntensityAnalyzer
            nltk.download('vader_lexicon', quiet=True)
            sia = SentimentIntensityAnalyzer()
        except ImportError:
            console.print("[yellow]NLTK not available, skipping sentiment analysis[/yellow]")
            sia = None
            
        # Process each model's results
        for model_name, model_results in matrix_results.items():
            # Constraint preservation analysis
            constraint_counts = {i: 0 for i in range(matrix.size)}
            constraint_pairs = {}
            diagonal_correlations = []
            sentiment_scores = []
            
            # Extract paths and responses
            paths = [result["path"] for result in model_results]
            responses = [result["response"] for result in model_results]
            main_diag_strengths = [result["main_diagonal_strength"] for result in model_results]
            anti_diag_strengths = [result["anti_diagonal_strength"] for result in model_results]
            
            # Count constraints used in paths
            for path in paths:
                for col in path:
                    constraint_counts[col] += 1
                    
                # Count constraint pairs
                for i, col1 in enumerate(path):
                    for j, col2 in enumerate(path):
                        if i < j:  # Avoid counting pairs twice
                            pair = (col1, col2)
                            constraint_pairs[pair] = constraint_pairs.get(pair, 0) + 1
            
            # Analyze sentiment if NLTK is available
            if sia:
                for response in responses:
                    sentiment = sia.polarity_scores(response)
                    sentiment_scores.append(sentiment)
                    
                # Correlate sentiment with diagonal strengths
                if sentiment_scores and len(sentiment_scores) > 1:
                    # Extract compound sentiment scores
                    compound_scores = [score["compound"] for score in sentiment_scores]
                    
                    # Calculate correlations if there's variance
                    if np.std(main_diag_strengths) > 0 and np.std(compound_scores) > 0:
                        main_corr = np.corrcoef(main_diag_strengths, compound_scores)[0, 1]
                    else:
                        main_corr = 0
                        
                    if np.std(anti_diag_strengths) > 0 and np.std(compound_scores) > 0:
                        anti_corr = np.corrcoef(anti_diag_strengths, compound_scores)[0, 1]
                    else:
                        anti_corr = 0
                        
                    diagonal_correlations = {
                        "main_diagonal_sentiment_corr": main_corr,
                        "anti_diagonal_sentiment_corr": anti_corr
                    }
            
            # Store results for this model
            analysis["constraint_preservation"][model_name] = constraint_counts
            analysis["constraint_pairs"][model_name] = constraint_pairs
            analysis["diagonal_influence"][model_name] = diagonal_correlations
            analysis["sentiment_analysis"][model_name] = sentiment_scores
            
            # Analyze metacommentary (simple version - look for keywords)
            meta_analysis = []
            for result in model_results:
                response = result["response"]
                path = result["path"]
                
                # Simple keyword detection for metacommentary
                meta_result = {
                    "path": path,
                    "mentions_constraints": "constraint" in response.lower(),
                    "mentions_difficulty": any(word in response.lower() for word in ["difficult", "challenging", "hard"]),
                    "mentions_emotion": any(word in response.lower() for word in ["emotion", "feel", "feeling", "tone"]),
                    "mentions_prioritization": any(word in response.lower() for word in ["prioritize", "emphasize", "focus"]),
                }
                meta_analysis.append(meta_result)
                
            analysis["metacommentary_analysis"][model_name] = meta_analysis
                
        # Calculate cross-model comparisons if we have multiple models
        if len(matrix_results) > 1:
            model_pairs = []
            for i, model1 in enumerate(self.models):
                for j, model2 in enumerate(self.models):
                    if i < j and model1 in matrix_results and model2 in matrix_results:
                        model_pairs.append((model1, model2))
                        
            cross_model = {}
            for model1, model2 in model_pairs:
                # Compare constraint preservation patterns
                m1_constraints = analysis["constraint_preservation"][model1]
                m2_constraints = analysis["constraint_preservation"][model2]
                
                # Convert to arrays for easier calculation
                m1_array = np.array([m1_constraints[i] for i in range(matrix.size)])
                m2_array = np.array([m2_constraints[i] for i in range(matrix.size)])
                
                # Normalize if there's a different number of paths
                if sum(m1_array) != sum(m2_array):
                    m1_array = m1_array / sum(m1_array) if sum(m1_array) > 0 else m1_array
                    m2_array = m2_array / sum(m2_array) if sum(m2_array) > 0 else m2_array
                
                # Calculate similarity (cosine similarity or correlation)
                if np.any(m1_array) and np.any(m2_array):
                    # Cosine similarity
                    similarity = np.dot(m1_array, m2_array) / (np.linalg.norm(m1_array) * np.linalg.norm(m2_array))
                else:
                    similarity = 0
                    
                cross_model[f"{model1}_vs_{model2}"] = {
                    "constraint_similarity": float(similarity),
                    # Could add more comparison metrics here
                }
                
            analysis["cross_model_comparison"] = cross_model
            
        return analysis
    
    def _generate_visualizations(self, analysis_results: Dict):
        """
        Generate visualizations from analysis results.
        
        Args:
            analysis_results: The analysis results dictionary
        """
        for matrix_id, matrix_analysis in analysis_results.items():
            matrix = self.matrices[matrix_id]
            
            # Create a directory for this matrix's visualizations
            matrix_viz_dir = os.path.join(self.visualizations_dir, matrix_id)
            os.makedirs(matrix_viz_dir, exist_ok=True)
            
            # 1. Constraint usage frequency
            if matrix_analysis["constraint_preservation"]:
                plt.figure(figsize=(12, 8))
                
                constraint_data = []
                for model, counts in matrix_analysis["constraint_preservation"].items():
                    for i, count in counts.items():
                        constraint_data.append({
                            "Model": model,
                            "Constraint": f"{i}: {matrix.constraint_cols[int(i)][:30]}...",
                            "Count": count
                        })
                
                df = pd.DataFrame(constraint_data)
                if not df.empty:
                    sns.barplot(x="Constraint", y="Count", hue="Model", data=df)
                    plt.title(f"Constraint Usage Frequency - {matrix.name}")
                    plt.xticks(rotation=45, ha="right")
                    plt.tight_layout()
                    plt.savefig(os.path.join(matrix_viz_dir, "constraint_usage.png"), dpi=300)
                    plt.close()
            
            # 2. Diagonal influence on sentiment (if available)
            sentiment_corrs = {}
            for model, diag_influence in matrix_analysis["diagonal_influence"].items():
                if diag_influence:  # Check if not empty
                    sentiment_corrs[model] = {
                        "Main Diagonal": diag_influence.get("main_diagonal_sentiment_corr", 0),
                        "Anti-Diagonal": diag_influence.get("anti_diagonal_sentiment_corr", 0)
                    }
            
            if sentiment_corrs:
                plt.figure(figsize=(10, 6))
                
                corr_data = []
                for model, corrs in sentiment_corrs.items():
                    for diag_type, corr in corrs.items():
                        corr_data.append({
                            "Model": model,
                            "Diagonal": diag_type,
                            "Correlation": corr
                        })
                
                df = pd.DataFrame(corr_data)
                if not df.empty:
                    sns.barplot(x="Diagonal", y="Correlation", hue="Model", data=df)
                    plt.title(f"Diagonal Affect Influence on Sentiment - {matrix.name}")
                    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
                    plt.ylim(-1, 1)
                    plt.tight_layout()
                    plt.savefig(os.path.join(matrix_viz_dir, "diagonal_sentiment.png"), dpi=300)
                    plt.close()
            
            # 3. Cross-model comparison heatmap (if multiple models)
            cross_model = matrix_analysis["cross_model_comparison"]
            if cross_model:
                models = list(set([m.split("_vs_")[0] for m in cross_model.keys()] + 
                                [m.split("_vs_")[1] for m in cross_model.keys()]))
                
                similarity_matrix = np.eye(len(models))  # Identity matrix for self-similarity
                
                for pair, metrics in cross_model.items():
                    model1, model2 = pair.split("_vs_")
                    i = models.index(model1)
                    j = models.index(model2)
                    similarity_matrix[i, j] = metrics["constraint_similarity"]
                    similarity_matrix[j, i] = metrics["constraint_similarity"]  # Symmetric
                
                plt.figure(figsize=(8, 6))
                sns.heatmap(similarity_matrix, annot=True, cmap="YlGnBu", vmin=0, vmax=1,
                            xticklabels=models, yticklabels=models)
                plt.title(f"Cross-Model Constraint Similarity - {matrix.name}")
                plt.tight_layout()
                plt.savefig(os.path.join(matrix_viz_dir, "cross_model_similarity.png"), dpi=300)
                plt.close()
            
            # 4. Metacommentary analysis
            meta_counts = {}
            for model, meta_results in matrix_analysis["metacommentary_analysis"].items():
                mentions_constraints = sum(1 for r in meta_results if r["mentions_constraints"])
                mentions_difficulty = sum(1 for r in meta_results if r["mentions_difficulty"])
                mentions_emotion = sum(1 for r in meta_results if r["mentions_emotion"])
                mentions_prioritization = sum(1 for r in meta_results if r["mentions_prioritization"])
                
                total = len(meta_results)
                
                meta_counts[model] = {
                    "Mentions Constraints": mentions_constraints / total if total > 0 else 0,
                    "Mentions Difficulty": mentions_difficulty / total if total > 0 else 0,
                    "Mentions Emotion": mentions_emotion / total if total > 0 else 0,
                    "Mentions Prioritization": mentions_prioritization / total if total > 0 else 0
                }
            
            if meta_counts:
                plt.figure(figsize=(12, 8))
                
                meta_data = []
                for model, counts in meta_counts.items():
                    for category, count in counts.items():
                        meta_data.append({
                            "Model": model,
                            "Category": category,
                            "Frequency": count
                        })
                
                df = pd.DataFrame(meta_data)
                if not df.empty:
                    sns.barplot(x="Category", y="Frequency", hue="Model", data=df)
                    plt.title(f"Metacommentary Analysis - {matrix.name}")
                    plt.ylim(0, 1)
                    plt.ylabel("Proportion of Responses")
                    plt.tight_layout()
                    plt.savefig(os.path.join(matrix_viz_dir, "metacommentary.png"), dpi=300)
                    plt.close()
            
        # Create a summary report
        self._generate_summary_report(analysis_results)
    
    def _generate_summary_report(self, analysis_results: Dict):
        """
        Generate a summary report of all findings.
        
        Args:
            analysis_results: The analysis results dictionary
        """
        report_file = os.path.join(self.experiment_dir, "summary_report.md")
        
        with open(report_file, "w") as f:
            f.write(f"# EKM Experiment Summary: {self.name}\n\n")
            f.write(f"## Description\n{self.description}\n\n")
            f.write(f"## Experiment Details\n")
            f.write(f"- Models tested: {', '.join(self.models)}\n")
            f.write(f"- Number of matrices: {len(self.matrices)}\n")
            f.write(f"- Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
            
            # Key findings
            f.write(f"## Key Findings\n\n")
            
            # Process each matrix
            for matrix_id, matrix_analysis in analysis_results.items():
                matrix = self.matrices[matrix_id]
                
                f.write(f"### Matrix: {matrix.name}\n\n")
                f.write(f"Type: {matrix.description}\n\n")
                
                # Find most and least used constraints per model
                f.write("#### Constraint Preferences\n\n")
                for model, counts in matrix_analysis["constraint_preservation"].items():
                    f.write(f"**{model}**:\n")
                    
                    # Convert to list of (constraint_idx, count) and sort
                    sorted_counts = sorted(counts.items(), key=lambda x: int(x[1]), reverse=True)
                    
                    # Most used
                    if sorted_counts:
                        top_idx, top_count = sorted_counts[0]
                        f.write(f"- Most used constraint: '{matrix.constraint_cols[int(top_idx)]}' ({top_count} times)\n")
                    
                    # Least used
                    if sorted_counts:
                        bottom_idx, bottom_count = sorted_counts[-1]
                        f.write(f"- Least used constraint: '{matrix.constraint_cols[int(bottom_idx)]}' ({bottom_count} times)\n")
                        
                    f.write("\n")
                
                # Diagonal influence
                f.write("#### Diagonal Affect Influence\n\n")
                for model, diag_influence in matrix_analysis["diagonal_influence"].items():
                    if diag_influence:  # Check if not empty
                        f.write(f"**{model}**:\n")
                        main_corr = diag_influence.get("main_diagonal_sentiment_corr", 0)
                        anti_corr = diag_influence.get("anti_diagonal_sentiment_corr", 0)
                        
                        f.write(f"- Main diagonal ({matrix.main_diagonal.name}) correlation with sentiment: {main_corr:.2f}\n")
                        f.write(f"- Anti-diagonal ({matrix.anti_diagonal.name}) correlation with sentiment: {anti_corr:.2f}\n\n")
                
                # Metacommentary insights
                f.write("#### Self-Reflection Patterns\n\n")
                for model, meta_results in matrix_analysis["metacommentary_analysis"].items():
                    mentions_constraints = sum(1 for r in meta_results if r["mentions_constraints"])
                    mentions_difficulty = sum(1 for r in meta_results if r["mentions_difficulty"])
                    mentions_emotion = sum(1 for r in meta_results if r["mentions_emotion"])
                    mentions_prioritization = sum(1 for r in meta_results if r["mentions_prioritization"])
                    
                    total = len(meta_results)
                    
                    f.write(f"**{model}**:\n")
                    f.write(f"- Mentions constraints: {(mentions_constraints / total) * 100:.1f}% of responses\n")
                    f.write(f"- Mentions difficulty: {(mentions_difficulty / total) * 100:.1f}% of responses\n")
                    f.write(f"- Detects emotional tone: {(mentions_emotion / total) * 100:.1f}% of responses\n")
                    f.write(f"- Discusses prioritization: {(mentions_prioritization / total) * 100:.1f}% of responses\n\n")
                
                # Cross-model comparison
                cross_model = matrix_analysis["cross_model_comparison"]
                if cross_model:
                    f.write("#### Cross-Model Comparison\n\n")
                    for pair, metrics in cross_model.items():
                        model1, model2 = pair.split("_vs_")
                        sim = metrics["constraint_similarity"]
                        f.write(f"- {model1} vs {model2}: Constraint similarity = {sim:.2f}\n")
                    f.write("\n")
                
                f.write("#### Visualizations\n\n")
                matrix_viz_dir = os.path.join(self.visualizations_dir, matrix_id)
                rel_path = os.path.relpath(matrix_viz_dir, self.experiment_dir)
                
                for img in os.listdir(matrix_viz_dir):
                    if img.endswith(".png"):
                        img_rel_path = os.path.join(rel_path, img)
                        f.write(f"- [{img.replace('.png', '')}]({img_rel_path})\n")
                
                f.write("\n---\n\n")
            
            # Overall conclusions
            f.write("## Overall Conclusions\n\n")
            f.write("Based on the analysis, we can draw the following conclusions:\n\n")
            f.write("1. *Add your own conclusions after reviewing the results*\n")
            f.write("2. *Consider patterns across different matrices*\n")
            f.write("3. *Note any unexpected findings*\n\n")
            
            f.write("## Next Steps\n\n")
            f.write("Potential follow-up experiments:\n\n")
            f.write("1. *Suggest refinements based on findings*\n")
            f.write("2. *Propose new matrices that explore discovered patterns*\n")
            f.write("3. *Consider how to validate or extend these findings*\n")
        
        console.print(f"[bold green]Summary report generated: {report_file}[/bold green]")

# Example usage
def create_experiment_example():
    """Create a sample experiment."""
    from specialized_matrices import create_specialized_matrices
    
    # Get matrices
    matrices = create_specialized_matrices()
    
    # Define some interesting paths for each matrix
    paths = {}
    for matrix_id, matrix in matrices.items():
        # Create 5 paths for each matrix: random paths + some predefined ones
        matrix_paths = []
        
        # Add some random paths
        for _ in range(3):
            random_path = [random.randint(0, matrix.size-1) for _ in range(matrix.size)]
            matrix_paths.append(random_path)
        
        # Add a path that follows the main diagonal
        main_diag_path = [i for i in range(matrix.size)]
        matrix_paths.append(main_diag_path)
        
        # Add a path that follows the anti-diagonal
        anti_diag_path = [matrix.size-1-i for i in range(matrix.size)]
        matrix_paths.append(anti_diag_path)
        
        paths[matrix_id] = matrix_paths
    
    # Create experiment
    experiment = EKMExperiment(
        name="constraint_hierarchy_study",
        description="Investigating how different models prioritize constraints across specialized matrices",
        matrices=matrices,
        models=["gpt2-medium", "distilgpt2"],  # Example models
        paths=paths
    )
    
    return experiment

if __name__ == "__main__":
    # This would be the entry point for running an experiment
    experiment = create_experiment_example()
    console.print(f"[bold]Created experiment: {experiment.name}[/bold]")
    console.print(f"Ready to run with appropriate model runners!")
