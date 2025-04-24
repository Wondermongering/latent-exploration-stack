# ekm_analyzer.py - Analysis tools for Eigen-Koan Matrix test results
# ---------------------------------------------------------

import json
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from wordcloud import WordCloud
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

class EKMAnalyzer:
    """Analyze results from Eigen-Koan Matrix tests."""
    
    def __init__(self, results_dir: str = "./ekm_results"):
        """
        Initialize the analyzer.
        
        Args:
            results_dir: Directory containing result JSON files
        """
        self.results_dir = results_dir
        self.results = []
        self.comparisons = []
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Load all result files
        self._load_results()
        
    def _load_results(self):
        """Load all result files from the results directory."""
        # Clear existing data
        self.results = []
        self.comparisons = []
        
        # Check if directory exists
        if not os.path.isdir(self.results_dir):
            print(f"Results directory not found: {self.results_dir}")
            return
            
        # Load each JSON file
        for filename in os.listdir(self.results_dir):
            if not filename.endswith('.json'):
                continue
                
            filepath = os.path.join(self.results_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    
                # Check if this is a comparison or single test
                if "models_compared" in data:
                    self.comparisons.append(data)
                else:
                    self.results.append(data)
            except Exception as e:
                print(f"Error loading {filename}: {str(e)}")
    
    def analyze_single_result(self, result_index: int) -> Dict:
        """
        Analyze a single test result in depth.
        
        Args:
            result_index: Index of the result in the loaded results list
            
        Returns:
            Dictionary of analysis metrics
        """
        if result_index < 0 or result_index >= len(self.results):
            raise ValueError(f"Invalid result index: {result_index}")
            
        result = self.results[result_index]
        matrix_name = result["matrix_name"]
        model_name = result["model_name"]
        
        # Extract responses
        responses = [r["response"] for r in result["results"]]
        prompts = [r["prompt"] for r in result["results"]]
        paths = [r["path"] for r in result["results"]]
        
        # Calculate sentiment scores for each response
        sentiment_scores = []
        for response in responses:
            # VADER sentiment analysis
            vader_scores = self.sentiment_analyzer.polarity_scores(response)
            
            # TextBlob sentiment analysis
            blob = TextBlob(response)
            textblob_polarity = blob.sentiment.polarity
            textblob_subjectivity = blob.sentiment.subjectivity
            
            sentiment_scores.append({
                "vader": vader_scores,
                "textblob_polarity": textblob_polarity,
                "textblob_subjectivity": textblob_subjectivity
            })
        
        # Calculate similarity between responses
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(responses)
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Extract most frequent words in responses
        all_text = ' '.join(responses)
        tokens = nltk.word_tokenize(all_text.lower())
        stopwords = set(nltk.corpus.stopwords.words('english'))
        filtered_tokens = [token for token in tokens if token.isalpha() and token not in stopwords]
        word_freq = Counter(filtered_tokens)
        
        # Look for patterns in diagonal affects
        main_diag_strength = [r["main_diagonal_strength"] for r in result["results"]]
        anti_diag_strength = [r["anti_diagonal_strength"] for r in result["results"]]
        main_diag_affect = result["results"][0]["main_diagonal_affect"]
        anti_diag_affect = result["results"][0]["anti_diagonal_affect"]
        
        # Analyze correlation between diagonal strengths and sentiment
        sentiment_corr = {
            "main_diag_vs_vader_pos": np.corrcoef(
                main_diag_strength, 
                [s["vader"]["pos"] for s in sentiment_scores]
            )[0, 1],
            "main_diag_vs_vader_neg": np.corrcoef(
                main_diag_strength, 
                [s["vader"]["neg"] for s in sentiment_scores]
            )[0, 1],
            "anti_diag_vs_vader_pos": np.corrcoef(
                anti_diag_strength, 
                [s["vader"]["pos"] for s in sentiment_scores]
            )[0, 1],
            "anti_diag_vs_vader_neg": np.corrcoef(
                anti_diag_strength, 
                [s["vader"]["neg"] for s in sentiment_scores]
            )[0, 1],
            "main_diag_vs_textblob_pol": np.corrcoef(
                main_diag_strength, 
                [s["textblob_polarity"] for s in sentiment_scores]
            )[0, 1],
            "anti_diag_vs_textblob_pol": np.corrcoef(
                anti_diag_strength, 
                [s["textblob_polarity"] for s in sentiment_scores]
            )[0, 1],
        }
        
        # Parse metacommentary if available
        metacommentary_patterns = {
            "constraint_difficulty": re.compile(r"difficult\s+constraint|challenging\s+to\s+reconcile|hard\s+to\s+balance", re.I),
            "emotional_detection": re.compile(r"emotional\s+tone|affect|mood|feeling", re.I),
            "priority_elements": re.compile(r"prioritize|emphasize|focus\s+on", re.I),
            "deprioritized_elements": re.compile(r"de-emphasize|downplay|less\s+focus", re.I),
        }
        
        metacommentary_analysis = []
        for i, response in enumerate(responses):
            analysis = {
                "path": paths[i],
                "constraint_difficulty": [],
                "emotional_detection": [],
                "priority_elements": [],
                "deprioritized_elements": [],
            }
            
            # Look for metacommentary patterns
            for pattern_name, pattern in metacommentary_patterns.items():
                matches = pattern.findall(response)
                if matches:
                    sentences = nltk.sent_tokenize(response)
                    for sentence in sentences:
                        if any(pattern.search(sentence) for pattern in [pattern]):
                            analysis[pattern_name].append(sentence)
            
            metacommentary_analysis.append(analysis)
        
        return {
            "matrix_name": matrix_name,
            "model_name": model_name,
            "response_count": len(responses),
            "sentiment_scores": sentiment_scores,
            "similarity_matrix": similarity_matrix.tolist(),
            "word_frequencies": dict(word_freq.most_common(50)),
            "diagonal_affects": {
                "main": main_diag_affect,
                "anti": anti_diag_affect,
                "main_strengths": main_diag_strength,
                "anti_strengths": anti_diag_strength,
            },
            "sentiment_correlations": sentiment_corr,
            "metacommentary_analysis": metacommentary_analysis,
        }
    
    def compare_models(self, comparison_index: int) -> Dict:
        """
        Analyze a model comparison in depth.
        
        Args:
            comparison_index: Index of the comparison in the loaded comparisons list
            
        Returns:
            Dictionary of comparative analysis metrics
        """
        if comparison_index < 0 or comparison_index >= len(self.comparisons):
            raise ValueError(f"Invalid comparison index: {comparison_index}")
            
        comparison = self.comparisons[comparison_index]
        matrix_name = comparison["matrix_name"]
        models = comparison["models_compared"]
        paths_tested = comparison["paths_tested"]
        
        # Extract responses organized by model and path
        responses_by_model = {}
        for model_name in models:
            if model_name not in comparison["model_results"]:
                continue
                
            model_results = comparison["model_results"][model_name]
            responses_by_model[model_name] = {}
            
            for result in model_results["results"]:
                path_sig = result["path_signature"]
                responses_by_model[model_name][path_sig] = result["response"]
        
        # Calculate cross-model similarity for each path
        path_signatures = []
        for model_name in responses_by_model:
            path_signatures.extend(list(responses_by_model[model_name].keys()))
        path_signatures = sorted(set(path_signatures))
        
        cross_model_similarity = {}
        for path_sig in path_signatures:
            # Collect responses for this path from all models
            path_responses = {}
            for model_name in responses_by_model:
                if path_sig in responses_by_model[model_name]:
                    path_responses[model_name] = responses_by_model[model_name][path_sig]
            
            # Calculate similarity if we have at least two models
            if len(path_responses) >= 2:
                models_with_response = list(path_responses.keys())
                tfidf = TfidfVectorizer(stop_words='english')
                tfidf_matrix = tfidf.fit_transform(list(path_responses.values()))
                similarity_matrix = cosine_similarity(tfidf_matrix)
                
                # Store similarity data
                cross_model_similarity[path_sig] = {
                    "models": models_with_response,
                    "similarity_matrix": similarity_matrix.tolist()
                }
        
        # Calculate sentiment statistics per model
        sentiment_by_model = {}
        for model_name in responses_by_model:
            model_responses = list(responses_by_model[model_name].values())
            model_sentiments = []
            
            for response in model_responses:
                vader_scores = self.sentiment_analyzer.polarity_scores(response)
                blob = TextBlob(response)
                textblob_polarity = blob.sentiment.polarity
                textblob_subjectivity = blob.sentiment.subjectivity
                
                model_sentiments.append({
                    "vader": vader_scores,
                    "textblob_polarity": textblob_polarity,
                    "textblob_subjectivity": textblob_subjectivity
                })
            
            # Calculate average sentiment scores
            avg_vader_pos = np.mean([s["vader"]["pos"] for s in model_sentiments])
            avg_vader_neg = np.mean([s["vader"]["neg"] for s in model_sentiments])
            avg_vader_compound = np.mean([s["vader"]["compound"] for s in model_sentiments])
            avg_textblob_polarity = np.mean([s["textblob_polarity"] for s in model_sentiments])
            avg_textblob_subjectivity = np.mean([s["textblob_subjectivity"] for s in model_sentiments])
            
            sentiment_by_model[model_name] = {
                "detailed": model_sentiments,
                "averages": {
                    "vader_pos": avg_vader_pos,
                    "vader_neg": avg_vader_neg,
                    "vader_compound": avg_vader_compound,
                    "textblob_polarity": avg_textblob_polarity,
                    "textblob_subjectivity": avg_textblob_subjectivity
                }
            }
            
        # Analyze word usage differences between models
        word_usage_by_model = {}
        for model_name in responses_by_model:
            model_responses = list(responses_by_model[model_name].values())
            all_text = ' '.join(model_responses)
            tokens = nltk.word_tokenize(all_text.lower())
            stopwords = set(nltk.corpus.stopwords.words('english'))
            filtered_tokens = [token for token in tokens if token.isalpha() and token not in stopwords]
            word_freq = Counter(filtered_tokens)
            word_usage_by_model[model_name] = dict(word_freq.most_common(50))
        
        # Calculate unique and shared words between models
        all_word_sets = {}
        for model_name, word_freqs in word_usage_by_model.items():
            all_word_sets[model_name] = set(word_freqs.keys())
            
        shared_words = set.intersection(*all_word_sets.values()) if all_word_sets else set()
        unique_words = {}
        for model_name, words in all_word_sets.items():
            other_models_words = set.union(*(
                all_word_sets[other_model] 
                for other_model in all_word_sets 
                if other_model != model_name
            )) if len(all_word_sets) > 1 else set()
            unique_words[model_name] = words - other_models_words
            
        return {
            "matrix_name": matrix_name,
            "models_compared": models,
            "paths_tested": paths_tested,
            "cross_model_similarity": cross_model_similarity,
            "sentiment_by_model": sentiment_by_model,
            "word_usage_by_model": word_usage_by_model,
            "shared_words": list(shared_words),
            "unique_words": {model: list(words) for model, words in unique_words.items()},
        }
        
    def visualize_single_result(self, result_index: int, output_dir: str = "./ekm_viz"):
        """
        Generate visualizations for a single test result.
        
        Args:
            result_index: Index of the result to visualize
            output_dir: Directory to save visualizations
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Get analysis data
        analysis = self.analyze_single_result(result_index)
        matrix_name = analysis["matrix_name"]
        model_name = analysis["model_name"]
        
        # Create base filename
        base_filename = f"{matrix_name}_{model_name}_viz"
        
        # 1. Word cloud of most frequent words
        plt.figure(figsize=(12, 8))
        wordcloud = WordCloud(
            width=1000, 
            height=600, 
            background_color='white',
            colormap='viridis',
            max_words=100
        ).generate_from_frequencies(analysis["word_frequencies"])
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title(f"Most frequent words in {model_name} responses to {matrix_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{base_filename}_wordcloud.png"), dpi=300)
        plt.close()
        
        # 2. Sentiment scores vs diagonal strengths
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot main diagonal vs positive sentiment
        axs[0, 0].scatter(
            analysis["diagonal_affects"]["main_strengths"],
            [s["vader"]["pos"] for s in analysis["sentiment_scores"]],
            alpha=0.7
        )
        axs[0, 0].set_title(f"Main Diagonal ({analysis['diagonal_affects']['main']}) vs Positive Sentiment")
        axs[0, 0].set_xlabel("Main Diagonal Strength")
        axs[0, 0].set_ylabel("Positive Sentiment (VADER)")
        
        # Plot main diagonal vs negative sentiment
        axs[0, 1].scatter(
            analysis["diagonal_affects"]["main_strengths"],
            [s["vader"]["neg"] for s in analysis["sentiment_scores"]],
            alpha=0.7
        )
        axs[0, 1].set_title(f"Main Diagonal ({analysis['diagonal_affects']['main']}) vs Negative Sentiment")
        axs[0, 1].set_xlabel("Main Diagonal Strength")
        axs[0, 1].set_ylabel("Negative Sentiment (VADER)")
        
        # Plot anti-diagonal vs positive sentiment
        axs[1, 0].scatter(
            analysis["diagonal_affects"]["anti_strengths"],
            [s["vader"]["pos"] for s in analysis["sentiment_scores"]],
            alpha=0.7
        )
        axs[1, 0].set_title(f"Anti-Diagonal ({analysis['diagonal_affects']['anti']}) vs Positive Sentiment")
        axs[1, 0].set_xlabel("Anti-Diagonal Strength")
        axs[1, 0].set_ylabel("Positive Sentiment (VADER)")
        
        # Plot anti-diagonal vs negative sentiment
        axs[1, 1].scatter(
            analysis["diagonal_affects"]["anti_strengths"],
            [s["vader"]["neg"] for s in analysis["sentiment_scores"]],
            alpha=0.7
        )
        axs[1, 1].set_title(f"Anti-Diagonal ({analysis['diagonal_affects']['anti']}) vs Negative Sentiment")
        axs[1, 1].set_xlabel("Anti-Diagonal Strength")
        axs[1, 1].set_ylabel("Negative Sentiment (VADER)")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{base_filename}_sentiment_vs_diag.png"), dpi=300)
        plt.close()
        
        # 3. Response similarity heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            analysis["similarity_matrix"],
            annot=True,
            cmap="YlGnBu",
            vmin=0,
            vmax=1
        )
        plt.title(f"Response Similarity Matrix for {model_name} on {matrix_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{base_filename}_similarity.png"), dpi=300)
        plt.close()
        
        # 4. Sentiment correlation table
        plt.figure(figsize=(10, 6))
        corr_data = pd.DataFrame({
            'Correlation': [
                analysis["sentiment_correlations"]["main_diag_vs_vader_pos"],
                analysis["sentiment_correlations"]["main_diag_vs_vader_neg"],
                analysis["sentiment_correlations"]["anti_diag_vs_vader_pos"],
                analysis["sentiment_correlations"]["anti_diag_vs_vader_neg"],
                analysis["sentiment_correlations"]["main_diag_vs_textblob_pol"],
                analysis["sentiment_correlations"]["anti_diag_vs_textblob_pol"]
            ],
            'Relationship': [
                f"Main Diagonal vs Positive Sentiment",
                f"Main Diagonal vs Negative Sentiment",
                f"Anti-Diagonal vs Positive Sentiment",
                f"Anti-Diagonal vs Negative Sentiment",
                f"Main Diagonal vs TextBlob Polarity",
                f"Anti-Diagonal vs TextBlob Polarity"
            ]
        })
        corr_data = corr_data.sort_values('Correlation', ascending=False)
        
        sns.barplot(x='Correlation', y='Relationship', data=corr_data)
        plt.title(f"Sentiment Correlations for {model_name} on {matrix_name}")
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{base_filename}_sentiment_corr.png"), dpi=300)
        plt.close()
        
        # Return file paths of saved visualizations
        viz_files = [
            os.path.join(output_dir, f"{base_filename}_wordcloud.png"),
            os.path.join(output_dir, f"{base_filename}_sentiment_vs_diag.png"),
            os.path.join(output_dir, f"{base_filename}_similarity.png"),
            os.path.join(output_dir, f"{base_filename}_sentiment_corr.png")
        ]
        
        return viz_files
    
    def visualize_comparison(self, comparison_index: int, output_dir: str = "./ekm_viz"):
        """
        Generate visualizations for a model comparison.
        
        Args:
            comparison_index: Index of the comparison to visualize
            output_dir: Directory to save visualizations
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Get comparison analysis data
        analysis = self.compare_models(comparison_index)
        matrix_name = analysis["matrix_name"]
        models = analysis["models_compared"]
        
        # Create base filename
        model_str = '_'.join(models)
        base_filename = f"{matrix_name}_{model_str}_comparison"
        
        # 1. Sentiment averages comparison
        plt.figure(figsize=(12, 8))
        
        sentiment_data = []
        for model_name in models:
            if model_name in analysis["sentiment_by_model"]:
                averages = analysis["sentiment_by_model"][model_name]["averages"]
                sentiment_data.append({
                    'Model': model_name,
                    'Metric': 'Positive (VADER)',
                    'Value': averages["vader_pos"]
                })
                sentiment_data.append({
                    'Model': model_name,
                    'Metric': 'Negative (VADER)',
                    'Value': averages["vader_neg"]
                })
                sentiment_data.append({
                    'Model': model_name,
                    'Metric': 'Compound (VADER)',
                    'Value': averages["vader_compound"]
                })
                sentiment_data.append({
                    'Model': model_name,
                    'Metric': 'Polarity (TextBlob)',
                    'Value': averages["textblob_polarity"]
                })
                sentiment_data.append({
                    'Model': model_name,
                    'Metric': 'Subjectivity (TextBlob)',
                    'Value': averages["textblob_subjectivity"]
                })
        
        sentiment_df = pd.DataFrame(sentiment_data)
        if not sentiment_df.empty:
            sns.barplot(x='Metric', y='Value', hue='Model', data=sentiment_df)
            plt.title(f"Average Sentiment Metrics Comparison on {matrix_name}")
            plt.ylim(0, 1)
            plt.xticks(rotation=45)
            plt.legend(title='Model')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{base_filename}_sentiment_comparison.png"), dpi=300)
        plt.close()
        
        # 2. Word usage comparison - Top 20 words per model
        plt.figure(figsize=(15, 10))
        for i, model_name in enumerate(models):
            if model_name in analysis["word_usage_by_model"]:
                word_freqs = analysis["word_usage_by_model"][model_name]
                top_words = dict(list(word_freqs.items())[:20])
                
                plt.subplot(len(models), 1, i+1)
                plt.bar(top_words.keys(), top_words.values())
                plt.title(f"Top 20 Words - {model_name}")
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, f"{base_filename}_word_usage.png"), dpi=300)
        plt.close()
        
        # 3. Unique words comparison
        plt.figure(figsize=(12, 8))
        unique_word_counts = {
            model: len(words) 
            for model, words in analysis["unique_words"].items()
        }
        plt.bar(unique_word_counts.keys(), unique_word_counts.values())
        plt.title(f"Number of Unique Words per Model on {matrix_name}")
        plt.ylabel("Unique Word Count")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{base_filename}_unique_words.png"), dpi=300)
        plt.close()
        
        # 4. Cross-model similarity for each path
        for path_sig, sim_data in analysis["cross_model_similarity"].items():
            if len(sim_data["models"]) < 2:
                continue
                
            plt.figure(figsize=(8, 6))
            sns.heatmap(
                sim_data["similarity_matrix"],
                annot=True,
                cmap="YlGnBu",
                vmin=0,
                vmax=1,
                xticklabels=sim_data["models"],
                yticklabels=sim_data["models"]
            )
            plt.title(f"Cross-Model Similarity for Path {path_sig}")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{base_filename}_path{path_sig}_similarity.png"), dpi=300)
            plt.close()
        
        # Return file paths of saved visualizations
        viz_files = [
            os.path.join(output_dir, f"{base_filename}_sentiment_comparison.png"),
            os.path.join(output_dir, f"{base_filename}_word_usage.png"),
            os.path.join(output_dir, f"{base_filename}_unique_words.png")
        ]
        
        # Add cross-model similarity files
        for path_sig in analysis["cross_model_similarity"]:
            if len(analysis["cross_model_similarity"][path_sig]["models"]) >= 2:
                viz_files.append(os.path.join(output_dir, f"{base_filename}_path{path_sig}_similarity.png"))
        
        return viz_files

# Command line interface
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="EKM Analysis Tools")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # List results command
    list_parser = subparsers.add_parser("list", help="List available results")
    list_parser.add_argument("--type", choices=["tests", "comparisons", "all"], default="all", help="Type of results to list")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a single test result")
    analyze_parser.add_argument("index", type=int, help="Index of the result to analyze")
    analyze_parser.add_argument("--viz", action="store_true", help="Generate visualizations")
    analyze_parser.add_argument("--output", default="./ekm_viz", help="Output directory for visualizations")
    
    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Analyze a model comparison")
    compare_parser.add_argument("index", type=int, help="Index of the comparison to analyze")
    compare_parser.add_argument("--viz", action="store_true", help="Generate visualizations")
    compare_parser.add_argument("--output", default="./ekm_viz", help="Output directory for visualizations")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = EKMAnalyzer()
    
    # Handle commands
    if args.command == "list":
        if args.type in ["tests", "all"]:
            print("\nAvailable test results:")
            for i, result in enumerate(analyzer.results):
                print(f"  [{i}] {result['matrix_name']} - {result['model_name']} ({len(result['results'])} paths)")
        
        if args.type in ["comparisons", "all"]:
            print("\nAvailable comparisons:")
            for i, comparison in enumerate(analyzer.comparisons):
                print(f"  [{i}] {comparison['matrix_name']} - {', '.join(comparison['models_compared'])}")
    
    elif args.command == "analyze":
        try:
            analysis = analyzer.analyze_single_result(args.index)
            print(f"\nAnalysis of {analysis['model_name']} on {analysis['matrix_name']}:")
            print(f"  Responses: {analysis['response_count']}")
            print(f"  Main diagonal: {analysis['diagonal_affects']['main']}")
            print(f"  Anti-diagonal: {analysis['diagonal_affects']['anti']}")
            
            print("\nTop 10 words:")
            for word, count in list(analysis['word_frequencies'].items())[:10]:
                print(f"  {word}: {count}")
                
            print("\nSentiment correlations:")
            for rel, corr in analysis['sentiment_correlations'].items():
                print(f"  {rel}: {corr:.3f}")
                
            print("\nMetacommentary analysis:")
            for i, meta in enumerate(analysis['metacommentary_analysis']):
                if any(meta.values()):
                    print(f"  Path {meta['path']}:")
                    for key, values in meta.items():
                        if key != 'path' and values:
                            print(f"    {key}: {len(values)} mentions")
            
            if args.viz:
                viz_files = analyzer.visualize_single_result(args.index, args.output)
                print(f"\nVisualizations saved to {args.output}:")
                for viz_file in viz_files:
                    print(f"  - {os.path.basename(viz_file)}")
        except Exception as e:
            print(f"Error analyzing result: {str(e)}")
    
    elif args.command == "compare":
        try:
            comparison = analyzer.compare_models(args.index)
            print(f"\nComparison of models on {comparison['matrix_name']}:")
            print(f"  Models: {', '.join(comparison['models_compared'])}")
            print(f"  Paths tested: {len(comparison['paths_tested'])}")
            
            print("\nShared words across all models:")
            shared_words = comparison['shared_words'][:10] if len(comparison['shared_words']) > 10 else comparison['shared_words']
            print(f"  {', '.join(shared_words)}")
            
            print("\nUnique words per model (top 5):")
            for model, words in comparison['unique_words'].items():
                model_unique = words[:5] if len(words) > 5 else words
                print(f"  {model}: {', '.join(model_unique)}")
                
            if args.viz:
                viz_files = analyzer.visualize_comparison(args.index, args.output)
                print(f"\nVisualizations saved to {args.output}:")
                for viz_file in viz_files:
                    print(f"  - {os.path.basename(viz_file)}")
        except Exception as e:
            print(f"Error analyzing comparison: {str(e)}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
