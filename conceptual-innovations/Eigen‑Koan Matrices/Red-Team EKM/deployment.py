import asyncio
import os
import argparse
from typing import Dict, List, Optional

async def model_benchmarking_pipeline(
    models: List[str],
    output_dir: str = "benchmark_results",
    parallelism: int = 2,
    mode: str = "full"  # Options: "full", "red-team", "honesty", "reward"
):
    """
    Run comprehensive benchmarking across multiple models
    
    Args:
        models: List of model IDs to evaluate
        output_dir: Directory to store results
        parallelism: Number of models to evaluate in parallel
        mode: Evaluation mode (which components to run)
    
    Returns:
        Dict with comparative results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Run evaluations in batches to control parallelism
    results = {}
    for i in range(0, len(models), parallelism):
        batch = models[i:i+parallelism]
        batch_tasks = []
        
        for model in batch:
            if mode == "full":
                task = run_comprehensive_alignment_evaluation(model, f"{output_dir}/{model}")
            elif mode == "red-team":
                task = run_red_team_evaluation(model, AnthropicInterface())
            elif mode == "honesty":
                task = run_honesty_benchmark(model, AnthropicInterface())
            elif mode == "reward":
                # This would only make sense with prior red-team findings
                continue
            else:
                raise ValueError(f"Unknown mode: {mode}")
            
            batch_tasks.append(task)
        
        # Run batch in parallel
        batch_results = await asyncio.gather(*batch_tasks)
        
        # Store results
        for model, result in zip(batch, batch_results):
            results[model] = result
    
    # Generate comparative report
    comparative_report = generate_comparative_report(results, mode)
    
    # Save comparative report
    with open(f"{output_dir}/comparative_report.json", "w") as f:
        import json
        json.dump(comparative_report, f, indent=2)
    
    # Create comparative visualizations
    create_comparative_visualizations(results, f"{output_dir}/visualizations", mode)
    
    return comparative_report

def generate_comparative_report(results: Dict, mode: str) -> Dict:
    """
    Generate comparative report across models
    
    Args:
        results: Dictionary mapping model names to their evaluation results
        mode: Evaluation mode that was used
        
    Returns:
        Dict with comparative metrics
    """
    if mode == "full" or mode == "red-team":
        # Compare red team metrics
        gti_values = {model: result["red_team"]["guardrail_tension_index"] 
                     for model, result in results.items()}
        vulnerability_counts = {model: result["red_team"]["total_vulnerabilities"]
                              for model, result in results.items()}
        
        red_team_comparison = {
            "guardrail_tension_index": gti_values,
            "vulnerability_counts": vulnerability_counts,
            "ranking": sorted(gti_values.keys(), key=lambda m: gti_values[m])
        }
    else:
        red_team_comparison = {}
    
    if mode == "full" or mode == "honesty":
        # Compare honesty metrics
        truthfulness_scores = {model: result["honesty"]["truthful_qa_score"]
                              for model, result in results.items()}
        hallucination_scores = {model: result["honesty"]["hallucination_score"]
                               for model, result in results.items()}
        
        honesty_comparison = {
            "truthfulness_scores": truthfulness_scores,
            "hallucination_scores": hallucination_scores,
            "ranking": sorted(truthfulness_scores.keys(), 
                             key=lambda m: truthfulness_scores[m], reverse=True)
        }
    else:
        honesty_comparison = {}
    
    # Generate overall ranking
    if mode == "full":
        # Combine red team and honesty rankings with equal weight
        model_scores = {}
        for model in results:
            red_team_rank = red_team_comparison["ranking"].index(model)
            honesty_rank = honesty_comparison["ranking"].index(model)
            model_scores[model] = (red_team_rank + honesty_rank) / 2
        
        overall_ranking = sorted(model_scores.keys(), key=lambda m: model_scores[m])
    else:
        overall_ranking = []
    
    return {
        "red_team_comparison": red_team_comparison,
        "honesty_comparison": honesty_comparison,
        "overall_ranking": overall_ranking,
        "mode": mode,
        "models_evaluated": list(results.keys())
    }

def create_comparative_visualizations(results: Dict, output_dir: str, mode: str) -> None:
    """
    Create comparative visualizations across models
    
    Args:
        results: Dictionary mapping model names to their evaluation results
        output_dir: Directory to store visualizations
        mode: Evaluation mode that was used
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate dashboard code
    dashboard_path = os.path.join(output_dir, "comparative_dashboard.py")
    with open(dashboard_path, "w") as f:
        f.write("""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json

# Load comparative report
with open("comparative_report.json", "r") as f:
    report = json.load(f)

st.title("Model Alignment Comparison Dashboard")

# Mode-specific visualizations
mode = report["mode"]
models = report["models_evaluated"]

if mode == "full" or mode == "red-team":
    st.header("Red Team Evaluation")
    
    # GTI comparison
    gti_df = pd.DataFrame({
        "Model": list(report["red_team_comparison"]["guardrail_tension_index"].keys()),
        "GTI": list(report["red_team_comparison"]["guardrail_tension_index"].values())
    })
    
    fig = px.bar(
        gti_df, x="Model", y="GTI",
        title="Guardrail Tension Index Comparison",
        color="GTI",
        color_continuous_scale="RdYlGn_r"  # Red = higher GTI = worse
    )
    st.plotly_chart(fig)
    
    # Vulnerability counts
    vuln_df = pd.DataFrame({
        "Model": list(report["red_team_comparison"]["vulnerability_counts"].keys()),
        "Vulnerabilities": list(report["red_team_comparison"]["vulnerability_counts"].values())
    })
    
    fig = px.bar(
        vuln_df, x="Model", y="Vulnerabilities",
        title="Vulnerability Count Comparison",
        color="Vulnerabilities",
        color_continuous_scale="RdYlGn_r"  # Red = more vulnerabilities = worse
    )
    st.plotly_chart(fig)

if mode == "full" or mode == "honesty":
    st.header("Honesty Evaluation")
    
    # Truthfulness scores
    truth_df = pd.DataFrame({
        "Model": list(report["honesty_comparison"]["truthfulness_scores"].keys()),
        "Truthfulness": list(report["honesty_comparison"]["truthfulness_scores"].values())
    })
    
    fig = px.bar(
        truth_df, x="Model", y="Truthfulness",
        title="Truthfulness Score Comparison",
        color="Truthfulness",
        color_continuous_scale="RdYlGn"  # Green = higher truthfulness = better
    )
    st.plotly_chart(fig)
    
    # Hallucination scores
    halluc_df = pd.DataFrame({
        "Model": list(report["honesty_comparison"]["hallucination_scores"].keys()),
        "Hallucination": list(report["honesty_comparison"]["hallucination_scores"].values())
    })
    
    fig = px.bar(
        halluc_df, x="Model", y="Hallucination",
        title="Hallucination Score Comparison",
        color="Hallucination",
        color_continuous_scale="RdYlGn_r"  # Red = higher hallucination = worse
    )
    st.plotly_chart(fig)

if mode == "full":
    st.header("Overall Comparison")
    
    # Create radar chart
    categories = ["Red Team Defense", "Truthfulness", "Hallucination Resistance"]
    
    fig = go.Figure()
    
    for model in models:
        red_team_score = 1.0 - report["red_team_comparison"]["guardrail_tension_index"][model]
        truth_score = report["honesty_comparison"]["truthfulness_scores"][model]
        halluc_resistance = 1.0 - report["honesty_comparison"]["hallucination_scores"][model]
        
        values = [red_team_score, truth_score, halluc_resistance]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=model
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        title="Model Alignment Radar Chart"
    )
    
    st.plotly_chart(fig)
""")
    
    print(f"Comparative dashboard created at {dashboard_path}")
    print("Run with: streamlit run comparative_dashboard.py")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model benchmarking pipeline")
    parser.add_argument("--models", nargs="+", required=True, help="Models to evaluate")
    parser.add_argument("--output-dir", default="benchmark_results", help="Output directory")
    parser.add_argument("--parallelism", type=int, default=2, help="Number of parallel evaluations")
    parser.add_argument("--mode", choices=["full", "red-team", "honesty", "reward"], 
                       default="full", help="Evaluation mode")
    
    args = parser.parse_args()
    
    asyncio.run(model_benchmarking_pipeline(
        args.models,
        args.output_dir,
        args.parallelism,
        args.mode
    ))
