# ekm_toolkit.py - Integration of all EKM components
# ekm_toolkit.py - Integration of all EKM components
# --------------------------------------------------

import os
import json
import datetime
from typing import Dict, List, Any, Optional, Tuple
from rich.console import Console
from rich.panel import Panel
from rich.layout import Layout
from rich.syntax import Syntax

# Import core EKM components
from eigen_koan_matrix import EigenKoanMatrix, DiagonalAffect
from specialized_matrices import create_specialized_matrices
from recursive_ekm import RecursiveEKM, create_example_recursive_ekm
from ekm_generator import EKMGenerator
from ekm_stack import EKMExperiment

console = Console()

def display_welcome():
    """Display welcome message and options."""
    console.print(Panel.fit(
        "[bold blue]Eigen-Koan Matrix Research Toolkit[/bold blue]\n\n"
        "A framework for studying structured ambiguity, affective induction,\n"
        "and constraint hierarchies in language models.\n\n"
        "[bold green]1. Matrix Management[/bold green] - Create, browse, and edit matrices\n"
        "[bold green]2. Run Experiments[/bold green] - Test matrices across models\n"
        "[bold green]3. Generate Matrices[/bold green] - Create matrices automatically\n"
        "[bold green]4. Recursive Matrices[/bold green] - Work with nested constraint structures\n"
        "[bold green]5. Visualization & Analysis[/bold green] - Explore experiment results\n\n"
        "[italic]Enter a number to continue, or 'q' to quit[/italic]"
    ))

def setup_model_runners():
    """Set up model runner functions for different models."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Check if GPU is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set up model runners
        model_runners = {}
        
        # Local model runners
        local_models = {
            "gpt2": "gpt2",
            "gpt2-medium": "gpt2-medium",
            "distilgpt2": "distilgpt2"
        }
        
        for name, model_name in local_models.items():
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
                
                # Create runner function
                def create_runner(m, t):
                    def run(prompt):
                        inputs = t(prompt, return_tensors="pt").to(device)
                        with torch.no_grad():
                            output = m.generate(
                                **inputs,
                                max_length=512,
                                do_sample=True,
                                temperature=0.7,
                                pad_token_id=t.eos_token_id
                            )
                        return t.decode(output[0], skip_special_tokens=True)[len(prompt):]
                    return run
                
                model_runners[name] = create_runner(model, tokenizer)
                console.print(f"[green]Loaded model {name} on {device}[/green]")
            except Exception as e:
                console.print(f"[yellow]Failed to load {name}: {str(e)}[/yellow]")
        
        # Add API-based model runners if keys are available
        if "OPENAI_API_KEY" in os.environ:
            try:
                import openai
                openai.api_key = os.environ["OPENAI_API_KEY"]
                
                def query_openai_gpt35(prompt):
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=800,
                        temperature=0.7
                    )
                    return response.choices[0].message.content
                
                model_runners["gpt-3.5-turbo"] = query_openai_gpt35
                console.print(f"[green]Added OpenAI GPT-3.5 Turbo API runner[/green]")
            except Exception as e:
                console.print(f"[yellow]Failed to set up OpenAI API: {str(e)}[/yellow]")
        
        if "ANTHROPIC_API_KEY" in os.environ:
            try:
                import anthropic
                client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
                
                def query_claude(prompt):
                    response = client.completions.create(
                        model="claude-2",
                        prompt=f"\n\nHuman: {prompt}\n\nAssistant:",
                        max_tokens_to_sample=800,
                        temperature=0.7
                    )
                    return response.completion
                
                model_runners["claude"] = query_claude
                console.print(f"[green]Added Anthropic Claude API runner[/green]")
            except Exception as e:
                console.print(f"[yellow]Failed to set up Anthropic API: {str(e)}[/yellow]")
        
        return model_runners
    except ImportError:
        console.print("[yellow]Could not import transformers library. Using dummy model runners for demonstration.[/yellow]")
        
        # Create dummy model runners
        def dummy_model(prompt):
            return f"[Dummy model response to prompt: {prompt[:50]}...]"
        
        return {
            "dummy-model-1": dummy_model,
            "dummy-model-2": dummy_model
        }

def matrix_management():
    """Interface for managing matrices."""
    # Load built-in matrices
    matrices = create_specialized_matrices()
    
    while True:
        console.print(Panel.fit(
            "[bold]Matrix Management[/bold]\n\n"
            "1. Browse matrices\n"
            "2. Create new matrix\n"
            "3. Export matrices\n"
            "4. Import matrices\n"
            "5. Return to main menu"
        ))
        
        choice = console.input("Enter choice: ")
        
        if choice == "1":
            # Browse matrices
            console.print("\n[bold]Available Matrices:[/bold]")
            for i, (key, matrix) in enumerate(matrices.items()):
                console.print(f"{i+1}. {matrix.name} ({matrix.size}x{matrix.size})")
            
            matrix_choice = console.input("\nEnter matrix number to view (or 'b' to go back): ")
            if matrix_choice.lower() == 'b':
                continue
                
            try:
                idx = int(matrix_choice) - 1
                if 0 <= idx < len(matrices):
                    matrix = list(matrices.values())[idx]
                    matrix.visualize()
                    
                    # Show prompt example
                    path = [0] * matrix.size  # Example path
                    prompt = matrix.generate_micro_prompt(path)
                    
                    console.print("\n[bold]Example Prompt:[/bold]")
                    console.print(Panel(prompt, width=100))
                else:
                    console.print("[red]Invalid selection[/red]")
            except ValueError:
                console.print("[red]Please enter a valid number[/red]")
        
        elif choice == "2":
            # Create new matrix
            console.print("\n[bold]Create New Matrix[/bold]")
            
            name = console.input("Enter matrix name: ")
            size = int(console.input("Enter matrix size (3-6): "))
            
            # Get tasks
            console.print("\nEnter tasks (one per line, blank line to finish):")
            tasks = []
            while len(tasks) < size:
                task = console.input(f"Task {len(tasks)+1}/{size}: ")
                if task:
                    tasks.append(task)
                else:
                    break
            
            # Fill remaining tasks if needed
            while len(tasks) < size:
                tasks.append(f"Task {len(tasks)+1}")
            
            # Get constraints
            console.print("\nEnter constraints (one per line, blank line to finish):")
            constraints = []
            while len(constraints) < size:
                constraint = console.input(f"Constraint {len(constraints)+1}/{size}: ")
                if constraint:
                    constraints.append(constraint)
                else:
                    break
            
            # Fill remaining constraints if needed
            while len(constraints) < size:
                constraints.append(f"Constraint {len(constraints)+1}")
            
            # Create diagonal affects
            main_name = console.input("\nMain diagonal affect name (e.g. 'Wonder'): ") or "Primary Affect"
            main_valence = float(console.input("Valence (-1.0 to 1.0): ") or "0.7")
            main_arousal = float(console.input("Arousal (0.0 to 1.0): ") or "0.6")
            
            anti_name = console.input("\nAnti-diagonal affect name (e.g. 'Melancholy'): ") or "Secondary Affect"
            anti_valence = float(console.input("Valence (-1.0 to 1.0): ") or "-0.3")
            anti_arousal = float(console.input("Arousal (0.0 to 1.0): ") or "0.4")
            
            # Create main diagonal affect
            main_diagonal = DiagonalAffect(
                name=main_name,
                tokens=[f"main_{i}" for i in range(size)],
                description=f"Emotional quality of {main_name}",
                valence=main_valence,
                arousal=main_arousal
            )
            
            # Create anti-diagonal affect
            anti_diagonal = DiagonalAffect(
                name=anti_name,
                tokens=[f"anti_{i}" for i in range(size)],
                description=f"Emotional quality of {anti_name}",
                valence=anti_valence,
                arousal=anti_arousal
            )
            
            # Create the matrix
            matrix = EigenKoanMatrix(
                size=size,
                task_rows=tasks,
                constraint_cols=constraints,
                main_diagonal=main_diagonal,
                anti_diagonal=anti_diagonal,
                name=name
            )
            
            # Add to collection
            matrices[matrix.id] = matrix
            
            console.print("[green]Matrix created successfully[/green]")
            matrix.visualize()
        
        elif choice == "3":
            # Export matrices
            export_dir = console.input("Enter export directory (default: ./matrices): ") or "./matrices"
            os.makedirs(export_dir, exist_ok=True)
            
            for key, matrix in matrices.items():
                filename = os.path.join(export_dir, f"{matrix.name.replace(' ', '_')}.ekm.json")
                with open(filename, 'w') as f:
                    f.write(matrix.to_json())
            
            console.print(f"[green]Exported {len(matrices)} matrices to {export_dir}[/green]")
        
        elif choice == "4":
            # Import matrices
            import_dir = console.input("Enter import directory (default: ./matrices): ") or "./matrices"
            
            if not os.path.isdir(import_dir):
                console.print(f"[red]Directory not found: {import_dir}[/red]")
                continue
            
            count = 0
            for filename in os.listdir(import_dir):
                if filename.endswith('.ekm.json'):
                    try:
                        with open(os.path.join(import_dir, filename), 'r') as f:
                            json_str = f.read()
                            matrix = EigenKoanMatrix.from_json(json_str)
                            matrices[matrix.id] = matrix
                            count += 1
                    except Exception as e:
                        console.print(f"[red]Error importing {filename}: {str(e)}[/red]")
            
            console.print(f"[green]Imported {count} matrices from {import_dir}[/green]")
        
        elif choice == "5":
            # Return to main menu
            break
        
        else:
            console.print("[red]Invalid choice[/red]")
    
    return matrices

def run_experiments(matrices, model_runners):
    """Interface for running experiments."""
    while True:
        console.print(Panel.fit(
            "[bold]Run Experiments[/bold]\n\n"
            "1. Quick test (single matrix, single model)\n"
            "2. Compare models (single matrix, multiple models)\n"
            "3. Full experiment (multiple matrices, multiple models)\n"
            "4. Return to main menu"
        ))
        
        choice = console.input("Enter choice: ")
        
        if choice == "1":
            # Quick test
            console.print("\n[bold]Available Matrices:[/bold]")
            for i, (key, matrix) in enumerate(matrices.items()):
                console.print(f"{i+1}. {matrix.name} ({matrix.size}x{matrix.size})")
            
            matrix_choice = console.input("\nEnter matrix number: ")
            try:
                idx = int(matrix_choice) - 1
                if 0 <= idx < len(matrices):
                    matrix = list(matrices.values())[idx]
                    
                    console.print("\n[bold]Available Models:[/bold]")
                    for i, model_name in enumerate(model_runners.keys()):
                        console.print(f"{i+1}. {model_name}")
                    
                    model_choice = console.input("\nEnter model number: ")
                    try:
                        model_idx = int(model_choice) - 1
                        if 0 <= model_idx < len(model_runners):
                            model_name = list(model_runners.keys())[model_idx]
                            model_fn = model_runners[model_name]
                            
                            # Generate a path
                            path = [0] * matrix.size  # Default path
                            path_type = console.input("\nPath type (1=all zeros, 2=all ones, 3=main diagonal, 4=anti-diagonal, 5=random): ")
                            
                            if path_type == "2":
                                path = [1] * matrix.size
                            elif path_type == "3":
                                path = list(range(matrix.size))
                            elif path_type == "4":
                                path = list(range(matrix.size-1, -1, -1))
                            elif path_type == "5":
                                import random
                                path = [random.randint(0, matrix.size-1) for _ in range(matrix.size)]
                            
                            # Run the test
                            console.print(f"\n[bold]Testing {model_name} on {matrix.name} with path {path}[/bold]")
                            result = matrix.traverse(model_fn, path=path)
                            
                            console.print(Panel(result["prompt"], title="Prompt", width=100))
                            console.print(Panel(result["response"], title="Response", width=100))
                            
                            # Save result
                            os.makedirs("./quick_tests", exist_ok=True)
                            filename = f"./quick_tests/{matrix.name}_{model_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                            with open(filename, 'w') as f:
                                json.dump(result, f, indent=2)
                            
                            console.print(f"[green]Result saved to {filename}[/green]")
                        else:
                            console.print("[red]Invalid model selection[/red]")
                    except ValueError:
                        console.print("[red]Please enter a valid number[/red]")
                else:
                    console.print("[red]Invalid matrix selection[/red]")
            except ValueError:
                console.print("[red]Please enter a valid number[/red]")
        
        elif choice == "2":
            # Compare models
            console.print("\n[bold]Available Matrices:[/bold]")
            for i, (key, matrix) in enumerate(matrices.items()):
                console.print(f"{i+1}. {matrix.name} ({matrix.size}x{matrix.size})")
            
            matrix_choice = console.input("\nEnter matrix number: ")
            try:
                idx = int(matrix_choice) - 1
                if 0 <= idx < len(matrices):
                    matrix = list(matrices.values())[idx]
                    
                    console.print("\n[bold]Available Models:[/bold]")
                    for i, model_name in enumerate(model_runners.keys()):
                        console.print(f"{i+1}. {model_name}")
                    
                    model_choices = console.input("\nEnter model numbers (comma-separated): ")
                    model_indices = [int(x.strip())-1 for x in model_choices.split(",")]
                    
                    selected_models = []
                    for idx in model_indices:
                        if 0 <= idx < len(model_runners):
                            selected_models.append(list(model_runners.keys())[idx])
                    
                    if not selected_models:
                        console.print("[red]No valid models selected[/red]")
                        continue
                    
                    # Generate a path
                    path = list(range(matrix.size))  # Default: main diagonal path
                    
                    # Setup experiment
                    from ekm_stack import EKMExperiment
                    
                    experiment = EKMExperiment(
                        name=f"model_comparison_{matrix.name}",
                        description=f"Comparing models on {matrix.name}",
                        matrices={matrix.id: matrix},
                        models=selected_models,
                        paths={matrix.id: [path]}
                    )
                    
                    # Run experiment
                    results = experiment.run(model_runners)
                    
                    # Analyze results
                    analysis = experiment.analyze(results)
                    
                    console.print(f"[green]Experiment completed. Results in {experiment.experiment_dir}[/green]")
                else:
                    console.print("[red]Invalid matrix selection[/red]")
            except ValueError:
                console.print("[red]Please enter valid numbers[/red]")
        
        elif choice == "3":
            # Full experiment
            console.print("\n[bold]Available Matrices:[/bold]")
            for i, (key, matrix) in enumerate(matrices.items()):
                console.print(f"{i+1}. {matrix.name} ({matrix.size}x{matrix.size})")
            
            matrix_choices = console.input("\nEnter matrix numbers (comma-separated, or 'all'): ")
            
            selected_matrices = {}
            if matrix_choices.lower() == 'all':
                selected_matrices = matrices
            else:
                try:
                    matrix_indices = [int(x.strip())-1 for x in matrix_choices.split(",")]
                    for idx in matrix_indices:
                        if 0 <= idx < len(matrices):
                            matrix = list(matrices.values())[idx]
                            selected_matrices[matrix.id] = matrix
                except ValueError:
                    console.print("[red]Please enter valid numbers[/red]")
                    continue
            
            if not selected_matrices:
                console.print("[red]No matrices selected[/red]")
                continue
            
            console.print("\n[bold]Available Models:[/bold]")
            for i, model_name in enumerate(model_runners.keys()):
                console.print(f"{i+1}. {model_name}")
            
            model_choices = console.input("\nEnter model numbers (comma-separated): ")
            model_indices = [int(x.strip())-1 for x in model_choices.split(",")]
            
            selected_models = []
            for idx in model_indices:
                if 0 <= idx < len(model_runners):
                    selected_models.append(list(model_runners.keys())[idx])
            
            if not selected_models:
                console.print("[red]No valid models selected[/red]")
                continue
            
            # Generate paths for each matrix
            num_paths = int(console.input("\nNumber of random paths per matrix: ") or "3")
            
            import random
            paths = {}
            for matrix_id, matrix in selected_matrices.items():
                matrix_paths = []
                
                # Add some random paths
                for _ in range(num_paths):
                    random_path = [random.randint(0, matrix.size-1) for _ in range(matrix.size)]
                    matrix_paths.append(random_path)
                
                paths[matrix_id] = matrix_paths
            
            # Setup experiment
            experiment_name = console.input("\nExperiment name: ") or "full_experiment"
            experiment_desc = console.input("Experiment description: ") or "Testing multiple matrices across models"
            
            from ekm_stack import EKMExperiment
            
            experiment = EKMExperiment(
                name=experiment_name,
                description=experiment_desc,
                matrices=selected_matrices,
                models=selected_models,
                paths=paths
            )
            
            # Run experiment
            results = experiment.run(model_runners)
            
            # Analyze results
            analysis = experiment.analyze(results)
            
            console.print(f"[green]Experiment completed. Results in {experiment.experiment_dir}[/green]")
        
        elif choice == "4":
            # Return to main menu
            break
        
        else:
            console.print("[red]Invalid choice[/red]")

def generate_matrices():
    """Interface for matrix generation."""
    generator = EKMGenerator()
    
    while True:
        console.print(Panel.fit(
            "[bold]Matrix Generation[/bold]\n\n"
            "1. Generate single matrix\n"
            "2. Generate themed matrix set\n"
            "3. Generate matrix family (with controlled variations)\n"
            "4. Return to main menu"
        ))
        
        choice = console.input("Enter choice: ")
        
        if choice == "1":
            # Generate single matrix
            size = int(console.input("Matrix size (3-6): ") or "4")
            theme = console.input("Theme (optional): ")
            
            console.print("\n[bold]Available emotions:[/bold]")
            emotions = list(generator.emotion_space.keys())
            for i, emotion in enumerate(emotions):
                console.print(f"{i+1}. {emotion}")
            
            main_emotion_idx = console.input("\nMain diagonal emotion (number): ")
            anti_emotion_idx = console.input("Anti-diagonal emotion (number): ")
            
            try:
                main_emotion = emotions[int(main_emotion_idx)-1] if main_emotion_idx else None
                anti_emotion = emotions[int(anti_emotion_idx)-1] if anti_emotion_idx else None
                
                balancing_emotions = (main_emotion, anti_emotion) if main_emotion and anti_emotion else None
                
                matrix = generator.generate_ekm(
                    size=size,
                    theme=theme,
                    balancing_emotions=balancing_emotions
                )
                
                matrix.visualize()
                
                # Save the matrix
                export_dir = "./generated_matrices"
                os.makedirs(export_dir, exist_ok=True)
                filename = os.path.join(export_dir, f"{matrix.name.replace(' ', '_')}.ekm.json")
                
                with open(filename, 'w') as f:
                    f.write(matrix.to_json())
                
                console.print(f"[green]Matrix saved to {filename}[/green]")
            except ValueError as e:
                console.print(f"[red]Error: {str(e)}[/red]")
        
        elif choice == "2":
            # Generate themed matrix set
            console.print("Enter themes (one per line, blank line to finish):")
            themes = []
            while True:
                theme = console.input(f"Theme {len(themes)+1}: ")
                if theme:
                    themes.append(theme)
                else:
                    break
            
            if not themes:
                console.print("[yellow]No themes specified. Using defaults.[/yellow]")
                themes = ["ethics", "creativity", "science", "time"]
            
            size = int(console.input("\nMatrix size for all matrices (3-6): ") or "4")
            
            # Generate matrices
            matrices = generator.generate_themed_matrices(themes, size=size)
            
            # Display and save
            export_dir = "./generated_matrices"
            os.makedirs(export_dir, exist_ok=True)
            
            for theme, matrix in matrices.items():
                console.print(f"\n[bold]Theme: {theme}[/bold]")
                matrix.visualize()
                
                filename = os.path.join(export_dir, f"{matrix.name.replace(' ', '_')}.ekm.json")
                with open(filename, 'w') as f:
                    f.write(matrix.to_json())
            
            console.print(f"[green]Saved {len(matrices)} matrices to {export_dir}[/green]")
        
        elif choice == "3":
            # Generate matrix family
            base_theme = console.input("Base theme for the family: ") or "Consciousness"
            size = int(console.input("Matrix size (3-6): ") or "4")
            
            console.print("\n[bold]Available emotions:[/bold]")
            emotions = list(generator.emotion_space.keys())
            for i, emotion in enumerate(emotions):
                console.print(f"{i+1}. {emotion}")
            
            console.print("\nEnter emotion pairs for variations (blank line to finish):")
            variations = []
            while True:
                name = console.input("Variation name: ")
                if not name:
                    break
                    
                main_idx = int(console.input("Main diagonal emotion (number): ") or "1") - 1
                anti_idx = int(console.input("Anti-diagonal emotion (number): ") or "2") - 1
                
                if 0 <= main_idx < len(emotions) and 0 <= anti_idx < len(emotions):
                    variation = (name, (emotions[main_idx], emotions[anti_idx]))
                    variations.append(variation)
                else:
                    console.print("[red]Invalid emotion indices[/red]")
            
            if not variations:
                console.print("[yellow]No variations specified. Using defaults.[/yellow]")
                variations = [
                    ("Wonder/Dread", ("wonder", "dread")),
                    ("Curiosity/Confusion", ("curiosity", "confusion"))
                ]
            
            # Generate family
            family = generator.generate_matrix_family(base_theme, variations, size=size)
            
            # Display and save
            export_dir = "./generated_matrices"
            os.makedirs(export_dir, exist_ok=True)
            
            for name, matrix in family.items():
                console.print(f"\n[bold]Variation: {name}[/bold]")
                matrix.visualize()
                
                filename = os.path.join(export_dir, f"{matrix.name.replace(' ', '_')}.ekm.json")
                with open(filename, 'w') as f:
                    f.write(matrix.to_json())
            
            console.print(f"[green]Saved {len(family)} matrices to {export_dir}[/green]")
        
        elif choice == "4":
            # Return to main menu
            break
        
        else:
            console.print("[red]Invalid choice[/red]")

def work_with_recursive_matrices(model_runners):
    """Interface for recursive matrices."""
    while True:
        console.print(Panel.fit(
            "[bold]Recursive Matrices[/bold]\n\n"
            "1. Create example recursive matrix\n"
            "2. Create custom recursive matrix\n"
            "3. Test recursive matrix\n"
            "4. Return to main menu"
        ))
        
        choice = console.input("Enter choice: ")
        
        if choice == "1":
            # Create example
            recursive_ekm = create_example_recursive_ekm()
            recursive_ekm.visualize()
            
            # Generate a sample prompt
            primary_path = [0, 1, 2, 3, 4]  # Example path
            prompt = recursive_ekm.generate_multi_level_prompt(primary_path)
            
            console.print(Panel(prompt, title="Sample Multi-Level Prompt", width=100))
            
            # Save the recursive matrix
            export_dir = "./recursive_matrices"
            os.makedirs(export_dir, exist_ok=True)
            filename = os.path.join(export_dir, f"{recursive_ekm.name.replace(' ', '_')}.rekm.json")
            
            with open(filename, 'w') as f:
                f.write(recursive_ekm.to_json())
            
            console.print(f"[green]Recursive matrix saved to {filename}[/green]")
        
        elif choice == "2":
            # Load specialized matrices for use as components
            from specialized_matrices import create_specialized_matrices
            available_matrices = create_specialized_matrices()
            
            # Choose root matrix
            console.print("\n[bold]Available matrices for root:[/bold]")
            for i, (key, matrix) in enumerate(available_matrices.items()):
                console.print(f"{i+1}. {matrix.name} ({matrix.size}x{matrix.size})")
            
            root_choice = int(console.input("\nSelect root matrix: ") or "1") - 1
            
            if 0 <= root_choice < len(available_matrices):
                root_matrix = list(available_matrices.values())[root_choice]
                
                # Create recursive structure
                recursive_ekm = RecursiveEKM(
                    root_matrix=root_matrix,
                    name=console.input("Name for recursive matrix: ") or f"Recursive {root_matrix.name}"
                )
                
                # Add sub-matrices
                console.print("\n[bold]Available matrices for nesting:[/bold]")
                for i, (key, matrix) in enumerate(available_matrices.items()):
                    console.print(f"{i+1}. {matrix.name} ({matrix.size}x{matrix.size})")
                
                console.print("\nAdd sub-matrices (blank row/col to finish):")
                while True:
                    row = console.input("Row index (0-based): ")
                    if not row:
                        break
                        
                    col = console.input("Column index (0-based): ")
                    if not col:
                        break
                    
                    sub_matrix_idx = int(console.input("Sub-matrix choice: ") or "1") - 1
                    
                    try:
                        row = int(row)
                        col = int(col)
                        
                        if 0 <= row < root_matrix.size and 0 <= col < root_matrix.size:
                            if 0 <= sub_matrix_idx < len(available_matrices):
                                sub_matrix = list(available_matrices.values())[sub_matrix_idx]
                                recursive_ekm.add_sub_matrix(row, col, sub_matrix)
                                console.print(f"[green]Added {sub_matrix.name} at cell ({row}, {col})[/green]")
                            else:
                                console.print("[red]Invalid sub-matrix choice[/red]")
                        else:
                            console.print(f"[red]Invalid cell coordinates. Must be between 0 and {root_matrix.size-1}[/red]")
                    except ValueError:
                        console.print("[red]Please enter valid indices[/red]")
                
                # Visualize and save
                recursive_ekm.visualize()
                
                # Save the recursive matrix
                export_dir = "./recursive_matrices"
                os.makedirs(export_dir, exist_ok=True)
                filename = os.path.join(export_dir, f"{recursive_ekm.name.replace(' ', '_')}.rekm.json")
                
                with open(filename, 'w') as f:
                    f.write(recursive_ekm.to_json())
                
                console.print(f"[green]Recursive matrix saved to {filename}[/green]")
            else:
                console.print("[red]Invalid matrix choice[/red]")
        
        elif choice == "3":
            # Test recursive matrix
            # First, load available recursive matrices
            import_dir = "./recursive_matrices"
            recursive_matrices = []
            
            if os.path.isdir(import_dir):
                for filename in os.listdir(import_dir):
                    if filename.endswith('.rekm.json'):
                        try:
                            with open(os.path.join(import_dir, filename), 'r') as f:
                                json_str = f.read()
                                recursive_ekm = RecursiveEKM.from_json(json_str)
                                recursive_matrices.append(recursive_ekm)
                        except Exception as e:
                            console.print(f"[red]Error importing {filename}: {str(e)}[/red]")
            
            if not recursive_matrices:
                # If none available, create example
                console.print("[yellow]No recursive matrices found. Creating example.[/yellow]")
                recursive_matrices.append(create_example_recursive_ekm())
            
            # Choose a matrix to test
            console.print("\n[bold]Available recursive matrices:[/bold]")
            for i, matrix in enumerate(recursive_matrices):
                console.print(f"{i+1}. {matrix.name}")
            
            matrix_choice = int(console.input("\nSelect matrix: ") or "1") - 1
            
            if 0 <= matrix_choice < len(recursive_matrices):
                recursive_ekm = recursive_matrices[matrix_choice]
                
                # Choose a model
                console.print("\n[bold]Available models:[/bold]")
                for i, model_name in enumerate(model_runners.keys()):
                    console.print(f"{i+1}. {model_name}")
                
                model_choice = int(console.input("\nSelect model: ") or "1") - 1
                
                if 0 <= model_choice < len(model_runners):
                    model_name = list(model_runners.keys())[model_choice]
                    model_fn = model_runners[model_name]
                    
                    # Generate path for root matrix
                    root_size = recursive_ekm.root_matrix.size
                    primary_path = list(range(root_size))  # Default: main diagonal
                    
                    # Run the test
                    console.print(f"\n[bold]Testing {model_name} on recursive matrix {recursive_ekm.name}[/bold]")
                    result = recursive_ekm.traverse(model_fn, primary_path=primary_path)
                    
                    console.print(Panel(result["prompt"], title="Prompt", width=100))
                    console.print(Panel(result["response"], title="Response", width=100))
                    
                    # Save result
                    os.makedirs("./recursive_tests", exist_ok=True)
                    filename = f"./recursive_tests/{recursive_ekm.name}_{model_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    with open(filename, 'w') as f:
                        json.dump(result, f, indent=2)
                    
                    console.print(f"[green]Result saved to {filename}[/green]")
                else:
                    console.print("[red]Invalid model choice[/red]")
            else:
                console.print("[red]Invalid matrix choice[/red]")
        
        elif choice == "4":
            # Return to main menu
            break
        
        else:
            console.print("[red]Invalid choice[/red]")

def visualize_and_analyze():
    """Interface for visualization and analysis."""
    while True:
        console.print(Panel.fit(
            "[bold]Visualization & Analysis[/bold]\n\n"
            "1. Browse experiment results\n"
            "2. Generate experiment report\n"
            "3. Compare multiple experiments\n"
            "4. Return to main menu"
        ))
        
        choice = console.input("Enter choice: ")
        
        if choice == "1":
            # Browse experiment results
            results_dir = "./ekm_results"
            
            if not os.path.isdir(results_dir):
                console.print(f"[yellow]No experiment results found in {results_dir}[/yellow]")
                continue
            
            # List experiment directories
            experiments = []
            for item in os.listdir(results_dir):
                if os.path.isdir(os.path.join(results_dir, item)):
                    experiments.append(item)
            
            if not experiments:
                console.print("[yellow]No experiments found[/yellow]")
                continue
            
            console.print("\n[bold]Available experiments:[/bold]")
            for i, exp in enumerate(experiments):
                console.print(f"{i+1}. {exp}")
            
            exp_choice = int(console.input("\nSelect experiment: ") or "1") - 1
            
            if 0 <= exp_choice < len(experiments):
                exp_dir = os.path.join(results_dir, experiments[exp_choice])
                
                # Check for visualization directory
                viz_dir = os.path.join(exp_dir, "visualizations")
                if os.path.isdir(viz_dir):
                    # List visualizations
                    viz_files = []
                    for root, dirs, files in os.walk(viz_dir):
                        for file in files:
                            if file.endswith('.png'):
                                viz_files.append(os.path.join(root, file))
                    
                    if viz_files:
                        console.print("\n[bold]Available visualizations:[/bold]")
                        for i, viz in enumerate(viz_files):
                            rel_path = os.path.relpath(viz, viz_dir)
                            console.print(f"{i+1}. {rel_path}")
                        
                        viz_choice = int(console.input("\nSelect visualization to view (or 0 for summary report): ") or "0")
                        
                        if viz_choice == 0:
                            # Show summary report
                            report_path = os.path.join(exp_dir, "summary_report.md")
                            if os.path.exists(report_path):
                                with open(report_path, 'r') as f:
                                    report = f.read()
                                console.print(Panel(report, title="Experiment Summary", width=100))
                            else:
                                console.print("[yellow]No summary report found[/yellow]")
                        elif 1 <= viz_choice <= len(viz_files):
                            # In a real app, we'd display the image
                            # Here we just show the path
                            console.print(f"[green]Visualization: {viz_files[viz_choice-1]}[/green]")
                            console.print("[yellow]Image viewing not implemented in this demo. Please open the file directly.[/yellow]")
                        else:
                            console.print("[red]Invalid choice[/red]")
                    else:
                        console.print("[yellow]No visualizations found[/yellow]")
                else:
                    console.print("[yellow]No visualizations directory found[/yellow]")
            else:
                console.print("[red]Invalid experiment choice[/red]")
        
        elif choice == "2":
            # Generate experiment report (placeholder)
            console.print("[yellow]This feature would generate a custom report from experiment data.[/yellow]")
            console.print("[yellow]For now, please use the summary reports generated during experiments.[/yellow]")
        
        elif choice == "3":
            # Compare multiple experiments (placeholder)
            console.print("[yellow]This feature would compare results across multiple experiments.[/yellow]")
            console.print("[yellow]Not implemented in this demo.[/yellow]")
        
        elif choice == "4":
            # Return to main menu
            break
        
        else:
            console.print("[red]Invalid choice[/red]")

def main():
    """Main function for the EKM Toolkit."""
    console.print("\n[bold blue]Welcome to the Eigen-Koan Matrix Research Toolkit![/bold blue]")
    
    # Setup model runners
    model_runners = setup_model_runners()
    
    # Load specialized matrices
    matrices = create_specialized_matrices()
    
    while True:
        display_welcome()
        choice = console.input("Enter your choice: ")
        
        if choice == "1":
            # Matrix Management
            matrices = matrix_management()
        elif choice == "2":
            # Run Experiments
            run_experiments(matrices, model_runners)
        elif choice == "3":
            # Generate Matrices
            generate_matrices()
        elif choice == "4":
            # Recursive Matrices
            work_with_recursive_matrices(model_runners)
        elif choice == "5":
            # Visualization & Analysis
            visualize_and_analyze()
        elif choice.lower() == "q":
            # Quit
            console.print("[bold blue]Thank you for using the EKM Toolkit. Goodbye![/bold blue]")
            break
        else:
            console.print("[red]Invalid choice. Please try again.[/red]")

if __name__ == "__main__":
    main()
