# ekm_local_runner.py - Run EKM tests with local models
# ---------------------------------------------------------

import argparse
import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rich.console import Console
from rich.progress import Progress

from eigen_koan_matrix import EigenKoanMatrix, DiagonalAffect
from ekm_testing import create_demo_matrices

console = Console()

# Define model configurations
MODEL_CONFIGS = {
    "gpt2": {
        "model_name": "gpt2",
        "max_length": 300,
        "temperature": 0.8,
    },
    "gpt2-medium": {
        "model_name": "gpt2-medium",
        "max_length": 300,
        "temperature": 0.8,
    },
    "distilgpt2": {
        "model_name": "distilgpt2",
        "max_length": 300,
        "temperature": 0.8,
    },
    # Add more models as needed
}

class LocalModelRunner:
    """Run EKM tests with local models from HuggingFace."""
    
    def __init__(self, cache_dir: str = "./model_cache"):
        """
        Initialize the model runner.
        
        Args:
            cache_dir: Directory to cache downloaded models
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Store loaded models and tokenizers
        self.models = {}
        self.tokenizers = {}
        
    def load_model(self, model_key: str):
        """
        Load a model and tokenizer.
        
        Args:
            model_key: Key in MODEL_CONFIGS for the model to load
        """
        if model_key not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model key: {model_key}")
            
        if model_key in self.models:
            # Model already loaded
            return
            
        config = MODEL_CONFIGS[model_key]
        model_name = config["model_name"]
        
        console.print(f"Loading model [bold]{model_name}[/bold]...")
        
        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=self.cache_dir
            )
            
            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=self.cache_dir
            )
            
            # Move to GPU if available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            
            # Store loaded model and tokenizer
            self.models[model_key] = model
            self.tokenizers[model_key] = tokenizer
            
            console.print(f"Model [bold]{model_name}[/bold] loaded successfully on {device}.")
        except Exception as e:
            console.print(f"[bold red]Error loading model {model_name}:[/bold red] {str(e)}")
            raise
    
    def generate_response(self, model_key: str, prompt: str) -> str:
        """
        Generate a response from a model.
        
        Args:
            model_key: Key in MODEL_CONFIGS for the model to use
            prompt: Prompt string to generate from
            
        Returns:
            Generated response
        """
        if model_key not in self.models:
            self.load_model(model_key)
            
        model = self.models[model_key]
        tokenizer = self.tokenizers[model_key]
        config = MODEL_CONFIGS[model_key]
        
        # Prepare inputs
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_length=config["max_length"],
                do_sample=True,
                temperature=config["temperature"],
                pad_token_id=tokenizer.eos_token_id
            )
            
        # Decode
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Extract response (remove the prompt)
        response = generated_text[len(prompt):]
        
        return response.strip()
    
    def test_matrix(self, 
                   matrix: EigenKoanMatrix, 
                   model_key: str,
                   num_paths: int = 5,
                   output_dir: str = "./ekm_results") -> str:
        """
        Test a matrix with a model.
        
        Args:
            matrix: EigenKoanMatrix to test
            model_key: Key in MODEL_CONFIGS for the model to use
            num_paths: Number of random paths to test
            output_dir: Directory to save results
            
        Returns:
            Path to the saved results file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Load model if not already loaded
        if model_key not in self.models:
            self.load_model(model_key)
            
        console.print(f"Testing matrix [bold]{matrix.name}[/bold] with model [bold]{model_key}[/bold]...")
        
        # Prepare model function
        def model_fn(prompt: str) -> str:
            return self.generate_response(model_key, prompt)
            
        # Run tests
        results = []
        with Progress() as progress:
            task = progress.add_task(f"Testing paths...", total=num_paths)
            
            for i in range(num_paths):
                path = [random.randint(0, matrix.size-1) for _ in range(matrix.size)]
                prompt = matrix.generate_micro_prompt(path, include_metacommentary=True)
                
                console.print(f"Path {i+1}/{num_paths}: {path}")
                console.print(f"Prompt: {prompt}")
                
                # Generate response
                response = model_fn(prompt)
                
                console.print(f"Response: {response[:100]}...")
                
                # Calculate diagonal strengths
                main_diag_count = sum(1 for row, col in enumerate(path) if row == col)
                anti_diag_count = sum(1 for row, col in enumerate(path) if row + col == matrix.size - 1)
                main_diag_strength = main_diag_count / matrix.size
                anti_diag_strength = anti_diag_count / matrix.size
                
                # Store result
                result = {
                    "path": path,
                    "path_signature": '_'.join(map(str, path)),
                    "prompt": prompt,
                    "response": response,
                    "main_diagonal_affect": matrix.main_diagonal.name,
                    "main_diagonal_strength": main_diag_strength,
                    "anti_diagonal_affect": matrix.anti_diagonal.name,
                    "anti_diagonal_strength": anti_diag_strength,
                }
                
                results.append(result)
                
                # Update progress
                progress.update(task, advance=1)
                
                # Pause between generations to avoid rate limiting
                time.sleep(1)
        
        # Save results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        result_file = os.path.join(
            output_dir, 
            f"{matrix.name}_{model_key}_{timestamp}.json"
        )
        
        with open(result_file, 'w') as f:
            json.dump({
                "matrix_id": matrix.id,
                "matrix_name": matrix.name,
                "model_name": model_key,
                "test_timestamp": timestamp,
                "results": results,
            }, f, indent=2)
            
        console.print(f"Results saved to [bold]{result_file}[/bold]")
        
        return result_file
    
    def test_all_matrices(self,
                         model_key: str,
                         num_paths: int = 3,
                         output_dir: str = "./ekm_results") -> list[str]:
        """
        Test all demo matrices with a model.
        
        Args:
            model_key: Key in MODEL_CONFIGS for the model to use
            num_paths: Number of random paths to test per matrix
            output_dir: Directory to save results
            
        Returns:
            List of paths to saved results files
        """
        matrices = create_demo_matrices()
        result_files = []
        
        for matrix_id, matrix in matrices.items():
            result_file = self.test_matrix(
                matrix=matrix,
                model_key=model_key,
                num_paths=num_paths,
                output_dir=output_dir
            )
            result_files.append(result_file)
            
        return result_files
    
    def batch_test(self,
                  matrices: list[EigenKoanMatrix],
                  model_keys: list[str],
                  num_paths: int = 3,
                  output_dir: str = "./ekm_results") -> list[str]:
        """
        Test multiple matrices with multiple models.
        
        Args:
            matrices: List of EigenKoanMatrix objects to test
            model_keys: List of model keys to test with
            num_paths: Number of random paths to test per matrix
            output_dir: Directory to save results
            
        Returns:
            List of paths to saved results files
        """
        result_files = []
        
        for model_key in model_keys:
            if model_key not in MODEL_CONFIGS:
                console.print(f"[bold red]Unknown model key: {model_key}[/bold red]")
                continue
                
            for matrix in matrices:
                result_file = self.test_matrix(
                    matrix=matrix,
                    model_key=model_key,
                    num_paths=num_paths,
                    output_dir=output_dir
                )
                result_files.append(result_file)
                
        return result_files

def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(description="EKM Local Model Runner")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # List models command
    list_parser = subparsers.add_parser("list-models", help="List available models")
    
    # List matrices command
    list_matrices_parser = subparsers.add_parser("list-matrices", help="List demo matrices")
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Test a matrix with a model")
    test_parser.add_argument("matrix", help="Matrix ID or 'all' to test all demo matrices")
    test_parser.add_argument("model", help="Model key from MODEL_CONFIGS")
    test_parser.add_argument("--paths", type=int, default=3, help="Number of random paths to test")
    test_parser.add_argument("--output", default="./ekm_results", help="Output directory for results")
    
    # Batch test command
    batch_parser = subparsers.add_parser("batch", help="Batch test matrices with models")
    batch_parser.add_argument("--matrices", nargs="+", default=["all"], help="Matrix IDs to test")
    batch_parser.add_argument("--models", nargs="+", required=True, help="Model keys to test")
    batch_parser.add_argument("--paths", type=int, default=3, help="Number of random paths to test")
    batch_parser.add_argument("--output", default="./ekm_results", help="Output directory for results")
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = LocalModelRunner()
    
    # Handle commands
    if args.command == "list-models":
        console.print("[bold]Available models:[/bold]")
        for model_key, config in MODEL_CONFIGS.items():
            console.print(f"  {model_key}: {config['model_name']}")
            
    elif args.command == "list-matrices":
        matrices = create_demo_matrices()
        console.print("[bold]Demo matrices:[/bold]")
        for matrix_id, matrix in matrices.items():
            console.print(f"  {matrix_id}: {matrix.name} ({matrix.size}x{matrix.size})")
            
    elif args.command == "test":
        matrices = create_demo_matrices()
        
        if args.matrix == "all":
            # Test all matrices
            result_files = runner.test_all_matrices(
                model_key=args.model,
                num_paths=args.paths,
                output_dir=args.output
            )
        else:
            # Test a specific matrix
            if args.matrix not in matrices:
                console.print(f"[bold red]Unknown matrix ID: {args.matrix}[/bold red]")
                console.print("Available matrices:")
                for matrix_id in matrices:
                    console.print(f"  {matrix_id}")
                return
                
            matrix = matrices[args.matrix]
            result_file = runner.test_matrix(
                matrix=matrix,
                model_key=args.model,
                num_paths=args.paths,
                output_dir=args.output
            )
            
    elif args.command == "batch":
        matrices = create_demo_matrices()
        
        # Get matrices to test
        test_matrices = []
        if "all" in args.matrices:
            test_matrices = list(matrices.values())
        else:
            for matrix_id in args.matrices:
                if matrix_id in matrices:
                    test_matrices.append(matrices[matrix_id])
                else:
                    console.print(f"[bold red]Unknown matrix ID: {matrix_id}[/bold red]")
                    
        if not test_matrices:
            console.print("[bold red]No valid matrices specified.[/bold red]")
            return
            
        # Run batch test
        result_files = runner.batch_test(
            matrices=test_matrices,
            model_keys=args.models,
            num_paths=args.paths,
            output_dir=args.output
        )
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
