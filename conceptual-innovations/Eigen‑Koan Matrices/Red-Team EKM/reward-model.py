import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Tuple, Union
import wandb


class PreferenceDataset(Dataset):
    """Dataset for preference pairs generated from EKM traversals"""
    
    def __init__(
        self, 
        preference_pairs: List[Dict],
        tokenizer,
        max_length: int = 512
    ):
        """
        Args:
            preference_pairs: List of dicts with 'prompt', 'chosen', 'rejected' keys
            tokenizer: Tokenizer for encoding inputs
            max_length: Maximum sequence length
        """
        self.preference_pairs = preference_pairs
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.preference_pairs)
    
    def __getitem__(self, idx):
        pair = self.preference_pairs[idx]
        prompt = pair['prompt']
        chosen = pair['chosen']
        rejected = pair['rejected']
        
        # Tokenize with proper formatting
        chosen_tokens = self.tokenizer(
            prompt + chosen,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        rejected_tokens = self.tokenizer(
            prompt + rejected,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        prompt_tokens = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Calculate prompt length for masking
        prompt_length = len(self.tokenizer.encode(prompt))
        
        return {
            "chosen_input_ids": chosen_tokens.input_ids.squeeze(),
            "chosen_attention_mask": chosen_tokens.attention_mask.squeeze(),
            "rejected_input_ids": rejected_tokens.input_ids.squeeze(),
            "rejected_attention_mask": rejected_tokens.attention_mask.squeeze(),
            "prompt_input_ids": prompt_tokens.input_ids.squeeze(),
            "prompt_attention_mask": prompt_tokens.attention_mask.squeeze(),
            "prompt_length": torch.tensor(prompt_length)
        }


class DPOWithKLRegularization(pl.LightningModule):
    """
    Direct Preference Optimization with KL regularization for alignment
    
    This implements the DPO algorithm with KL regularization to prevent the model
    from diverging too far from the reference model, balancing alignment with
    capabilities preservation.
    """
    
    def __init__(
        self,
        model_name_or_path: str,
        ref_model_name_or_path: str = None,
        beta: float = 0.1,
        kl_coef: float = 0.05,
        learning_rate: float = 5e-6
    ):
        """
        Args:
            model_name_or_path: Model to be fine-tuned
            ref_model_name_or_path: Reference model (uses same as model if None)
            beta: Temperature parameter for DPO loss
            kl_coef: KL penalty coefficient
            learning_rate: Learning rate for optimizer
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Load policy model (the one being trained)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        
        # Load reference model (frozen)
        if ref_model_name_or_path is None:
            ref_model_name_or_path = model_name_or_path
        self.ref_model = AutoModelForCausalLM.from_pretrained(ref_model_name_or_path)
        
        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False
            
        self.beta = beta
        self.kl_coef = kl_coef
        self.learning_rate = learning_rate
        
    def forward(self, input_ids, attention_mask=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)
    
    def get_logps(
        self, 
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor, 
        prompt_length: int,
        model: nn.Module
    ) -> torch.Tensor:
        """Get log probabilities from model, masked to only consider response tokens"""
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]  # Shift left
        
        # Select response tokens
        response_range = slice(prompt_length, -1)
        
        # Get logps for response tokens only
        logps = F.log_softmax(logits, dim=-1)
        
        # Select the log probabilities of the actual next tokens
        target_ids = input_ids[:, 1:]  # Shift right
        
        # Gather the logps of the target tokens
        logps_targets = torch.gather(
            logps, 
            dim=2, 
            index=target_ids.unsqueeze(-1)
        ).squeeze(-1)
        
        # Mask out prompt tokens
        response_mask = attention_mask[:, 1:].clone()  # Shift right
        for i in range(len(response_mask)):
            response_mask[i, :prompt_length[i]] = 0  # Mask prompt tokens
        
        # Apply response mask
        logps_targets = logps_targets * response_mask
        
        # Sum log probs over sequence
        response_lengths = response_mask.sum(dim=1)
        logps_sum = logps_targets.sum(dim=1) / response_lengths
        
        return logps_sum
    
    def compute_kl_divergence(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        prompt_length: int
    ) -> torch.Tensor:
        """Compute KL divergence between policy and reference model"""
        # Get outputs from both models
        with torch.no_grad():
            ref_outputs = self.ref_model(input_ids=input_ids, attention_mask=attention_mask)
        policy_outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Get logits and shift
        ref_logits = ref_outputs.logits[:, :-1, :]
        policy_logits = policy_outputs.logits[:, :-1, :]
        
        # Compute KL divergence only on response tokens
        response_mask = attention_mask[:, 1:].clone()
        for i in range(len(response_mask)):
            response_mask[i, :prompt_length[i]] = 0
        
        # Compute KL
        kl_div = F.kl_div(
            F.log_softmax(policy_logits, dim=-1),
            F.softmax(ref_logits, dim=-1),
            reduction='none'
        ).sum(dim=-1)
        
        # Apply response mask and average
        kl_div = (kl_div * response_mask).sum(dim=1) / response_mask.sum(dim=1)
        
        return kl_div
        
    def training_step(self, batch, batch_idx):
        # Calculate log probs for chosen and rejected completions
        chosen_logps = self.get_logps(
            batch["chosen_input_ids"],
            batch["chosen_attention_mask"],
            batch["prompt_length"],
            self.model
        )
        
        rejected_logps = self.get_logps(
            batch["rejected_input_ids"],
            batch["rejected_attention_mask"],
            batch["prompt_length"],
            self.model
        )
        
        # Get reference model log probs
        with torch.no_grad():
            ref_chosen_logps = self.get_logps(
                batch["chosen_input_ids"],
                batch["chosen_attention_mask"],
                batch["prompt_length"],
                self.ref_model
            )
            
            ref_rejected_logps = self.get_logps(
                batch["rejected_input_ids"],
                batch["rejected_attention_mask"],
                batch["prompt_length"],
                self.ref_model
            )
        
        # Compute advantages
        chosen_rewards = chosen_logps - ref_chosen_logps.detach()
        rejected_rewards = rejected_logps - ref_rejected_logps.detach()
        
        # Compute DPO loss
        logits = self.beta * (chosen_rewards - rejected_rewards)
        loss_dpo = -F.logsigmoid(logits).mean()
        
        # Compute KL divergence penalty
        kl_chosen = self.compute_kl_divergence(
            batch["chosen_input_ids"],
            batch["chosen_attention_mask"],
            batch["prompt_length"]
        )
        
        kl_rejected = self.compute_kl_divergence(
            batch["rejected_input_ids"],
            batch["rejected_attention_mask"],
            batch["prompt_length"]
        )
        
        kl_penalty = (kl_chosen + kl_rejected).mean() / 2
        
        # Final loss with KL regularization
        loss = loss_dpo + self.kl_coef * kl_penalty
        
        # Log metrics
        self.log("train/loss", loss)
        self.log("train/loss_dpo", loss_dpo)
        self.log("train/kl_penalty", kl_penalty)
        self.log("train/chosen_rewards", chosen_rewards.mean())
        self.log("train/rejected_rewards", rejected_rewards.mean())
        self.log("train/reward_gap", (chosen_rewards - rejected_rewards).mean())
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        # Similar implementation as training step, but without gradient calculation
        with torch.no_grad():
            chosen_logps = self.get_logps(
                batch["chosen_input_ids"],
                batch["chosen_attention_mask"],
                batch["prompt_length"],
                self.model
            )
            
            rejected_logps = self.get_logps(
                batch["rejected_input_ids"],
                batch["rejected_attention_mask"],
                batch["prompt_length"],
                self.model
            )
            
            ref_chosen_logps = self.get_logps(
                batch["chosen_input_ids"],
                batch["chosen_attention_mask"],
                batch["prompt_length"],
                self.ref_model
            )
            
            ref_rejected_logps = self.get_logps(
                batch["rejected_input_ids"],
                batch["rejected_attention_mask"],
                batch["prompt_length"],
                self.ref_model
            )
        
        # Compute metrics
        chosen_rewards = chosen_logps - ref_chosen_logps
        rejected_rewards = rejected_logps - ref_rejected_logps
        
        # Log validation metrics
        self.log("val/chosen_rewards", chosen_rewards.mean())
        self.log("val/rejected_rewards", rejected_rewards.mean())
        self.log("val/reward_gap", (chosen_rewards - rejected_rewards).mean())
        
        # Compute accuracy (% where chosen > rejected)
        accuracy = (chosen_logps > rejected_logps).float().mean()
        self.log("val/accuracy", accuracy)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)


class EKMPreferenceGenerator:
    """
    Generates preference pairs from EKM traversals for reward model training
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        ekm_grid,
        safety_evaluator
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.ekm_grid = ekm_grid
        self.safety_evaluator = safety_evaluator
    
    def generate_preference_pairs(self, num_pairs=1000):
        """Generate preference pairs from EKM traversals"""
        preference_pairs = []
        
        # Generate traversal paths
        traverser = AdversarialTraverser(
            self.ekm_grid,
            self._model_evaluator,
            self.safety_evaluator
        )
        findings = traverser.traverse()
        
        # Use findings to create preference pairs
        for finding in findings:
            if "trigger_prompt" in finding and "responses" in finding:
                prompt = finding["trigger_prompt"]
                
                # Get response with lowest safety score (rejected)
                safety_scores = [
                    self.safety_evaluator(prompt, resp) for resp in finding["responses"]
                ]
                
                if len(safety_scores) >= 2:
                    # Find min and max safety responses
                    min_idx = safety_scores.index(min(safety_scores))
                    max_idx = safety_scores.index(max(safety_scores))
                    
                    # Only add if there's a meaningful difference
                    if safety_scores[max_idx] - safety_scores[min_idx] > 0.2:
                        pair = {
                            "prompt": prompt,
                            "chosen": finding["responses"][max_idx],
                            "rejected": finding["responses"][min_idx],
                            "safety_gap": safety_scores[max_idx] - safety_scores[min_idx]
                        }
                        preference_pairs.append(pair)
        
        # If needed, generate more pairs with systematic variations
        while len(preference_pairs) < num_pairs:
            # Implementation of additional pair generation
            pass
            
        return preference_pairs
    
    def _model_evaluator(self, prompt):
        """Evaluate model on prompt"""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(
            inputs.input_ids,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


def generate_ekm_dpo_dataset(ekm_grid, model, tokenizer, safety_evaluator, num_pairs=5000):
    """Generate a DPO dataset from EKM traversals"""
    generator = EKMPreferenceGenerator(model, tokenizer, ekm_grid, safety_evaluator)
    preference_pairs = generator.generate_preference_pairs(num_pairs)
    
    # Create train/val split
    from sklearn.model_selection import train_test_split
    train_pairs, val_pairs = train_test_split(preference_pairs, test_size=0.1)
    
    # Create datasets
    train_dataset = PreferenceDataset(train_pairs, tokenizer)
    val_dataset = PreferenceDataset(val_pairs, tokenizer)
    
    return train_dataset, val_dataset


def train_dpo_model(
    model_name: str,
    train_dataset: Dataset,
    val_dataset: Dataset,
    output_dir: str = "dpo_model",
    batch_size: int = 8,
    max_epochs: int = 3,
    beta: float = 0.1,
    kl_coef: float = 0.05,
    learning_rate: float = 5e-6
):
    """Train a DPO model using Lightning"""
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Create model
    model = DPOWithKLRegularization(
        model_name_or_path=model_name,
        beta=beta,
        kl_coef=kl_coef,
        learning_rate=learning_rate
    )
    
    # Setup wandb logging
    wandb_logger = pl.loggers.WandbLogger(project="ekm-dpo")
    
    # Setup callbacks
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=output_dir,
        filename="dpo-model-{epoch:02d}-{val/reward_gap:.4f}",
        monitor="val/reward_gap",
        mode="max",
        save_top_k=3
    )
    
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor="val/reward_gap",
        patience=3,
        mode="max"
    )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None
    )
    
    # Train model
    trainer.fit(model, train_loader, val_loader)
    
    # Return best model path
    return checkpoint_callback.best_model_path 
