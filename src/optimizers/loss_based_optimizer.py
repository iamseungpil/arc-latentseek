"""
Loss-based LatentSeek optimizer inspired by CompressARC
Replaces policy gradient with direct loss optimization
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
import logging

from ..data import ARCProblem, BARCOutput
from ..generators.barc_generator import BARCGenerator
from ..executors.code_executor import CodeExecutor
from ..evaluators.arc_evaluator import ARCEvaluator

logger = logging.getLogger(__name__)


@dataclass
class LossOptimizationResult:
    """Result from loss-based optimization"""
    initial_output: BARCOutput
    final_output: BARCOutput
    initial_loss: float
    final_loss: float
    best_output: BARCOutput
    best_loss: float
    optimization_steps: int
    success: bool


class LossBasedOptimizer:
    """Loss-based hidden state optimizer inspired by CompressARC"""
    
    def __init__(
        self,
        model,
        tokenizer,
        lr: float = 0.01,
        max_steps: int = 20,
        kl_weight: float = 0.1,
        early_stop_threshold: float = 0.01
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.lr = lr
        self.max_steps = max_steps
        self.kl_weight = kl_weight
        self.early_stop_threshold = early_stop_threshold
        
        # Component modules
        self.executor = CodeExecutor()
        self.evaluator = ARCEvaluator()
    
    def optimize_with_loss(
        self,
        problem: ARCProblem,
        initial_output: BARCOutput,
        target_outputs: Optional[List[np.ndarray]] = None
    ) -> LossOptimizationResult:
        """
        Optimize hidden states using direct loss instead of policy gradient
        
        Args:
            problem: ARC problem to solve
            initial_output: Initial BARC output with code
            target_outputs: Optional target outputs for supervised loss
        
        Returns:
            LossOptimizationResult with optimized output
        """
        logger.info(f"Starting loss-based optimization for problem {problem.uid}")
        
        # Get initial hidden states
        hidden_states, token_positions = self._capture_hidden_states(
            problem, initial_output
        )
        
        if hidden_states is None:
            logger.error("Failed to capture hidden states")
            return self._create_failed_result(initial_output)
        
        # Find description region for focused optimization
        desc_start, desc_end = self._find_description_region(
            initial_output, token_positions
        )
        
        if desc_start is None or desc_end is None:
            logger.warning("No description found, optimizing full sequence")
            desc_start, desc_end = 0, len(hidden_states)
        
        # Extract region to optimize
        original_hidden = hidden_states[desc_start:desc_end].clone().detach()
        optimized_hidden = torch.nn.Parameter(hidden_states[desc_start:desc_end].clone())
        
        # Setup optimizer
        optimizer = torch.optim.Adam([optimized_hidden], lr=self.lr)
        
        best_loss = float('inf')
        best_output = initial_output
        best_hidden = optimized_hidden.clone()
        
        # Optimization loop
        for step in range(self.max_steps):
            optimizer.zero_grad()
            
            # Compute multi-component loss
            total_loss, loss_components = self._compute_loss(
                optimized_hidden,
                original_hidden,
                hidden_states,
                desc_start,
                desc_end,
                problem,
                target_outputs
            )
            
            # Backward and optimize
            total_loss.backward()
            optimizer.step()
            
            # Generate new output with optimized hidden states
            new_hidden_states = hidden_states.clone()
            new_hidden_states[desc_start:desc_end] = optimized_hidden
            
            new_output = self._generate_from_hidden_states(
                new_hidden_states, problem
            )
            
            # Track best
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_output = new_output
                best_hidden = optimized_hidden.clone()
            
            # Log progress
            if step % 5 == 0:
                logger.info(
                    f"Step {step}: loss={total_loss.item():.4f}, "
                    f"components={loss_components}"
                )
            
            # Early stopping
            if total_loss.item() < self.early_stop_threshold:
                logger.info(f"Early stopping at step {step}")
                break
        
        # Create final output with best hidden states
        final_hidden_states = hidden_states.clone()
        final_hidden_states[desc_start:desc_end] = best_hidden
        final_output = self._generate_from_hidden_states(
            final_hidden_states, problem
        )
        
        return LossOptimizationResult(
            initial_output=initial_output,
            final_output=final_output,
            initial_loss=float('inf'),  # Not computed
            final_loss=best_loss,
            best_output=best_output,
            best_loss=best_loss,
            optimization_steps=step + 1,
            success=True
        )
    
    def _compute_loss(
        self,
        optimized_hidden: torch.Tensor,
        original_hidden: torch.Tensor,
        full_hidden_states: torch.Tensor,
        start_idx: int,
        end_idx: int,
        problem: ARCProblem,
        target_outputs: Optional[List[np.ndarray]] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute multi-component loss for optimization
        
        Components:
        1. KL divergence (regularization)
        2. Structure preservation loss
        3. Pattern consistency loss
        4. Output validity loss
        5. Target matching loss (if targets provided)
        """
        loss_components = {}
        
        # 1. KL Divergence Loss (CompressARC style)
        # Ensures optimized hidden states don't deviate too much from original
        kl_loss = F.kl_div(
            F.log_softmax(optimized_hidden.view(-1, optimized_hidden.size(-1)), dim=-1),
            F.softmax(original_hidden.view(-1, original_hidden.size(-1)), dim=-1),
            reduction='batchmean'
        )
        loss_components['kl'] = kl_loss.item()
        
        # 2. Structure Preservation Loss
        # Maintains structural coherence of hidden representations
        structure_loss = self._compute_structure_loss(
            optimized_hidden, original_hidden
        )
        loss_components['structure'] = structure_loss.item()
        
        # 3. Pattern Consistency Loss
        # Ensures pattern recognition capabilities are preserved
        pattern_loss = self._compute_pattern_loss(
            optimized_hidden, full_hidden_states, start_idx, end_idx
        )
        loss_components['pattern'] = pattern_loss.item()
        
        # 4. Output Validity Loss
        # Encourages valid ARC grid outputs
        validity_loss = self._compute_validity_loss(
            optimized_hidden, problem
        )
        loss_components['validity'] = validity_loss.item()
        
        # 5. Target Matching Loss (if available)
        if target_outputs is not None:
            target_loss = self._compute_target_loss(
                optimized_hidden, full_hidden_states, start_idx, end_idx,
                problem, target_outputs
            )
            loss_components['target'] = target_loss.item()
        else:
            target_loss = 0.0
        
        # Weighted combination
        total_loss = (
            self.kl_weight * kl_loss +
            0.3 * structure_loss +
            0.2 * pattern_loss +
            0.2 * validity_loss +
            0.3 * target_loss if target_outputs else 0.0
        )
        
        return total_loss, loss_components
    
    def _compute_structure_loss(
        self,
        optimized: torch.Tensor,
        original: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute structural similarity loss
        Preserves relative distances between hidden states
        """
        # Compute pairwise distances
        opt_distances = torch.cdist(optimized, optimized)
        orig_distances = torch.cdist(original, original)
        
        # MSE between distance matrices
        structure_loss = F.mse_loss(opt_distances, orig_distances)
        
        return structure_loss
    
    def _compute_pattern_loss(
        self,
        optimized: torch.Tensor,
        full_hidden: torch.Tensor,
        start_idx: int,
        end_idx: int
    ) -> torch.Tensor:
        """
        Compute pattern consistency loss
        Ensures optimized region maintains coherence with surrounding context
        """
        # Get context windows
        context_before = full_hidden[max(0, start_idx-10):start_idx]
        context_after = full_hidden[end_idx:min(len(full_hidden), end_idx+10)]
        
        if len(context_before) > 0 and len(context_after) > 0:
            # Compute attention-like scores between optimized and context
            before_similarity = F.cosine_similarity(
                optimized[:5].mean(dim=0, keepdim=True),
                context_before.mean(dim=0, keepdim=True)
            )
            after_similarity = F.cosine_similarity(
                optimized[-5:].mean(dim=0, keepdim=True),
                context_after.mean(dim=0, keepdim=True)
            )
            
            # Loss encourages high similarity with context
            pattern_loss = 2.0 - before_similarity - after_similarity
        else:
            pattern_loss = torch.tensor(0.0, device=optimized.device)
        
        return pattern_loss
    
    def _compute_validity_loss(
        self,
        optimized: torch.Tensor,
        problem: ARCProblem
    ) -> torch.Tensor:
        """
        Compute output validity loss
        Encourages generation of valid ARC grids
        """
        # Project to logits space
        if hasattr(self.model, 'lm_head'):
            logits = self.model.lm_head(optimized)
        else:
            # Fallback: use mean hidden state similarity
            return torch.tensor(0.0, device=optimized.device)
        
        # Compute entropy - lower entropy means more confident predictions
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
        
        # Encourage lower entropy (more confident predictions)
        validity_loss = entropy
        
        return validity_loss
    
    def _compute_target_loss(
        self,
        optimized: torch.Tensor,
        full_hidden: torch.Tensor,
        start_idx: int,
        end_idx: int,
        problem: ARCProblem,
        target_outputs: List[np.ndarray]
    ) -> torch.Tensor:
        """
        Compute loss against target outputs (if available)
        This enables semi-supervised optimization
        """
        # Generate output from current hidden states
        current_hidden = full_hidden.clone()
        current_hidden[start_idx:end_idx] = optimized
        
        # Get predicted tokens
        if hasattr(self.model, 'lm_head'):
            logits = self.model.lm_head(current_hidden)
            predicted_tokens = torch.argmax(logits, dim=-1)
            
            # Decode to text
            predicted_text = self.tokenizer.decode(predicted_tokens)
            
            # Execute code and get outputs
            try:
                # Extract code from predicted text
                code = self._extract_code_from_text(predicted_text)
                outputs = []
                for train_pair in problem.train_pairs:
                    result = self.executor.execute(code, train_pair.x)
                    if result['success']:
                        outputs.append(result['output'])
                
                # Compute accuracy against targets
                if len(outputs) == len(target_outputs):
                    accuracies = []
                    for pred, target in zip(outputs, target_outputs):
                        if pred.shape == target.shape:
                            accuracy = (pred == target).mean()
                            accuracies.append(accuracy)
                    
                    # Loss is negative accuracy
                    target_loss = 1.0 - torch.tensor(np.mean(accuracies), device=optimized.device)
                else:
                    target_loss = torch.tensor(1.0, device=optimized.device)
            except:
                # If execution fails, high loss
                target_loss = torch.tensor(1.0, device=optimized.device)
        else:
            target_loss = torch.tensor(0.0, device=optimized.device)
        
        return target_loss
    
    def _capture_hidden_states(
        self,
        problem: ARCProblem,
        output: BARCOutput
    ) -> Tuple[Optional[torch.Tensor], Optional[List[int]]]:
        """Capture hidden states during generation"""
        # Implementation would be similar to LatentSeek's capture method
        # Simplified placeholder
        return None, None
    
    def _find_description_region(
        self,
        output: BARCOutput,
        token_positions: List[int]
    ) -> Tuple[Optional[int], Optional[int]]:
        """Find description token positions"""
        # Implementation would locate description in code
        # Simplified placeholder
        return None, None
    
    def _generate_from_hidden_states(
        self,
        hidden_states: torch.Tensor,
        problem: ARCProblem
    ) -> BARCOutput:
        """Generate new output from modified hidden states"""
        # Implementation would generate text from hidden states
        # Simplified placeholder
        return BARCOutput(
            code="# Generated code",
            description="Generated description",
            concepts="Generated concepts",
            plan="Generated plan"
        )
    
    def _extract_code_from_text(self, text: str) -> str:
        """Extract code from generated text"""
        # Simple extraction between code markers
        if "```python" in text:
            start = text.find("```python") + 9
            end = text.find("```", start)
            if end > start:
                return text[start:end].strip()
        return text
    
    def _create_failed_result(
        self,
        initial_output: BARCOutput
    ) -> LossOptimizationResult:
        """Create a failed optimization result"""
        return LossOptimizationResult(
            initial_output=initial_output,
            final_output=initial_output,
            initial_loss=float('inf'),
            final_loss=float('inf'),
            best_output=initial_output,
            best_loss=float('inf'),
            optimization_steps=0,
            success=False
        )


class MultiTensorLossOptimizer(LossBasedOptimizer):
    """
    Extended loss-based optimizer with multi-tensor evaluation components
    Replaces GLM-based reward with direct loss computation
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Loss component weights inspired by CompressARC
        self.loss_weights = {
            'example_accuracy': 0.3,
            'color_transformation': 0.2,
            'spatial_transformation': 0.2,
            'pattern_recognition': 0.2,
            'structural_integrity': 0.1
        }
    
    def compute_multitensor_loss(
        self,
        generated_outputs: List[np.ndarray],
        expected_outputs: List[np.ndarray]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute multi-dimensional loss without GLM
        Direct computation of 5 tensor components
        """
        losses = {}
        
        # 1. Example Accuracy Loss
        example_losses = []
        for gen, exp in zip(generated_outputs, expected_outputs):
            if gen.shape == exp.shape:
                # Pixel-wise accuracy
                accuracy = (gen == exp).mean()
                example_losses.append(1.0 - accuracy)
            else:
                example_losses.append(1.0)  # Max loss for shape mismatch
        
        losses['example_accuracy'] = np.mean(example_losses)
        
        # 2. Color Transformation Loss
        color_loss = self._compute_color_transformation_loss(
            generated_outputs, expected_outputs
        )
        losses['color_transformation'] = color_loss
        
        # 3. Spatial Transformation Loss
        spatial_loss = self._compute_spatial_transformation_loss(
            generated_outputs, expected_outputs
        )
        losses['spatial_transformation'] = spatial_loss
        
        # 4. Pattern Recognition Loss
        pattern_loss = self._compute_pattern_recognition_loss(
            generated_outputs, expected_outputs
        )
        losses['pattern_recognition'] = pattern_loss
        
        # 5. Structural Integrity Loss
        structure_loss = self._compute_structural_integrity_loss(
            generated_outputs, expected_outputs
        )
        losses['structural_integrity'] = structure_loss
        
        # Weighted combination
        total_loss = sum(
            losses[key] * self.loss_weights[key] 
            for key in losses.keys()
        )
        
        return torch.tensor(total_loss, requires_grad=True), losses
    
    def _compute_color_transformation_loss(
        self,
        generated: List[np.ndarray],
        expected: List[np.ndarray]
    ) -> float:
        """Compute color mapping accuracy loss"""
        color_errors = []
        
        for gen, exp in zip(generated, expected):
            if gen.shape != exp.shape:
                color_errors.append(1.0)
                continue
            
            # Get unique colors in expected
            exp_colors = set(exp.flatten())
            gen_colors = set(gen.flatten())
            
            # Check if color mapping is consistent
            color_map = {}
            for i in range(exp.shape[0]):
                for j in range(exp.shape[1]):
                    exp_color = exp[i, j]
                    gen_color = gen[i, j]
                    
                    if exp_color in color_map:
                        if color_map[exp_color] != gen_color:
                            # Inconsistent mapping
                            color_errors.append(1.0)
                            break
                    else:
                        color_map[exp_color] = gen_color
        
        return np.mean(color_errors) if color_errors else 0.0
    
    def _compute_spatial_transformation_loss(
        self,
        generated: List[np.ndarray],
        expected: List[np.ndarray]
    ) -> float:
        """Compute spatial accuracy loss"""
        spatial_errors = []
        
        for gen, exp in zip(generated, expected):
            if gen.shape != exp.shape:
                # Shape mismatch is a spatial error
                spatial_errors.append(1.0)
            else:
                # Compute structural similarity
                # Check row/column patterns
                row_sim = np.mean([
                    1.0 - np.corrcoef(gen[i], exp[i])[0, 1] 
                    if np.std(gen[i]) > 0 and np.std(exp[i]) > 0 else 0.5
                    for i in range(gen.shape[0])
                ])
                col_sim = np.mean([
                    1.0 - np.corrcoef(gen[:, j], exp[:, j])[0, 1]
                    if np.std(gen[:, j]) > 0 and np.std(exp[:, j]) > 0 else 0.5
                    for j in range(gen.shape[1])
                ])
                spatial_errors.append((row_sim + col_sim) / 2)
        
        return np.mean(spatial_errors) if spatial_errors else 0.0
    
    def _compute_pattern_recognition_loss(
        self,
        generated: List[np.ndarray],
        expected: List[np.ndarray]
    ) -> float:
        """Compute pattern detection accuracy loss"""
        # Check if patterns are consistently applied
        pattern_errors = []
        
        # For simplicity, check if the transformation is consistent
        # across all examples
        if len(generated) > 1:
            # Compare transformations between examples
            for i in range(len(generated) - 1):
                if generated[i].shape == expected[i].shape and \
                   generated[i+1].shape == expected[i+1].shape:
                    # Check if similar patterns produce similar outputs
                    pattern_sim = 1.0 - np.corrcoef(
                        generated[i].flatten(), 
                        generated[i+1].flatten()
                    )[0, 1] if generated[i].size > 1 else 0.5
                    pattern_errors.append(pattern_sim)
                else:
                    pattern_errors.append(1.0)
        
        return np.mean(pattern_errors) if pattern_errors else 0.0
    
    def _compute_structural_integrity_loss(
        self,
        generated: List[np.ndarray],
        expected: List[np.ndarray]
    ) -> float:
        """Compute structural preservation loss"""
        structure_errors = []
        
        for gen, exp in zip(generated, expected):
            # Check basic structural properties
            errors = 0.0
            
            # 1. Shape match
            if gen.shape != exp.shape:
                errors += 0.5
            
            # 2. Value range check (ARC uses 0-9)
            if np.any(gen < 0) or np.any(gen > 9):
                errors += 0.25
            
            # 3. Object count preservation (simplified)
            gen_objects = len(np.unique(gen))
            exp_objects = len(np.unique(exp))
            if gen_objects != exp_objects:
                errors += 0.25
            
            structure_errors.append(errors)
        
        return np.mean(structure_errors) if structure_errors else 0.0