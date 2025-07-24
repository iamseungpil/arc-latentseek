#!/usr/bin/env python3
"""
Simple experiment runner with fixed LatentSeek implementation
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data import load_arc_problems, ARCProblem
from src.generators import BARCGenerator, GeneratorConfig
from src.executors import CodeExecutor
from src.evaluators import GLMEvaluator
from src.optimizers import LatentSeekOptimizer

# Import the fixed opt_generation module
sys.path.insert(0, str(project_root / "src" / "optimizers"))
from opt_generation_fixed import optimized_generation


class FixedLatentSeekOptimizer(LatentSeekOptimizer):
    """Override with fixed opt_generation implementation"""
    
    def _generate_from_optimized_states(self, 
                                      problem: ARCProblem,
                                      optimized_states: torch.Tensor,
                                      base_input_ids: torch.Tensor,
                                      start_index: int,
                                      prompt_length: int) -> Optional[BARCOutput]:
        """Use fixed opt_generation logic"""
        try:
            # Create a simple reward model wrapper
            class SimpleRewardModel:
                def __init__(self, evaluator, problem):
                    self.evaluator = evaluator
                    self.problem = problem
                
                def get_reward(self, question, answer):
                    # Extract code from answer if needed
                    code = self._extract_code(answer)
                    if not code:
                        return -1.0
                    
                    result = self.evaluator.evaluate_single(self.problem, code)
                    return result.reward
                
                def _extract_code(self, text):
                    import re
                    code_pattern = r'```python\n(.*?)\n```'
                    match = re.search(code_pattern, text, re.DOTALL)
                    if match:
                        return match.group(1)
                    return text if 'def ' in text else None
            
            reward_model = SimpleRewardModel(self.glm_evaluator, problem)
            
            # Generate using fixed implementation
            final_answer, reward_history, _, _, _ = optimized_generation(
                reward_model=reward_model,
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.model.device,
                question=str(problem),
                input_text=self.tokenizer.apply_chat_template(
                    self.barc_generator._create_prompt(problem),
                    tokenize=False,
                    add_generation_prompt=True
                ),
                original_answer="",  # Not used in our case
                original_hidden_states_list=optimized_states.unbind(0),  # Convert to list
                input_ids=base_input_ids,
                start_index=start_index,
                max_num_steps=1,  # Single step since we're already in optimization loop
                lr=self.lr,
                k=self.k,
                prompt_length=prompt_length
            )
            
            # Extract code and create BARCOutput
            code = reward_model._extract_code(final_answer)
            if code:
                return BARCOutput(
                    code=code,
                    description=None,
                    concepts=None,
                    plan=None,
                    raw_response=final_answer
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error in fixed generation: {e}")
            return None


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'simple_experiment_fixed_gpu{sys.argv[1] if len(sys.argv) > 1 else "5"}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def run_experiment(problem_ids: List[str], gpu_id: int = 5):
    """Run fixed experiment on specified problems"""
    
    # Configuration
    config = GeneratorConfig(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        temperature=0.7,
        max_tokens=2048,
        device=f"cuda:{gpu_id}"
    )
    
    # Initialize components
    logger.info(f"Initializing components on GPU {gpu_id}")
    generator = BARCGenerator(config)
    executor = CodeExecutor()
    evaluator = GLMEvaluator(device=config.device)
    optimizer = FixedLatentSeekOptimizer(
        barc_generator=generator,
        code_executor=executor,
        glm_evaluator=evaluator,
        lr=0.03,
        max_steps=10,
        k=0.2,  # Optimize 20% of tokens
        reward_threshold=0.5
    )
    
    # Load problems
    problems = load_arc_problems("evaluation")
    selected_problems = {pid: problems[pid] for pid in problem_ids if pid in problems}
    
    results = {}
    
    for problem_id, problem in selected_problems.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Running fixed experiment on {problem_id}")
        logger.info(f"{'='*60}")
        
        try:
            # Generate initial candidates
            logger.info("Generating initial candidates...")
            candidates = generator.generate(problem, num_candidates=2)
            
            for i, candidate in enumerate(candidates):
                logger.info(f"\nProcessing candidate {i}")
                
                # Evaluate initial candidate
                initial_result = evaluator.evaluate_single(problem, candidate.code)
                logger.info(f"Initial evaluation: accuracy={initial_result.accuracy:.1%}, reward={initial_result.reward:.3f}")
                
                # Apply fixed optimization
                logger.info("Applying fixed LatentSeek optimization...")
                opt_result = optimizer.optimize(problem, candidate, initial_result.reward)
                
                # Evaluate optimized result
                final_result = evaluator.evaluate_single(problem, opt_result.final_output.code)
                logger.info(f"Final evaluation: accuracy={final_result.accuracy:.1%}, reward={final_result.reward:.3f}")
                
                # Store results
                result_key = f"{problem_id}_c{i}"
                results[result_key] = {
                    "problem_id": problem_id,
                    "candidate_idx": i,
                    "initial_reward": initial_result.reward,
                    "initial_accuracy": initial_result.accuracy,
                    "final_reward": final_result.reward,
                    "final_accuracy": final_result.accuracy,
                    "optimization_steps": opt_result.optimization_steps,
                    "converged": opt_result.converged,
                    "reward_history": opt_result.reward_history
                }
                
                # Save intermediate results
                with open(f'fixed_results_gpu{gpu_id}/intermediate_{result_key}.json', 'w') as f:
                    json.dump(results[result_key], f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed on problem {problem_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    # Save final results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(f'fixed_results_gpu{gpu_id}', exist_ok=True)
    
    with open(f'fixed_results_gpu{gpu_id}/results_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("="*60)
    
    total_initial_accuracy = sum(r["initial_accuracy"] for r in results.values()) / len(results)
    total_final_accuracy = sum(r["final_accuracy"] for r in results.values()) / len(results)
    
    logger.info(f"Total problems: {len(selected_problems)}")
    logger.info(f"Total candidates: {len(results)}")
    logger.info(f"Average initial accuracy: {total_initial_accuracy:.1%}")
    logger.info(f"Average final accuracy: {total_final_accuracy:.1%}")
    logger.info(f"Improvement: {total_final_accuracy - total_initial_accuracy:.1%}")


if __name__ == "__main__":
    import torch
    
    # Get GPU ID from command line
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    
    # Select problems (use same ones as before for comparison)
    problem_ids = ["2a5f8217", "bf89d739", "feca6190"]
    
    # Additional problems if specified
    if "--all" in sys.argv:
        all_problems = load_arc_problems("evaluation")
        problem_ids = list(all_problems.keys())[:10]  # First 10 problems
    
    run_experiment(problem_ids, gpu_id)