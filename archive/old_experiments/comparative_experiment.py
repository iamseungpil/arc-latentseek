"""
Comparative experiment with 4 conditions:
1. Basic prompt + Basic reward
2. GLM description prompt + Basic reward  
3. Basic prompt + Multi-tensor reward
4. GLM description prompt + Multi-tensor reward
"""

import os
import sys
import json
import logging
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

sys.path.append('/home/ubuntu/arc-latentseek')

from src.data import ARCDataLoader, ARCProblem
from src.generators import BARCGenerator, BARCOutput
from src.executors import CodeExecutor, ExecutionResult, GridRenderer
from src.evaluators import GLMEvaluator, EvaluationResult, RewardModel, MultiTensorEvaluator, convert_multitensor_to_reward
from src.optimizers import LatentSeekOptimizer, OptimizationResult

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentCondition:
    """Single experimental condition"""
    name: str
    use_glm_description: bool
    use_multitensor_reward: bool
    gpu_id: int


@dataclass
class ExperimentResult:
    """Results from a single experiment"""
    condition: str
    problem_id: str
    candidate_id: int
    optimization_steps: List[Dict]  # Step-by-step results
    final_accuracy: float
    final_reward: float
    total_time: float


class ComparativeExperiment:
    """Runs comparative experiments across 4 conditions"""
    
    def __init__(self, output_dir: str = "comparative_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Load GLM descriptions
        with open('glm_problem_descriptions.json', 'r') as f:
            self.glm_descriptions = json.load(f)
        
        # Initialize components
        self.loader = ARCDataLoader()
        self.generator = BARCGenerator()
        self.executor = CodeExecutor()
        self.evaluator = GLMEvaluator()
        self.multitensor_evaluator = MultiTensorEvaluator()
        self.reward_model = RewardModel()
        self.optimizer = LatentSeekOptimizer()
        self.renderer = GridRenderer()
        
        # Experiment conditions
        self.conditions = [
            ExperimentCondition("basic_basic", False, False, 4),
            ExperimentCondition("glm_basic", True, False, 4), 
            ExperimentCondition("basic_multitensor", False, True, 5),
            ExperimentCondition("glm_multitensor", True, True, 5)
        ]
        
        # Selected arc-py validation problems
        self.problems = ['f3e62deb', '639f5a19', '319f2597']
    
    def run_single_experiment(
        self, 
        condition: ExperimentCondition, 
        problem_id: str,
        candidate_idx: int
    ) -> ExperimentResult:
        """Run single experiment for one condition and problem"""
        logger.info(f"Running {condition.name} on {problem_id} candidate {candidate_idx}")
        start_time = time.time()
        
        # Load problem
        problem = self.loader.get_problem_by_id(problem_id)
        
        # Generate candidates with/without GLM description
        candidates = self._generate_candidates(problem, condition)
        candidate = candidates[candidate_idx]
        
        # Initial evaluation
        initial_result = self._evaluate_candidate(candidate, problem, condition)
        
        # Optimization
        optimization_steps = []
        best_reward = initial_result['reward']
        
        # Track step 0 (initial)
        step_result = {
            'step': 0,
            'reward': initial_result['reward'],
            'accuracy': initial_result['accuracy'],
            'description': candidate.description if hasattr(candidate, 'description') else ""
        }
        optimization_steps.append(step_result)
        
        # Run optimization for 5 steps
        for step in range(1, 6):
            # Apply LatentSeek optimization
            optimized_candidate = self._optimize_candidate(
                candidate, problem, condition, step
            )
            
            # Evaluate optimized candidate
            eval_result = self._evaluate_candidate(optimized_candidate, problem, condition)
            
            step_result = {
                'step': step,
                'reward': eval_result['reward'],
                'accuracy': eval_result['accuracy'], 
                'description': optimized_candidate.description if hasattr(optimized_candidate, 'description') else ""
            }
            optimization_steps.append(step_result)
            
            if eval_result['reward'] > best_reward:
                best_reward = eval_result['reward']
                candidate = optimized_candidate
        
        total_time = time.time() - start_time
        
        return ExperimentResult(
            condition=condition.name,
            problem_id=problem_id,
            candidate_id=candidate_idx,
            optimization_steps=optimization_steps,
            final_accuracy=optimization_steps[-1]['accuracy'],
            final_reward=optimization_steps[-1]['reward'],
            total_time=total_time
        )
    
    def _generate_candidates(
        self, 
        problem: ARCProblem, 
        condition: ExperimentCondition
    ) -> List[BARCOutput]:
        """Generate candidates with/without GLM description"""
        
        if condition.use_glm_description:
            # Use GLM description as additional context
            glm_desc = self.glm_descriptions.get(problem.uid, "")
            prompt_addition = f"\nProblem context: {glm_desc}\n"
        else:
            prompt_addition = ""
        
        # Generate 2 candidates
        candidates = self.generator.generate(
            problem, 
            num_candidates=2,
            additional_prompt=prompt_addition
        )
        
        return candidates
    
    def _evaluate_candidate(
        self, 
        candidate: BARCOutput, 
        problem: ARCProblem, 
        condition: ExperimentCondition
    ) -> Dict:
        """Evaluate candidate using appropriate reward function"""
        
        # Execute code
        exec_results = []
        for train_pair in problem.train_pairs:
            result = self.executor.execute(candidate.code, train_pair.x)
            exec_results.append(result)
        
        if condition.use_multitensor_reward:
            # Use multi-tensor evaluation
            generated_outputs = []
            for result in exec_results:
                if result.success and result.output is not None:
                    generated_outputs.append(result.output)
                else:
                    # Use empty grid for failed executions
                    generated_outputs.append(np.zeros_like(problem.train_pairs[0].y))
            
            multitensor_result = self.multitensor_evaluator.evaluate_multitensor(
                problem, generated_outputs
            )
            reward = convert_multitensor_to_reward(multitensor_result)
            
        else:
            # Use basic reward
            glm_result = self.evaluator.evaluate(
                problem, candidate, exec_results
            )
            reward = self.reward_model.compute_reward(exec_results, glm_result.score)
        
        # Calculate accuracy
        accuracy = self._calculate_accuracy(problem, exec_results)
        
        return {
            'reward': reward,
            'accuracy': accuracy,
            'exec_results': exec_results
        }
    
    def _optimize_candidate(
        self, 
        candidate: BARCOutput, 
        problem: ARCProblem, 
        condition: ExperimentCondition,
        step: int
    ) -> BARCOutput:
        """Apply LatentSeek optimization"""
        
        # Use actual LatentSeek optimization
        optimization_result = self.optimizer.optimize(
            problem=problem,
            barc_output=candidate,
            num_steps=1,  # Single step for incremental optimization
            use_description_based=True  # Always use description-based optimization
        )
        
        if optimization_result.optimized_output:
            return optimization_result.optimized_output
        else:
            # Return original if optimization failed
            return candidate
    
    def _calculate_accuracy(
        self, 
        problem: ARCProblem, 
        exec_results: List[ExecutionResult]
    ) -> float:
        """Calculate accuracy percentage"""
        if not exec_results:
            return 0.0
        
        correct = 0
        for i, result in enumerate(exec_results):
            if result.success and result.output is not None:
                expected = problem.train_pairs[i].y
                if np.array_equal(result.output, expected):
                    correct += 1
        
        return (correct / len(exec_results)) * 100.0
    
    def run_all_experiments(self):
        """Run all experiments across conditions and problems"""
        all_results = []
        
        for condition in self.conditions:
            logger.info(f"Running condition: {condition.name}")
            
            for problem_id in self.problems:
                for candidate_idx in range(2):  # 2 candidates per problem
                    result = self.run_single_experiment(
                        condition, problem_id, candidate_idx
                    )
                    all_results.append(result)
                    
                    # Save intermediate results
                    self._save_result(result)
        
        # Save summary
        self._save_summary(all_results)
        return all_results
    
    def _save_result(self, result: ExperimentResult):
        """Save single experiment result"""
        filename = f"{result.condition}_{result.problem_id}_c{result.candidate_id}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(asdict(result), f, indent=2)
    
    def _save_summary(self, results: List[ExperimentResult]):
        """Save experiment summary"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_experiments': len(results),
            'conditions': [c.name for c in self.conditions],
            'problems': self.problems,
            'results_by_condition': {}
        }
        
        # Aggregate by condition
        for condition in self.conditions:
            condition_results = [r for r in results if r.condition == condition.name]
            summary['results_by_condition'][condition.name] = {
                'avg_final_accuracy': np.mean([r.final_accuracy for r in condition_results]),
                'avg_final_reward': np.mean([r.final_reward for r in condition_results]),
                'avg_time': np.mean([r.total_time for r in condition_results]),
                'num_experiments': len(condition_results)
            }
        
        with open(os.path.join(self.output_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info("Experiment summary saved")


def main():
    """Main function"""
    experiment = ComparativeExperiment()
    results = experiment.run_all_experiments()
    
    print("\\n" + "="*60)
    print("COMPARATIVE EXPERIMENT COMPLETED")
    print("="*60)
    
    # Print summary
    for condition in experiment.conditions:
        condition_results = [r for r in results if r.condition == condition.name]
        if condition_results:
            avg_acc = np.mean([r.final_accuracy for r in condition_results])
            avg_reward = np.mean([r.final_reward for r in condition_results])
            print(f"{condition.name}: Acc={avg_acc:.1f}%, Reward={avg_reward:.3f}")


if __name__ == "__main__":
    import numpy as np
    main()