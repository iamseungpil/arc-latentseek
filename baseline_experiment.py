"""
Baseline experiment: BARC-only inference without LatentSeek
Test pure BARC performance on 5 problems with 10 attempts each
"""

import os
import sys
import json
import time
import logging
from typing import Dict, List
from datetime import datetime
from dataclasses import dataclass, asdict

sys.path.append('/home/ubuntu/arc-latentseek')

from src.data import ARCDataLoader
from src.generators import BARCGenerator
from src.executors import CodeExecutor
from src.executors import GridRenderer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class BaselineResult:
    """Result from a single baseline attempt"""
    problem_id: str
    attempt: int
    success: bool
    accuracy: float
    code: str
    description: str
    execution_time: float


class BaselineExperiment:
    """Run baseline BARC-only experiments"""
    
    def __init__(self, output_dir: str = "baseline_results_gpu7"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/logs", exist_ok=True)
        os.makedirs(f"{output_dir}/visualizations", exist_ok=True)
        
        # Initialize components
        logger.info("Initializing components...")
        self.loader = ARCDataLoader()
        self.barc_generator = BARCGenerator()
        self.code_executor = CodeExecutor()
        self.renderer = GridRenderer()
        
        # Use same 5 problems as comparative experiments
        self.problems = ['2072aba6', 'bb52a14b', '136b0064', 'ea9794b1', 'f5aa3634']
        self.attempts_per_problem = 10
        
    def run_single_attempt(self, problem_id: str, attempt_num: int) -> BaselineResult:
        """Run a single BARC inference attempt"""
        logger.info(f"Problem {problem_id}, Attempt {attempt_num + 1}/{self.attempts_per_problem}")
        
        start_time = time.time()
        
        # Load problem
        problem = self.loader.get_problem_by_id(problem_id)
        
        # Generate single candidate with BARC
        # Using same parameters as in main.py
        candidates = self.barc_generator.generate(
            problem,
            num_candidates=1,
            temperature=0.8,
            max_new_tokens=2048
        )
        
        if not candidates:
            logger.error(f"No candidate generated for {problem_id}")
            return BaselineResult(
                problem_id=problem_id,
                attempt=attempt_num,
                success=False,
                accuracy=0.0,
                code="",
                description="Failed to generate candidate",
                execution_time=time.time() - start_time
            )
        
        candidate = candidates[0]
        
        # Execute code
        exec_result = self.code_executor.execute(candidate.code, problem)
        
        # Calculate accuracy
        accuracy = exec_result.accuracy * 100.0  # Convert to percentage
        success = accuracy >= 100.0
        
        execution_time = time.time() - start_time
        
        # Log result
        logger.info(f"  Accuracy: {accuracy:.1f}%, Success: {success}, Time: {execution_time:.2f}s")
        
        # Save detailed log
        log_data = {
            'problem_id': problem_id,
            'attempt': attempt_num,
            'success': success,
            'accuracy': accuracy,
            'code': candidate.code,
            'description': candidate.description if hasattr(candidate, 'description') else "",
            'execution_time': execution_time,
            'execution_success': exec_result.success,
            'timestamp': datetime.now().isoformat()
        }
        
        log_file = os.path.join(
            self.output_dir, 
            "logs", 
            f"{problem_id}_attempt_{attempt_num}.json"
        )
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        # Create visualization for successful attempts
        if success:
            try:
                vis_path = os.path.join(
                    self.output_dir,
                    "visualizations",
                    f"{problem_id}_attempt_{attempt_num}_success.png"
                )
                self.renderer.render_problem_with_output(
                    problem,
                    exec_result.output_grids,
                    vis_path
                )
                logger.info(f"  Saved successful visualization to {vis_path}")
            except Exception as e:
                logger.error(f"  Failed to create visualization: {e}")
        
        return BaselineResult(
            problem_id=problem_id,
            attempt=attempt_num,
            success=success,
            accuracy=accuracy,
            code=candidate.code,
            description=candidate.description if hasattr(candidate, 'description') else "",
            execution_time=execution_time
        )
    
    def run_all_experiments(self):
        """Run all baseline experiments"""
        all_results = []
        problem_results = {}
        
        logger.info("="*60)
        logger.info("Starting BARC Baseline Experiments")
        logger.info(f"Problems: {self.problems}")
        logger.info(f"Attempts per problem: {self.attempts_per_problem}")
        logger.info("="*60)
        
        for problem_id in self.problems:
            logger.info(f"\nTesting problem: {problem_id}")
            logger.info("-"*40)
            
            problem_results[problem_id] = {
                'attempts': [],
                'successes': 0,
                'total_accuracy': 0.0
            }
            
            for attempt in range(self.attempts_per_problem):
                try:
                    result = self.run_single_attempt(problem_id, attempt)
                    all_results.append(result)
                    
                    problem_results[problem_id]['attempts'].append(result)
                    if result.success:
                        problem_results[problem_id]['successes'] += 1
                    problem_results[problem_id]['total_accuracy'] += result.accuracy
                    
                except Exception as e:
                    logger.error(f"Error in attempt {attempt} for {problem_id}: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Problem summary
            success_rate = problem_results[problem_id]['successes'] / self.attempts_per_problem * 100
            avg_accuracy = problem_results[problem_id]['total_accuracy'] / self.attempts_per_problem
            logger.info(f"\n{problem_id} Summary:")
            logger.info(f"  Success rate: {success_rate:.0f}% ({problem_results[problem_id]['successes']}/{self.attempts_per_problem})")
            logger.info(f"  Average accuracy: {avg_accuracy:.1f}%")
        
        # Save summary
        self._save_summary(all_results, problem_results)
        
        return all_results
    
    def _save_summary(self, all_results: List[BaselineResult], problem_results: Dict):
        """Save experiment summary"""
        total_attempts = len(all_results)
        total_successes = sum(1 for r in all_results if r.success)
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_attempts': total_attempts,
            'total_successes': total_successes,
            'overall_success_rate': total_successes / total_attempts * 100 if total_attempts > 0 else 0,
            'problems': self.problems,
            'attempts_per_problem': self.attempts_per_problem,
            'problem_summaries': {}
        }
        
        for problem_id, data in problem_results.items():
            summary['problem_summaries'][problem_id] = {
                'attempts': self.attempts_per_problem,
                'successes': data['successes'],
                'success_rate': data['successes'] / self.attempts_per_problem * 100,
                'average_accuracy': data['total_accuracy'] / self.attempts_per_problem
            }
        
        with open(os.path.join(self.output_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        print("\n" + "="*60)
        print("BASELINE EXPERIMENT SUMMARY")
        print("="*60)
        print(f"Total attempts: {total_attempts}")
        print(f"Total successes: {total_successes}")
        print(f"Overall success rate: {summary['overall_success_rate']:.1f}%")
        print("\nPer-problem results:")
        for problem_id, data in summary['problem_summaries'].items():
            print(f"  {problem_id}: {data['success_rate']:.0f}% success ({data['successes']}/{data['attempts']}), avg accuracy: {data['average_accuracy']:.1f}%")


def main():
    """Main function"""
    experiment = BaselineExperiment()
    results = experiment.run_all_experiments()
    
    print(f"\nResults saved to: baseline_results_gpu7/")


if __name__ == "__main__":
    main()