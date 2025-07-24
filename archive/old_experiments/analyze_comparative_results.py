"""
Analyze comparative experiment results from GPU 5 and 6
"""
import json
import os
from collections import defaultdict

def analyze_results():
    results = defaultdict(lambda: {
        'problems': defaultdict(list),
        'summary': {
            'total_experiments': 0,
            'perfect_solutions': 0,
            'accuracies': [],
            'best_accuracies': [],
            'steps_to_best': [],
            'problems_with_100': set()
        }
    })
    
    # Analyze GPU 5 (basic_basic)
    for file in os.listdir('/home/ubuntu/arc-latentseek/comparative_results_gpu5'):
        if file.endswith('_result.json'):
            with open(f'/home/ubuntu/arc-latentseek/comparative_results_gpu5/{file}', 'r') as f:
                data = json.load(f)
                condition = data['condition']
                problem_id = data['problem_id']
                
                results[condition]['problems'][problem_id].append({
                    'candidate': data['candidate_id'],
                    'final_accuracy': data['final_accuracy'],
                    'best_accuracy': data['best_accuracy'],
                    'best_step': data['best_step'],
                    'total_time': data['total_time']
                })
                
                results[condition]['summary']['total_experiments'] += 1
                results[condition]['summary']['accuracies'].append(data['final_accuracy'])
                results[condition]['summary']['best_accuracies'].append(data['best_accuracy'])
                results[condition]['summary']['steps_to_best'].append(data['best_step'])
                
                if data['best_accuracy'] >= 100.0:
                    results[condition]['summary']['perfect_solutions'] += 1
                    results[condition]['summary']['problems_with_100'].add(problem_id)
    
    # Analyze GPU 6 (basic_multitensor)
    for file in os.listdir('/home/ubuntu/arc-latentseek/comparative_results_gpu6'):
        if file.endswith('_result.json'):
            with open(f'/home/ubuntu/arc-latentseek/comparative_results_gpu6/{file}', 'r') as f:
                data = json.load(f)
                condition = data['condition']
                problem_id = data['problem_id']
                
                results[condition]['problems'][problem_id].append({
                    'candidate': data['candidate_id'],
                    'final_accuracy': data['final_accuracy'],
                    'best_accuracy': data['best_accuracy'],
                    'best_step': data['best_step'],
                    'total_time': data['total_time']
                })
                
                results[condition]['summary']['total_experiments'] += 1
                results[condition]['summary']['accuracies'].append(data['final_accuracy'])
                results[condition]['summary']['best_accuracies'].append(data['best_accuracy'])
                results[condition]['summary']['steps_to_best'].append(data['best_step'])
                
                if data['best_accuracy'] >= 100.0:
                    results[condition]['summary']['perfect_solutions'] += 1
                    results[condition]['summary']['problems_with_100'].add(problem_id)
    
    # Print analysis
    print("="*80)
    print("COMPARATIVE EXPERIMENT ANALYSIS - GPU 5 & 6")
    print("="*80)
    
    for condition, data in results.items():
        print(f"\n{condition.upper()}")
        print("-"*40)
        
        summary = data['summary']
        if summary['total_experiments'] == 0:
            continue
            
        avg_final_acc = sum(summary['accuracies']) / len(summary['accuracies'])
        avg_best_acc = sum(summary['best_accuracies']) / len(summary['best_accuracies'])
        avg_steps = sum(summary['steps_to_best']) / len(summary['steps_to_best'])
        
        print(f"Total experiments: {summary['total_experiments']}")
        print(f"Perfect solutions (100%): {summary['perfect_solutions']}")
        print(f"Average final accuracy: {avg_final_acc:.1f}%")
        print(f"Average best accuracy: {avg_best_acc:.1f}%")
        print(f"Average steps to best: {avg_steps:.1f}")
        print(f"Problems with 100% accuracy: {sorted(summary['problems_with_100'])}")
        
        # Problem breakdown
        print("\nPer-problem results:")
        for problem_id, candidates in sorted(data['problems'].items()):
            best_acc = max(c['best_accuracy'] for c in candidates)
            print(f"  {problem_id}: best={best_acc:.0f}%, candidates={len(candidates)}")
    
    # Compare conditions
    print("\n" + "="*80)
    print("CONDITION COMPARISON")
    print("="*80)
    
    if 'basic_basic' in results and 'basic_multitensor' in results:
        bb_data = results['basic_basic']['summary']
        bm_data = results['basic_multitensor']['summary']
        
        print("\nProblems solved by both conditions:")
        both_solved = bb_data['problems_with_100'].intersection(bm_data['problems_with_100'])
        print(f"  {sorted(both_solved)}")
        
        print("\nProblems solved by basic_basic only:")
        bb_only = bb_data['problems_with_100'] - bm_data['problems_with_100']
        print(f"  {sorted(bb_only)}")
        
        print("\nProblems solved by basic_multitensor only:")
        bm_only = bm_data['problems_with_100'] - bb_data['problems_with_100']
        print(f"  {sorted(bm_only)}")

if __name__ == "__main__":
    analyze_results()