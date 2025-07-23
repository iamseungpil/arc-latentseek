#!/usr/bin/env python3
"""
Continuous experiment monitor that checks for errors and restarts if needed
"""
import os
import time
import subprocess
import re
from datetime import datetime

def check_experiments_running():
    """Check if tmux sessions are running"""
    try:
        result = subprocess.run(['tmux', 'ls'], capture_output=True, text=True)
        gpu5_running = 'gpu5_experiment' in result.stdout
        gpu6_running = 'gpu6_experiment' in result.stdout
        return gpu5_running, gpu6_running
    except:
        return False, False

def check_for_errors(log_file):
    """Check for recent errors in log file"""
    if not os.path.exists(log_file):
        return None
    
    try:
        # Get last 1000 lines
        result = subprocess.run(['tail', '-1000', log_file], capture_output=True, text=True)
        lines = result.stdout.split('\n')
        
        # Look for errors
        errors = []
        for i, line in enumerate(lines):
            if 'ERROR' in line or 'Error' in line or 'error' in line:
                errors.append((i, line))
        
        # Return most recent error if any
        if errors:
            return errors[-1][1]
        return None
    except:
        return None

def check_progress(log_file):
    """Check experiment progress"""
    if not os.path.exists(log_file):
        return None
    
    try:
        result = subprocess.run(['tail', '-100', log_file], capture_output=True, text=True)
        
        # Count optimization steps
        step_matches = re.findall(r'Step (\d+):', result.stdout)
        last_step = int(step_matches[-1]) if step_matches else 0
        
        # Check for perfect accuracy
        perfect_count = result.stdout.count('Perfect accuracy achieved')
        
        # Check for description extraction
        desc_found = 'Found description tokens' in result.stdout
        
        return {
            'last_step': last_step,
            'perfect_count': perfect_count,
            'description_working': desc_found
        }
    except:
        return None

def restart_experiments():
    """Kill and restart experiments"""
    print(f"\n[{datetime.now()}] Restarting experiments...")
    
    # Kill existing sessions
    subprocess.run(['tmux', 'kill-session', '-t', 'gpu5_experiment'], stderr=subprocess.DEVNULL)
    subprocess.run(['tmux', 'kill-session', '-t', 'gpu6_experiment'], stderr=subprocess.DEVNULL)
    
    # Wait a bit
    time.sleep(5)
    
    # Restart
    subprocess.run(['./run_experiments_final.sh'])
    print("Experiments restarted!")

def main():
    """Main monitoring loop"""
    print("Starting continuous experiment monitor...")
    print("Press Ctrl+C to stop")
    
    error_count = 0
    last_restart_time = time.time()
    
    while True:
        # Find latest results directory
        results_dirs = [d for d in os.listdir('.') if d.startswith('results_comparative_')]
        if not results_dirs:
            print("No results directories found")
            time.sleep(30)
            continue
        
        latest_dir = sorted(results_dirs)[-1]
        gpu5_log = os.path.join(latest_dir, 'gpu5_log.txt')
        gpu6_log = os.path.join(latest_dir, 'gpu6_log.txt')
        
        # Check if experiments are running
        gpu5_running, gpu6_running = check_experiments_running()
        
        print(f"\n[{datetime.now()}] Status Check")
        print(f"Results dir: {latest_dir}")
        print(f"GPU5 running: {gpu5_running}, GPU6 running: {gpu6_running}")
        
        # Check for errors
        gpu5_error = check_for_errors(gpu5_log)
        gpu6_error = check_for_errors(gpu6_log)
        
        if gpu5_error:
            print(f"GPU5 Error: {gpu5_error[:100]}...")
            error_count += 1
        if gpu6_error:
            print(f"GPU6 Error: {gpu6_error[:100]}...")
            error_count += 1
        
        # Check progress
        gpu5_progress = check_progress(gpu5_log)
        gpu6_progress = check_progress(gpu6_log)
        
        if gpu5_progress:
            print(f"GPU5 Progress: Step {gpu5_progress['last_step']}, Perfect: {gpu5_progress['perfect_count']}")
        if gpu6_progress:
            print(f"GPU6 Progress: Step {gpu6_progress['last_step']}, Perfect: {gpu6_progress['perfect_count']}")
        
        # Decide if restart is needed
        need_restart = False
        
        # If experiments crashed
        if not gpu5_running or not gpu6_running:
            print("One or more experiments not running!")
            need_restart = True
        
        # If too many errors
        if error_count > 5:
            print("Too many errors detected!")
            need_restart = True
            error_count = 0
        
        # Don't restart too frequently
        time_since_restart = time.time() - last_restart_time
        if need_restart and time_since_restart > 300:  # 5 minutes
            restart_experiments()
            last_restart_time = time.time()
            error_count = 0
        
        # Wait before next check
        time.sleep(60)

if __name__ == '__main__':
    main()