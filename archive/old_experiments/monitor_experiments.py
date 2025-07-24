#!/usr/bin/env python3
"""
Experiment monitor that automatically restarts experiments on failure
"""

import subprocess
import time
import logging
import signal
import sys
from pathlib import Path
from datetime import datetime
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ExperimentMonitor:
    def __init__(self):
        self.experiments = {
            "multitensor": {
                "script": "experiment_multitensor.py",
                "gpu": 5,
                "tmux_session": "gpu5_fixed",
                "conda_env": "unsloth_env",
                "max_retries": 3,
                "retry_count": 0,
                "process": None
            },
            "glm": {
                "script": "experiment_glm.py", 
                "gpu": 4,
                "tmux_session": "gpu4_fixed",
                "conda_env": "unsloth_env",
                "max_retries": 3,
                "retry_count": 0,
                "process": None
            },
            "compress": {
                "script": "experiment_compress_loss.py",
                "gpu": 3,
                "tmux_session": "gpu3_compress",
                "conda_env": "unsloth_env",
                "max_retries": 3,
                "retry_count": 0,
                "process": None
            }
        }
        self.running = True
        
    def check_gpu_availability(self, gpu_id):
        """Check if GPU is available"""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits", f"-i={gpu_id}"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                used, total = map(int, result.stdout.strip().split(', '))
                usage_percent = (used / total) * 100
                return usage_percent < 90  # GPU is available if less than 90% used
            return False
        except:
            return False
    
    def start_experiment(self, exp_name):
        """Start an experiment in tmux"""
        exp = self.experiments[exp_name]
        
        if not self.check_gpu_availability(exp["gpu"]):
            logger.warning(f"GPU {exp['gpu']} not available for {exp_name}")
            return False
            
        logger.info(f"Starting {exp_name} experiment on GPU {exp['gpu']}")
        
        # Kill existing tmux session if exists
        subprocess.run(["tmux", "kill-session", "-t", exp["tmux_session"]], capture_output=True)
        time.sleep(1)
        
        # Create new tmux session and run experiment
        cmd = f"""
        tmux new-session -d -s {exp["tmux_session"]} "
        cd /home/ubuntu/arc-latentseek && 
        source ~/.bashrc && 
        conda activate {exp["conda_env"]} && 
        export CUDA_VISIBLE_DEVICES={exp["gpu"]} && 
        python {exp['script']} 2>&1 | tee logs/{exp_name}_$(date +%Y%m%d_%H%M%S).log
        "
        """
        
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"Successfully started {exp_name} in tmux session {exp['tmux_session']}")
            exp["retry_count"] = 0
            return True
        else:
            logger.error(f"Failed to start {exp_name}: {result.stderr}")
            return False
    
    def check_experiment_status(self, exp_name):
        """Check if experiment is still running"""
        exp = self.experiments[exp_name]
        
        # Check if tmux session exists
        result = subprocess.run(
            ["tmux", "list-sessions", "-F", "#{session_name}"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0 and exp["tmux_session"] in result.stdout:
            # Check for errors in recent logs
            log_files = sorted(Path("logs").glob(f"{exp_name}_*.log"))
            if log_files:
                latest_log = log_files[-1]
                
                # Check last 50 lines for errors
                try:
                    with open(latest_log, 'r') as f:
                        lines = f.readlines()[-50:]
                        content = ''.join(lines)
                        
                        # Check for common error patterns
                        error_patterns = [
                            "AttributeError:",
                            "KeyError:",
                            "RuntimeError:",
                            "CUDA out of memory",
                            "Traceback (most recent call last):",
                            "TypeError:",
                            "ValueError:"
                        ]
                        
                        for pattern in error_patterns:
                            if pattern in content:
                                logger.warning(f"Error detected in {exp_name}: {pattern}")
                                return False
                except:
                    pass
            
            return True
        
        return False
    
    def monitor_loop(self):
        """Main monitoring loop"""
        logger.info("Starting experiment monitor...")
        
        # Create logs directory if it doesn't exist
        Path("logs").mkdir(exist_ok=True)
        
        # Start all experiments initially
        for exp_name in ["multitensor", "glm"]:
            if not self.check_experiment_status(exp_name):
                self.start_experiment(exp_name)
        
        # Monitor loop
        while self.running:
            time.sleep(30)  # Check every 30 seconds
            
            for exp_name, exp in self.experiments.items():
                if exp_name in ["multitensor", "glm"]:  # Only monitor these two
                    if not self.check_experiment_status(exp_name):
                        logger.warning(f"{exp_name} experiment has stopped or encountered an error")
                        
                        if exp["retry_count"] < exp["max_retries"]:
                            exp["retry_count"] += 1
                            logger.info(f"Restarting {exp_name} (attempt {exp['retry_count']}/{exp['max_retries']})")
                            
                            # Wait a bit before restarting
                            time.sleep(10)
                            
                            if self.start_experiment(exp_name):
                                logger.info(f"Successfully restarted {exp_name}")
                            else:
                                logger.error(f"Failed to restart {exp_name}")
                        else:
                            logger.error(f"{exp_name} has failed {exp['max_retries']} times. Not restarting.")
            
            # Log status
            self.log_status()
    
    def log_status(self):
        """Log current status of all experiments"""
        status = {}
        for exp_name in ["multitensor", "glm"]:
            exp = self.experiments[exp_name]
            running = self.check_experiment_status(exp_name)
            
            # Check results
            results_path = Path(f"results/{exp_name}/results.json")
            problems_completed = 0
            if results_path.exists():
                try:
                    with open(results_path) as f:
                        data = json.load(f)
                        problems_completed = len(data.get("problems", []))
                except:
                    pass
            
            status[exp_name] = {
                "running": running,
                "retry_count": exp["retry_count"],
                "problems_completed": problems_completed
            }
        
        logger.info(f"Status: {status}")
    
    def shutdown(self, signum, frame):
        """Graceful shutdown"""
        logger.info("Shutting down monitor...")
        self.running = False
        sys.exit(0)

if __name__ == "__main__":
    monitor = ExperimentMonitor()
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, monitor.shutdown)
    signal.signal(signal.SIGTERM, monitor.shutdown)
    
    try:
        monitor.monitor_loop()
    except Exception as e:
        logger.error(f"Monitor crashed: {e}")
        import traceback
        logger.error(traceback.format_exc())