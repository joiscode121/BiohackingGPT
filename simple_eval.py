from datetime import datetime
import json
import os
import time

class SimpleEval:
    def __init__(self):
        """Initialize simple evaluation system"""
        self.log_file = "eval_logs.jsonl"
        self.ensure_log_file()
    
    def ensure_log_file(self):
        """Create log file if it doesn't exist"""
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                pass  # Create empty file
    
    def reset_metrics(self):
        """Reset all evaluation metrics by clearing the log file"""
        try:
            with open(self.log_file, 'w') as f:
                pass  # Clear file contents
            return True
        except Exception as e:
            print(f"Failed to reset metrics: {str(e)}")
            return False
    
    def log_interaction(self, question: str, response: str, latency: float):
        """Log a simple interaction with basic metrics"""
        try:
            # Prepare metrics
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "latency_seconds": latency,
                "response_length": len(response.split())
            }
            
            # Log to file
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(metrics) + '\n')
                
        except Exception as e:
            print(f"Failed to log evaluation: {str(e)}")
            return None
    
    def get_metrics_summary(self):
        """Get summary of recent metrics"""
        try:
            metrics = []
            with open(self.log_file, 'r') as f:
                for line in f:
                    try:
                        metrics.append(json.loads(line))
                    except:
                        continue
            
            if not metrics:
                return None
            
            # Calculate basic stats
            total = len(metrics)
            avg_latency = sum(m["latency_seconds"] for m in metrics) / total
            avg_length = sum(m["response_length"] for m in metrics) / total
            
            return {
                "total_interactions": total,
                "average_latency": round(avg_latency, 2),
                "average_response_length": round(avg_length, 2)
            }
            
        except Exception as e:
            print(f"Failed to get metrics summary: {str(e)}")
            return None
