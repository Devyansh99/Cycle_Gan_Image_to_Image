"""
Metrics tracking for Writer-Aware CycleGAN.
Logs metrics to checkpoints folder.
"""
import json
from pathlib import Path


class MetricsLogger:
    """Track and save training metrics"""
    
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.metrics_file = self.checkpoint_dir / "metrics.json"
        self.metrics = {"epochs": [], "losses": {}}
        
    def log_epoch(self, epoch, losses):
        """Log metrics for an epoch"""
        self.metrics["epochs"].append(epoch)
        for key, value in losses.items():
            if key not in self.metrics["losses"]:
                self.metrics["losses"][key] = []
            self.metrics["losses"][key].append(float(value))
        
        # Save to file
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def print_summary(self, epoch):
        """Print epoch summary"""
        if not self.metrics["losses"]:
            return
        
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch} METRICS SUMMARY:")
        print(f"{'='*60}")
        
        for loss_name, loss_values in self.metrics["losses"].items():
            if loss_values:
                latest = loss_values[-1]
                avg = sum(loss_values) / len(loss_values)
                print(f"  {loss_name:15s}: Latest={latest:8.4f}  Avg={avg:8.4f}")
        
        print(f"{'='*60}\n")
