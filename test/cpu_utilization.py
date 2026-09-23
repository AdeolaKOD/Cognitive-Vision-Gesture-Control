import psutil
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import subprocess
import sys
import os

def plot_cpu_utilization(duration_seconds=60, sample_interval=0.5):
    print(f"Launching main.py and collecting CPU data for {duration_seconds} seconds... Please wait.")
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    main_script = os.path.join(project_root, 'src', 'main.py')
    
    # Start the HandGestureControl application
    process = subprocess.Popen([sys.executable, main_script], cwd=project_root)
    
    timestamps = []
    cpu_usage = []
    start_time = time.time()
    
    # Collect real-time CPU data
    while (time.time() - start_time) < duration_seconds:
        current_time = time.time() - start_time
        
        # Use system-wide cpu_percent (like the original script) and block for sample_interval
        val = psutil.cpu_percent(interval=sample_interval)
        
        # Append to both arrays AFTER the blocking call so their lengths always match
        timestamps.append(current_time)
        cpu_usage.append(val)
        
        # Check if the process was closed early by the user
        if process.poll() is not None:
            print("HandGestureControl process terminated prematurely.")
            break
            
    # Terminate the process after the test if it's still running
    if process.poll() is None:
        process.terminate()
        
    mean_cpu = np.mean(cpu_usage) if cpu_usage else 0
    
    # Generate the plot
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, cpu_usage, color='red', label='CPU Usage (%)')
    plt.axhline(y=mean_cpu, color='blue', linestyle='--', label=f'Mean: {mean_cpu:.1f}%')
    
    # Formatting
    plt.title('Real CPU Utilization During Operation')
    plt.xlabel('Time (Seconds)')
    plt.ylabel('CPU Usage (%)')
    plt.ylim(0, 100)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper right')
    
    output_dir = os.path.join(project_root, 'testResult')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'cpu_utilization.png')
    
    plt.savefig(output_path)
    print(f'Plot saved as {output_path}')

if __name__ == '__main__':
    plot_cpu_utilization(duration_seconds=60)
