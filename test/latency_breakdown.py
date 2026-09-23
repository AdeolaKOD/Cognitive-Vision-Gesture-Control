import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
import os
import subprocess
import sys

def plot_latency_breakdown():
    print("Launching main.py to collect data. Move your hands around to test the pipeline latency.")
    print("Close the HandGestureControl window when you are finished.")
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    main_script = os.path.join(project_root, 'src', 'main.py')
    
    # Start the HandGestureControl application and wait for the user to close it
    process = subprocess.Popen([sys.executable, main_script], cwd=project_root)
    process.wait()

    layers = ['Acquisition', 'Pre-processing', 'Inference', 'Logic', 'Actuation']
    
    csv_path = os.path.join(project_root, 'testResult', 'latency_data.csv')
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Ensure main.py ran successfully.")
        return

    # To calculate the mean latency of each layer
    layer_totals = [0.0] * 5
    row_count = 0

    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader, None) # Skip header
        for row in reader:
            if len(row) == 5:
                try:
                    for i in range(5):
                        layer_totals[i] += float(row[i])
                    row_count += 1
                except ValueError:
                    pass
                    
    if row_count == 0:
        print("No valid data found in latency_data.csv.")
        return
        
    latencies = [total / row_count for total in layer_totals]
    
    # Generate the bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(layers, latencies, color='skyblue', edgecolor='black', width=0.6)
    
    # Formatting
    plt.title('Real Performance Breakdown per Pipeline Layer')
    plt.ylabel('Average Latency (ms)')
    # Add a bit of padding to the top of the max bar
    plt.ylim(0, max(latencies) * 1.2 if max(latencies) > 0 else 16)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.xticks(rotation=15)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.1f}ms', va='bottom', ha='center')
    
    output_dir = os.path.join(project_root, 'testResult')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'latency_breakdown.pdf')
    
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    print(f'Plot saved as {output_path}')

if __name__ == '__main__':
    plot_latency_breakdown()
