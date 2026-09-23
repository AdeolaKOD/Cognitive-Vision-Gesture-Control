import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
import os
import subprocess
import sys

def plot_transfer_function():
    print("Launching main.py to collect data. Please perform the volume clutch gesture.")
    print("Close the HandGestureControl window when you are finished.")
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    main_script = os.path.join(project_root, 'src', 'main.py')
    
    # Start the HandGestureControl application and wait for the user to close it
    process = subprocess.Popen([sys.executable, main_script], cwd=project_root)
    process.wait()
    
    # Read the data
    theoretical_pixels = np.linspace(0, 250, 500)
    theoretical_scalar = np.interp(theoretical_pixels, [45, 205], [0, 100])
    
    recorded_pixels = []
    recorded_scalar = []
    
    csv_path = os.path.join(project_root, 'testResult', 'transfer_data.csv')
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Ensure main.py ran successfully.")
        return
            
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader, None) # Skip header
        for row in reader:
            if len(row) == 2:
                try:
                    recorded_pixels.append(float(row[0]))
                    recorded_scalar.append(float(row[1]))
                except ValueError:
                    pass
    
    if not recorded_pixels:
        print("No valid data found in transfer_data.csv. Please generate some data by performing the volume gesture.")
        return
        
    # Generate the plot
    plt.figure(figsize=(10, 6))
    plt.scatter(recorded_pixels, recorded_scalar, color='green', alpha=0.5, label='Recorded Mapping')
    plt.plot(theoretical_pixels, theoretical_scalar, color='navy', linestyle='--', label='Theoretical Vol Transfer')
    
    # Formatting
    plt.title('Transfer Function: Euclidean Distance to System Actuation')
    plt.xlabel('Raw System Output (Pixels)')
    plt.ylabel('Calculated Scalar (%)')
    plt.xlim(0, 250)
    plt.ylim(-5, 105)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper left')
    
    output_dir = os.path.join(project_root, 'testResult')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'transfer_function.png')
    
    plt.savefig(output_path)
    print(f'Plot saved as {output_path}')

if __name__ == '__main__':
    plot_transfer_function()
