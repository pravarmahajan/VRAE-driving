import sys
import glob

log_files = glob.glob("vrnn_50_0.2*")
print log_files

for fname in log_files:
    with open(fname, 'r') as f:
        print fname
        trip_accuracy_line = [l for l in f if l.startswith("Trip level prediction")]
        if len(trip_accuracy_line) > 0:
            print(trip_accuracy_line)
            trip_accuracy = float(trip_accuracy_line[0].split('=')[1].strip())
            print(trip_accuracy)
