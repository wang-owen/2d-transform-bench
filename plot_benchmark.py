import sys
import matplotlib.pyplot as plt
import csv

if len(sys.argv) not in [2, 3]:
    print(f"usage: {sys.argv[0]} <input> [output]", file=sys.stderr)
    exit(1)

SERIES = [
    (["DFT_time_ms"], "DFT", "#1a6faf"),
    (["DFT_T_time_ms"], "DFT (MT)", "#74b9e8"),
    (["FFT_ITER_time_ms"], "Iterative FFT", "#1a7a4a"),
    (["FFT_RECUR_time_ms"], "Recursive FFT", "#4cb87a"),
    (["FFT_T_time_ms"], "FFT (MT)", "#a8dfc0"),
    (["DCT_time_ms"], "DCT", "#af1a1a"),
    (["DCT_T_time_ms"], "DCT (MT)", "#e87474"),
]

plt.figure(figsize=(14, 9))

Ns = []
series_data = {label: [] for _, label, _ in SERIES}

with open(sys.argv[1]) as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames

    # Resolve which column name to use for each series
    resolved = []
    for aliases, label, color in SERIES:
        col = next((a for a in aliases if a in fieldnames), None)
        resolved.append((col, label, color))

    for row in reader:
        Ns.append(int(row["N"]))
        for col, label, _ in resolved:
            if col is not None:
                series_data[label].append(float(row[col]))

for col, label, color in resolved:
    if col is not None:
        plt.plot(Ns, series_data[label], label=label, color=color)

plt.xlabel("Input size N")
plt.ylabel("Average runtime (ms)")
plt.title("Transform algorithm performances")
plt.legend()
plt.xscale("log")
plt.yscale("log")
plt.xticks(Ns, Ns)
plt.grid(True, which="both", ls="--", alpha=0.4)

if len(sys.argv) == 3:
    plt.savefig(sys.argv[2])
