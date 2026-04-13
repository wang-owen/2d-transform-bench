import sys
import matplotlib.pyplot as plt
import csv

if len(sys.argv) not in [2, 3]:
    print(f"usage: {sys.argv[0]} <input> [output]", file=sys.stderr)
    exit(1)

Ns, dft_times, fft_iter_times, fft_recur_times, dct_times = [], [], [], [], []

with open(sys.argv[1]) as f:
    reader = csv.DictReader(f)
    for row in reader:
        Ns.append(int(row["N"]))
        dft_times.append(float(row["DFT_time_ms"]))
        fft_iter_times.append(float(row["FFT_ITER_time_ms"]))
        fft_recur_times.append(float(row["FFT_RECUR_time_ms"]))
        dct_times.append(float(row["DCT_time_ms"]))

plt.plot(Ns, dft_times, label="DFT")
plt.plot(Ns, fft_iter_times, label="Iterative FFT")
plt.plot(Ns, fft_recur_times, label="Recursive FFT")
plt.plot(Ns, dct_times, label="DCT")
plt.xlabel("Input size N")
plt.ylabel("Average runtime (ms)")
plt.title("Transform algorithm performances")
plt.legend()
plt.yscale("log")

if len(sys.argv) == 3:
    plt.savefig(sys.argv[2])

plt.show()
