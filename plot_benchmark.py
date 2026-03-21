import matplotlib.pyplot as plt
import csv

Ns, dft_times, fft_iter_times, fft_recur_times = [], [], [], []

with open("timings.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        Ns.append(int(row["N"]))
        dft_times.append(float(row["DFT_time_ms"]))
        fft_iter_times.append(float(row["FFT_ITER_time_ms"]))
        fft_recur_times.append(float(row["FFT_RECUR_time_ms"]))

plt.plot(Ns, dft_times, "r-o", label="DFT")
plt.plot(Ns, fft_iter_times, "b-o", label="Iterative FFT")
plt.plot(Ns, fft_recur_times, "g-o", label="Recursive FFT")
plt.xlabel("Input size N")
plt.ylabel("Average runtime (ms)")
plt.title("Fourier algorithm performances")
plt.legend()
plt.yscale("log")
plt.show()
