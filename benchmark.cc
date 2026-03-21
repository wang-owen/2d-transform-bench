#include "dft2.h"
#include "fft2.h"

#include <algorithm>
#include <chrono>
#include <complex>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

enum class Transform { DFT, FFT_ITER, FFT_RECUR };

std::vector<std::complex<double>> generate_data(int N) {
  std::vector<std::complex<double>> data(N * N);
  std::mt19937 rng(1);
  std::uniform_real_distribution<double> dist(0.0, 1.0);

  for (auto &x : data) {
    x = {dist(rng), dist(rng)};
  }

  return data;
}

double average_runtime_ms(std::vector<std::complex<double>> data, int N,
                          Transform transform, int runs) {
  double total = 0.0;

  for (int i = 0; i < runs; ++i) {
    auto start = std::chrono::high_resolution_clock::now();

    switch (transform) {
    case Transform::DFT: {
      dft2::internal::dft2(data, N, N);
      break;
    }
    case Transform::FFT_ITER: {
      for (size_t y = 0; y < N; ++y) {
        fft2::internal::fft_strided_iter(data, fft2::internal::FFTDir::Forward,
                                         y * N, N, 1);
      }
      for (size_t x = 0; x < N; ++x) {
        fft2::internal::fft_strided_iter(data, fft2::internal::FFTDir::Forward,
                                         x, N, N);
      }
      break;
    }
    case Transform::FFT_RECUR: {
      for (size_t y = 0; y < N; ++y) {
        fft2::internal::fft_strided_recur(data, fft2::internal::FFTDir::Forward,
                                          y * N, N, 1);
      }
      for (size_t x = 0; x < N; ++x) {
        fft2::internal::fft_strided_recur(data, fft2::internal::FFTDir::Forward,
                                          x, N, N);
      }
      break;
    }
    }

    auto end = std::chrono::high_resolution_clock::now();
    total += std::chrono::duration<double, std::milli>(end - start).count();
  }

  return total / runs;
}

int main() {
  std::ofstream csv("timings.csv");
  csv << "N,DFT_time_ms,FFT_ITER_time_ms,FFT_RECUR_time_ms\n";

  std::vector<size_t> sizes = {64, 128, 256, 512, 1024};

  for (auto N : sizes) {
    auto data = generate_data(N);

    double dft_time =
        average_runtime_ms(data, N, Transform::DFT,
                           std::max(1, 50 * (64 / (int)N) * (64 / (int)N)));
    double fft_iter_time = average_runtime_ms(data, N, Transform::FFT_ITER, 50);
    double fft_recur_time =
        average_runtime_ms(data, N, Transform::FFT_RECUR, 50);

    csv << N << "," << dft_time << "," << fft_iter_time << "," << fft_recur_time
        << "\n";
    std::cout << "N=" << N << ": DFT=" << dft_time
              << "ms, FFT_ITER=" << fft_iter_time
              << "ms, FFT_RECUR=" << fft_recur_time << "ms\n";
  }
}
