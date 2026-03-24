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

double avg_runtime(std::vector<std::complex<double>> data, int N,
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
      for (int y = 0; y < N; ++y) {
        fft2::internal::fft_strided_iter(data, y * N, N, 1);
      }
      for (int x = 0; x < N; ++x) {
        fft2::internal::fft_strided_iter(data, x, N, N);
      }
      break;
    }
    case Transform::FFT_RECUR: {
      for (int y = 0; y < N; ++y) {
        fft2::internal::fft_strided_recur(data, y * N, N, 1);
      }
      for (int x = 0; x < N; ++x) {
        fft2::internal::fft_strided_recur(data, x, N, N);
      }
      break;
    }
    default: {
      throw std::invalid_argument("Expected std::vector<std::complex<double>>");
    }
    }

    auto end = std::chrono::high_resolution_clock::now();
    total += std::chrono::duration<double, std::milli>(end - start).count();
  }

  return total / runs;
}

double avg_runtime(std::vector<double> data, int N, Transform transform,
                   int runs) {
  double total = 0.0;

  for (int i = 0; i < runs; ++i) {
    auto start = std::chrono::high_resolution_clock::now();

    switch (transform) {
    case Transform::DCT: {
      for (int y = 0; y < N; ++y) {
        dct2::internal::dct1(data, y * N, N, 1);
      }
      for (int x = 0; x < N; ++x) {
        dct2::internal::dct1(data, x, N, N);
      }
      break;
    }
    default: {
      throw std::invalid_argument("Expected std::vector<double>");
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

  std::vector<int> sizes = {64, 128, 256, 512, 1024};

  for (int N : sizes) {
    auto complex_data = generate_data_complex(N);
    auto real_data = generate_data_real(N);

    int runs = std::max(1, 50 * (64 / N) * (64 / N));

    double dft_time = avg_runtime(complex_data, N, Transform::DFT, runs);
    double fft_iter_time =
        avg_runtime(complex_data, N, Transform::FFT_ITER, 50);
    double fft_recur_time =
        avg_runtime(complex_data, N, Transform::FFT_RECUR, 50);
    double dct_time = avg_runtime(real_data, N, Transform::DCT, runs);

    csv << N << "," << dft_time << "," << fft_iter_time << "," << fft_recur_time
        << "\n";
    std::cout << "N=" << N << ": DFT=" << dft_time
              << "ms, FFT_ITER=" << fft_iter_time
              << "ms, FFT_RECUR=" << fft_recur_time << "ms\n";
  }
}
