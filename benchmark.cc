#include "dct2.h"
#include "dft2.h"
#include "fft2.h"

#include <algorithm>
#include <chrono>
#include <complex>
#include <csignal>
#include <cstdlib>
#include <format>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

enum class Transform {
  DFT,
  DFT_T,
  FFT_ITER,
  FFT_RECUR,
  FFT_T,
  DCT,
  DCT_T,
};

std::vector<float> generate_data_real(int N) {
  std::vector<float> data(N * N);
  std::mt19937 rng(1);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);

  for (auto &x : data) {
    x = dist(rng);
  }

  return data;
}

std::vector<std::complex<float>> generate_data_complex(int N) {
  std::vector<std::complex<float>> data(N * N);
  std::mt19937 rng(1);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);

  for (auto &x : data) {
    x = {dist(rng), dist(rng)};
  }

  return data;
}

float avg_runtime(std::vector<std::complex<float>> data, int N,
                  Transform transform, int runs) {
  float total = 0.0f;

  for (int i = 0; i < runs; ++i) {
    auto start = std::chrono::high_resolution_clock::now();

    switch (transform) {
    case Transform::DFT: {
      dft2::internal::dft2_separated(data, N, N);
      break;
    }
    case Transform::DFT_T: {
      dft2::internal::dft2_separated_threaded(data, N, N);
      break;
    }
    case Transform::FFT_ITER: {
      fft2::internal::fft_iter(data, N, N);
      break;
    }
    case Transform::FFT_RECUR: {
      fft2::internal::fft_recur(data, N, N);
      break;
    }
    case Transform::FFT_T: {
      fft2::internal::fft_threaded(data, N, N);
      break;
    }
    default: {
      throw std::invalid_argument("Expected std::vector<std::complex<float>>");
    }
    }

    auto end = std::chrono::high_resolution_clock::now();
    total += std::chrono::duration<float, std::milli>(end - start).count();
  }

  return total / runs;
}

float avg_runtime(std::vector<float> data, int N, Transform transform,
                  int runs) {
  float total = 0.0f;

  for (int i = 0; i < runs; ++i) {
    auto start = std::chrono::high_resolution_clock::now();

    switch (transform) {
    case Transform::DCT: {
      dct2::internal::dct2(data, N, N);
      break;
    }
    case Transform::DCT_T: {
      dct2::internal::dct2_threaded(data, N, N);
      break;
    }
    default: {
      throw std::invalid_argument("Expected std::vector<float>");
    }
    }

    auto end = std::chrono::high_resolution_clock::now();
    total += std::chrono::duration<float, std::milli>(end - start).count();
  }

  return total / runs;
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::cerr << std::format("usage: {} <output>\n", argv[0]);
    exit(1);
  }

  std::ofstream csv(argv[1]);
  csv << "N,DFT_time_ms,DFT_T_time_ms,FFT_ITER_time_ms,FFT_RECUR_time_ms,FFT_T_"
         "time_ms,DCT_time_ms,DCT_T_time_ms\n";

  std::vector<int> sizes = {64, 128, 256, 512, 1024, 2048};

  for (int N : sizes) {
    auto complex_data = generate_data_complex(N);
    auto real_data = generate_data_real(N);

    int dft_dct_runs = std::max(1, static_cast<int>(50.0f * pow(64.0f / N, 4)));
    int fft_runs =
        std::max(1, static_cast<int>(50.0f * (64.0f * 64.0f * log(64.0f)) /
                                     (N * N * log(N))));

    float dft_time = avg_runtime(complex_data, N, Transform::DFT, dft_dct_runs);
    float dft_t_time =
        avg_runtime(complex_data, N, Transform::DFT_T, dft_dct_runs);
    float fft_iter_time =
        avg_runtime(complex_data, N, Transform::FFT_ITER, fft_runs);
    float fft_recur_time =
        avg_runtime(complex_data, N, Transform::FFT_RECUR, fft_runs);
    float fft_t_time = avg_runtime(complex_data, N, Transform::FFT_T, fft_runs);
    float dct_time = avg_runtime(real_data, N, Transform::DCT, dft_dct_runs);
    float dct_t_time =
        avg_runtime(real_data, N, Transform::DCT_T, dft_dct_runs);

    csv << N << "," << dft_time << ',' << dft_t_time << ',' << fft_iter_time
        << ',' << fft_recur_time << ',' << fft_t_time << ',' << dct_time << ','
        << dct_t_time << '\n';
    std::cout << "N=" << N << ": DFT=" << dft_time << "ms, DFT_T=" << dft_t_time
              << "ms, FFT_ITER=" << fft_iter_time
              << "ms, FFT_RECUR=" << fft_recur_time
              << "ms, FFT_T=" << fft_t_time << "ms, DCT=" << dct_time
              << "ms, DCT_T=" << dct_t_time << "ms\n";
  }
}
