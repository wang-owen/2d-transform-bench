#include "dft2.h"
#include "fft2.h"
#include "stb/stb_image.h"
#include "stb/stb_image_write.h"

#include <bit>
#include <iostream>
#include <random>

unsigned char *generate_data(int N) {
  auto *data = new unsigned char[N * N];
  std::mt19937 rng(1);
  std::uniform_real_distribution<float> dist(0.0f, 255.0f);

  for (int i = 0; i < N * N; ++i) {
    data[i] = dist(rng);
  }

  return data;
}

void copy_data(const unsigned char *src, unsigned char *dest, int N,
               int destStride) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      dest[i * destStride + j] = src[i * N + j];
    }
  }
}

[[nodiscard]]
bool is_equal(const unsigned char *data1, const unsigned char *data2, int M,
              int N, int stride1, int stride2, float tol = 1.0f) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      if (std::abs(data1[i * stride1 + j] - data2[i * stride2 + j]) > tol) {
        return false;
      }
    }
  }
  return true;
}

void print_data(std::ostream &out, const unsigned char *data, int M, int N) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      out << static_cast<float>(data[i * N + j]) << " \n"[j == N - 1];
    }
  }
}

int main() {
  const int N = 8;

  auto fftData = generate_data(N);
  fft2::transform(fftData, N, N);

  int dftM = std::bit_ceil(static_cast<unsigned int>(N));
  int dftN = std::bit_ceil(static_cast<unsigned int>(N));
  auto dftData = new unsigned char[dftM * dftN]();
  copy_data(fftData, dftData, N, dftN);
  dft2::transform(dftData, dftN, dftM);

  if (is_equal(fftData, dftData, N, N, N, dftN)) {
    std::cout << "SUCCESS\n";
  } else {
    std::cout << "FAILURE\n";

    std::cerr << "---Test 1---\n";
    print_data(std::cerr, fftData, N, N);
    std::cerr << "\n---Test 2---\n";
    print_data(std::cerr, dftData, N, N);
  }
}
