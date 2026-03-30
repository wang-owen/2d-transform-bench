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
  std::uniform_real_distribution<double> dist(0.0, 1.0);

  for (int i = 0; i < N * N; ++i) {
    data[i] = dist(rng);
  }

  return data;
}

void copyData(const unsigned char *src, unsigned char *dest, int N,
              int destStride) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      dest[i * destStride + j] = src[i * N + j];
    }
  }
}

bool checkEqual(const unsigned char *data1, const unsigned char *data2, int M,
                int N, int stride1, int stride2) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      if (data1[i * stride1 + j] != data2[i * stride2 + j]) {
        return false;
      }
    }
  }
  return true;
}

void printData(std::ostream &out, const unsigned char *data, int M, int N) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      out << static_cast<double>(data[i * N + j]) << " \n"[j == N - 1];
    }
  }
}

int main() {
  const int N = 128;

  auto fftData = generate_data(N);
  fft2::transform(fftData, N, N);

  int dftM = std::bit_ceil(static_cast<unsigned int>(N));
  int dftN = std::bit_ceil(static_cast<unsigned int>(N));
  auto dftData = new unsigned char[dftM * dftN]();
  copyData(fftData, dftData, N, dftN);
  dft2::transform(dftData, dftN, dftM);

  if (checkEqual(fftData, dftData, N, N, N, dftN)) {
    std::cout << "SUCCESS\n";
  } else {
    std::cout << "FAILURE\n";

    std::cerr << "---Test 1---\n";
    printData(std::cerr, fftData, N, N);
    std::cerr << "\n---Test 2---\n";
    printData(std::cerr, dftData, N, N);
  }
}
