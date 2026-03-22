#include "dft2.h"
#include "fft2.h"
#include "stb/stb_image.h"
#include "stb/stb_image_write.h"

#include <bit>
#include <format>
#include <iostream>
#include <ostream>
#include <string>

constexpr int GRAYSCALE = 1;
constexpr double DEFAULT_RATIO = 0.1;

void copyData(const unsigned char *src, unsigned char *dest, int M, int N,
              int destStride) {
  for (int i = 0; i < M; ++i) {
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

int main(int argc, char *argv[]) {
  const std::string usage = std::format("usage: {} <input path>\n", argv[0]);

  if (argc != 2) {
    std::cerr << usage;
    exit(1);
  }

  double ratio = DEFAULT_RATIO;

  int width, height, channels;
  unsigned char *data =
      stbi_load(argv[1], &width, &height, &channels, GRAYSCALE);

  if (data == nullptr) {
    std::cerr << "Failed to open image at " << argv[1] << '\n';
    exit(1);
  }

  auto fftData = new unsigned char[height * width];
  copyData(data, fftData, height, width, width);
  fft2::transform(fftData, width, height, GRAYSCALE, ratio);

  int dftM = std::bit_ceil(static_cast<unsigned int>(height));
  int dftN = std::bit_ceil(static_cast<unsigned int>(width));
  auto dftData = new unsigned char[dftM * dftN]();
  copyData(data, dftData, height, width, dftN);
  dft2::transform(dftData, dftN, dftM, GRAYSCALE, ratio);

  if (checkEqual(fftData, dftData, height, width, width, dftN)) {
    std::cout << "SUCCESS\n";
  } else {
    std::cout << "FAILURE\n";

    std::cerr << "---Test 1---\n";
    printData(std::cerr, fftData, height, width);
    std::cerr << "\n---Test 2---\n";
    printData(std::cerr, dftData, height, width);

    stbi_write_png("fft2_equality.png", width, height, GRAYSCALE, fftData,
                   width);
    stbi_write_png("dft2_equality.png", width, height, GRAYSCALE, dftData,
                   width);
  }

  stbi_image_free(fftData);
  stbi_image_free(dftData);
  stbi_image_free(data);
}
