#include "dft2.h"

#include <algorithm>
#include <numbers>

using namespace std::complex_literals;

namespace dft2 {

namespace internal {

void dft2(std::vector<std::complex<double>> &data, int N, int M) {
  const std::complex<double> Wn =
      std::exp(std::complex<double>(0, -2.0 * std::numbers::pi / N));
  const std::complex<double> Wm =
      std::exp(std::complex<double>(0, -2.0 * std::numbers::pi / M));

  const auto input = data;

  for (int k = 0; k < N; ++k) {
    for (int l = 0; l < M; ++l) {
      std::complex<double> F = 0;
      for (int n = 0; n < N; ++n) {
        const auto Wnk = std::pow(Wn, n * k);
        for (int j = 0; j < M; ++j) {
          F += input[n * M + j] * Wnk * std::pow(Wm, j * l);
        }
      }
      data[k * M + l] = F;
    }
  }
}

void idft2(std::vector<std::complex<double>> &data, int N, int M) {
  const std::complex<double> Wn =
      std::exp(std::complex<double>(0, 2.0 * std::numbers::pi / N));
  const std::complex<double> Wm =
      std::exp(std::complex<double>(0, 2.0 * std::numbers::pi / M));

  const auto input = data;

  for (int n = 0; n < N; ++n) {
    for (int j = 0; j < M; ++j) {
      std::complex<double> f = 0;
      for (int k = 0; k < N; ++k) {
        const auto Wnk = std::pow(Wn, n * k);
        for (int l = 0; l < M; ++l) {
          f += input[k * M + l] * Wnk * std::pow(Wm, j * l);
        }
      }
      data[n * M + j] = f / static_cast<double>(N * M);
    }
  }
}

} // namespace internal

void transform(unsigned char *data, int width, int height, int channels,
               double ratio) {

  const auto nextPowerOf2 = [](int v) {
    --v;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    ++v;
    return v;
  };

  const int N = nextPowerOf2(height);
  const int M = nextPowerOf2(width);

  std::vector<std::complex<double>> img(N * M, 0);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      img[y * M + x] = static_cast<double>(data[y * width + x]) / 255.0;
    }
  }

  internal::dft2(img, N, M);

  const auto dc = img[0];

  std::vector<double> mags(N * M);

  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < M; ++j) {
      mags[i * M + j] = std::abs(img[i * M + j]);
    }
  }

  const size_t idx =
      std::min(static_cast<size_t>(std::floor((1.0 - ratio) * mags.size())),
               mags.size() - 1);
  std::nth_element(mags.begin(), mags.begin() + idx, mags.end());
  const double thresh = mags[idx];

  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < M; ++j) {
      if (std::abs(img[i * M + j]) < thresh) {
        img[i * M + j] = 0;
      }
    }
  }

  img[0] = dc;

  internal::idft2(img, N, M);

  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < width; ++x) {
      double v = std::clamp(img[y * M + x].real(), 0.0, 1.0);
      data[y * width + x] = static_cast<unsigned char>(v * 255.0);
    }
  }
}

} // namespace dft2