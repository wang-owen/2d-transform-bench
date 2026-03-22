#include "dft2.h"

#include <algorithm>
#include <numbers>

using namespace std::complex_literals;

namespace dft2 {

namespace internal {

void dft2(std::vector<std::complex<double>> &data, int M, int N) {
  const std::complex<double> W_M =
      std::exp(std::complex<double>(0, -2.0 * std::numbers::pi / M));
  const std::complex<double> W_N =
      std::exp(std::complex<double>(0, -2.0 * std::numbers::pi / N));

  const auto input = data;

  std::complex<double> W_M_step = 1.0;
  for (int k = 0; k < M; ++k) {
    std::complex<double> W_N_step = 1.0;

    for (int l = 0; l < N; ++l) {
      std::complex<double> W_M_cur = 1.0;
      std::complex<double> F = 0;

      for (int n = 0; n < M; ++n) {
        std::complex<double> W_N_cur = 1.0;

        for (int j = 0; j < N; ++j) {
          F += input[n * N + j] * W_N_cur * W_M_cur;
          W_N_cur *= W_N_step;
        }
        W_M_cur *= W_M_step;
      }
      W_N_step *= W_N;
      data[k * N + l] = F;
    }
    W_M_step *= W_M;
  }
}

void idft2(std::vector<std::complex<double>> &data, int M, int N) {
  const std::complex<double> W_M =
      std::exp(std::complex<double>(0, 2.0 * std::numbers::pi / M));
  const std::complex<double> W_N =
      std::exp(std::complex<double>(0, 2.0 * std::numbers::pi / N));

  const auto input = data;

  std::complex<double> W_M_step = 1.0;
  for (int n = 0; n < M; ++n) {
    std::complex<double> W_N_step = 1.0;

    for (int j = 0; j < N; ++j) {
      std::complex<double> W_M_cur = 1.0;
      std::complex<double> f = 0;

      for (int k = 0; k < M; ++k) {
        std::complex<double> W_N_cur = 1.0;

        for (int l = 0; l < N; ++l) {
          f += input[k * N + l] * W_M_cur * W_N_cur;
          W_N_cur *= W_N_step;
        }
        W_M_cur *= W_M_step;
      }
      W_N_step *= W_N;
      data[n * N + j] = f / static_cast<double>(M * N);
    }
    W_M_step *= W_M;
  }
}

} // namespace internal

void transform(unsigned char *data, int width, int height, int channels,
               double ratio) {
  const int M = height;
  const int N = width;

  std::vector<std::complex<double>> img(M * N, 0);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      img[y * N + x] = static_cast<double>(data[y * width + x]) / 255.0;
    }
  }

  internal::dft2(img, M, N);

  const auto dc = img[0];

  std::vector<double> mags(M * N);

  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      mags[i * N + j] = std::abs(img[i * N + j]);
    }
  }

  const size_t idx =
      std::min(static_cast<size_t>(std::floor((1.0 - ratio) * mags.size())),
               mags.size() - 1);
  std::nth_element(mags.begin(), mags.begin() + idx, mags.end());
  const double thresh = mags[idx];

  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      if (std::abs(img[i * N + j]) < thresh) {
        img[i * N + j] = 0;
      }
    }
  }

  img[0] = dc;

  internal::idft2(img, M, N);

  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < width; ++x) {
      double v = std::clamp(img[y * N + x].real(), 0.0, 1.0);
      data[y * width + x] = static_cast<unsigned char>(v * 255.0);
    }
  }
}

} // namespace dft2
