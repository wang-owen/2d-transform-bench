#include "fft2.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <numbers>
#include <utility>

namespace fft2 {

namespace internal {

void bit_reversal_strided(std::vector<std::complex<double>> &a, int length,
                          int start, int stride) {
  size_t j = 0;
  for (size_t i = 1; i < length; i++) {
    size_t bit = length >> 1;
    while (j & bit) {
      j ^= bit;
      bit >>= 1;
    }
    j |= bit;

    if (i < j) {
      std::swap(a[start + i * stride], a[start + j * stride]);
    }
  }
}

void fft_strided_iter(std::vector<std::complex<double>> &data, FFTDir dir,
                      size_t start, size_t N, size_t stride) {
  const int direction = static_cast<int>(dir);
  const double scale = direction == 1 ? 1 : 0.5;

  for (size_t size = N; size > 1; size >>= 1) {
    const auto WN = std::exp(
        std::complex<double>(0, -direction * 2 * std::numbers::pi / size));

    for (size_t block = 0; block < N / size; ++block) {
      size_t n1 = start + block * size * stride;
      std::complex<double> w = 1;
      for (size_t n = 0; n < (size >> 1); ++n) {
        const size_t i1 = n1 + n * stride;
        const size_t i2 = n1 + (n + (size >> 1)) * stride;

        auto a = data[i1];
        auto b = data[i2];

        data[i1] = scale * (a + b);
        data[i2] = scale * (a - b) * w;

        w *= WN;
      }
    }
  }

  bit_reversal_strided(data, N, start, stride);
}

void fft_strided_recur_(std::vector<std::complex<double>> &data, int direction,
                        size_t start, size_t N, size_t stride, double scale) {
  if (N <= 1)
    return;

  const auto WN =
      std::exp(std::complex<double>(0, -direction * 2 * std::numbers::pi / N));

  std::complex<double> w = 1;
  for (size_t n = 0; n < N / 2; ++n) {
    const size_t i1 = start + n * stride;
    const size_t i2 = start + (n + N / 2) * stride;

    auto a = data[i1];
    auto b = data[i2];

    data[i1] = scale * (a + b);
    data[i2] = scale * (a - b) * w;

    w *= WN;
  }

  fft_strided_recur_(data, direction, start, N / 2, stride, scale);
  fft_strided_recur_(data, direction, start + (N / 2) * stride, N / 2, stride,
                     scale);
}

void fft_strided_recur(std::vector<std::complex<double>> &data, FFTDir dir,
                       size_t start, size_t N, size_t stride) {
  const int direction = static_cast<int>(dir);
  const double scale = direction == 1 ? 1 : 0.5;

  fft_strided_recur_(data, direction, start, N, stride, scale);
  bit_reversal_strided(data, N, start, stride);
}

} // namespace internal

void transform(unsigned char *data, int width, int height, int channels,
               double ratio, bool recur) {
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
  for (size_t y = 0; y < height; ++y)
    for (size_t x = 0; x < width; ++x)
      img[y * M + x] = static_cast<double>(data[y * width + x]) / 255.0;

  std::function<void(std::vector<std::complex<double>> &, internal::FFTDir,
                     size_t, size_t, size_t)>
      fft_ptr =
          recur ? internal::fft_strided_recur : internal::fft_strided_iter;

  for (size_t y = 0; y < N; ++y)
    fft_ptr(img, internal::FFTDir::Forward, y * M, M, 1);
  for (size_t x = 0; x < M; ++x)
    fft_ptr(img, internal::FFTDir::Forward, x, N, M);

  std::vector<double> mags;
  mags.reserve(img.size());
  for (auto &v : img) {
    mags.push_back(std::abs(v));
  }

  const size_t idx =
      std::min(static_cast<size_t>(std::floor((1.0 - ratio) * mags.size())),
               mags.size() - 1);
  std::nth_element(mags.begin(), mags.begin() + idx, mags.end());
  const double thresh = mags[idx];

  for (size_t i = 1; i < img.size(); ++i)
    if (std::abs(img[i]) < thresh)
      img[i] = 0;

  for (size_t y = 0; y < N; ++y)
    fft_ptr(img, internal::FFTDir::Inverse, y * M, M, 1);
  for (size_t x = 0; x < M; ++x)
    fft_ptr(img, internal::FFTDir::Inverse, x, N, M);

  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < width; ++x) {
      double v = std::clamp(img[y * M + x].real(), 0.0, 1.0);
      data[y * width + x] = static_cast<unsigned char>(v * 255.0);
    }
  }
}

} // namespace fft2
