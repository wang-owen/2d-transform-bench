#include "fft2.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <numbers>
#include <utility>

namespace fft2 {

namespace internal {

namespace {

void bit_reversal_strided(std::vector<std::complex<double>> &a, int length,
                          int start, int stride) {
  int j = 0;
  for (int i = 1; i < length; i++) {
    int bit = length >> 1;
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

void fft_strided_recur_impl(std::vector<std::complex<double>> &data,
                            int direction, int start, int N, int stride,
                            double scale) {
  if (N <= 1) {
    return;
  }

  const auto WN =
      std::exp(std::complex<double>(0, -direction * 2 * std::numbers::pi / N));

  std::complex<double> w = 1;
  for (int n = 0; n < N / 2; ++n) {
    const int i1 = start + n * stride;
    const int i2 = start + (n + N / 2) * stride;

    auto a = data[i1];
    auto b = data[i2];

    data[i1] = scale * (a + b);
    data[i2] = scale * (a - b) * w;

    w *= WN;
  }

  fft_strided_recur_impl(data, direction, start, N / 2, stride, scale);
  fft_strided_recur_impl(data, direction, start + (N / 2) * stride, N / 2,
                         stride, scale);
}

} // namespace

void fft_strided_recur(std::vector<std::complex<double>> &data, int start,
                       int N, int stride, Dir dir) {
  const int direction = static_cast<int>(dir);
  const double scale = direction == 1 ? 1 : 0.5;

  fft_strided_recur_impl(data, direction, start, N, stride, scale);
  bit_reversal_strided(data, N, start, stride);
}

void fft_strided_iter(std::vector<std::complex<double>> &data, int start, int N,
                      int stride, Dir dir) {
  const int direction = static_cast<int>(dir);
  const double scale = direction == 1 ? 1 : 0.5;

  for (int size = N; size > 1; size >>= 1) {
    const auto WN = std::exp(
        std::complex<double>(0, -direction * 2 * std::numbers::pi / size));

    for (int block = 0; block < N / size; ++block) {
      int n1 = start + block * size * stride;
      std::complex<double> w = 1;
      for (int n = 0; n < (size >> 1); ++n) {
        const int i1 = n1 + n * stride;
        const int i2 = n1 + (n + (size >> 1)) * stride;

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

} // namespace internal

void transform(unsigned char *data, int width, int height, double ratio,
               internal::Method method) {
  const int M = std::bit_ceil(static_cast<unsigned int>(height));
  const int N = std::bit_ceil(static_cast<unsigned int>(width));

  std::vector<std::complex<double>> img(M * N, 0);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      img[y * N + x] = static_cast<double>(data[y * width + x]) / 255.0;
    }
  }

  auto fft_ptr = method == internal::Method::ITER ? internal::fft_strided_iter
                                                  : internal::fft_strided_recur;

  for (int y = 0; y < M; ++y) {
    fft_ptr(img, y * N, N, 1, internal::Dir::Forward);
  }
  for (int x = 0; x < N; ++x) {
    fft_ptr(img, x, M, N, internal::Dir::Forward);
  }

  std::vector<double> mags;
  mags.reserve(img.size());
  for (auto &v : img) {
    mags.push_back(std::abs(v));
  }

  const int idx =
      std::min(static_cast<int>(std::floor((1.0 - ratio) * mags.size())),
               static_cast<int>(mags.size()) - 1);
  std::nth_element(mags.begin(), mags.begin() + idx, mags.end());
  const double thresh = mags[idx];

  for (size_t i = 1; i < img.size(); ++i) {
    if (std::abs(img[i]) < thresh) {
      img[i] = 0;
    }
  }

  for (int y = 0; y < M; ++y) {
    fft_ptr(img, y * N, N, 1, internal::Dir::Inverse);
  }
  for (int x = 0; x < N; ++x) {
    fft_ptr(img, x, M, N, internal::Dir::Inverse);
  }

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      double v = std::clamp(img[y * N + x].real(), 0.0, 1.0);
      data[y * width + x] = static_cast<unsigned char>(v * 255.0);
    }
  }
}

} // namespace fft2
