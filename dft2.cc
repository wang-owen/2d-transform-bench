#include "dft2.h"

#include <algorithm>
#include <numbers>

namespace dft2 {

namespace internal {

namespace {

void dft_strided_impl(std::vector<std::complex<double>> &data,
                      std::vector<std::complex<double>> &scratch, int start,
                      int N, int stride, Dir dir) {
  const int direction = static_cast<int>(dir);

  for (int i = 0; i < N; ++i) {
    scratch[i] = data[start + i * stride];
  }

  std::vector<std::complex<double>> table(N);
  for (int k = 0; k < N; ++k) {
    table[k] = std::exp(
        std::complex<double>(0, -direction * 2.0 * k * std::numbers::pi / N));
  }

  for (int k = start; k < start + N * stride; k += stride) {
    std::complex<double> F = 0;
    for (int n = 0; n < N; ++n) {
      F += scratch[n] * table[(((k - start) / stride) * n) % N];
    }
    data[k] = F;
  }
}

} // namespace

void dft2_strided(std::vector<std::complex<double>> &data, int M, int N,
                  Dir dir) {
  std::vector<std::complex<double>> scratch(std::max(M, N));

  for (int y = 0; y < M; ++y) {
    dft_strided_impl(data, scratch, y * N, N, 1, dir);
  }
  for (int x = 0; x < N; ++x) {
    dft_strided_impl(data, scratch, x, M, N, dir);
  }

  if (dir == Dir::Forward) {
    const double scale = 1.0 / (M * N);
    for (auto &n : data) {
      n *= scale;
    }
  }
}

[[deprecated("use dft2_strided instead")]]
void dft2(std::vector<std::complex<double>> &data, int M, int N, Dir dir) {
  const int direction = static_cast<int>(dir);
  const double scale = direction == 1 ? 1.0 : 1.0 / (M * N);
  const std::complex<double> W_M = std::exp(
      std::complex<double>(0, -direction * 2.0 * std::numbers::pi / M));
  const std::complex<double> W_N = std::exp(
      std::complex<double>(0, -direction * 2.0 * std::numbers::pi / N));

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
      data[k * N + l] = scale * F;
    }
    W_M_step *= W_M;
  }
}

} // namespace internal

void transform(unsigned char *data, int width, int height, double ratio) {
  const int M = height;
  const int N = width;

  std::vector<std::complex<double>> img(M * N, 0);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      img[y * N + x] = static_cast<double>(data[y * width + x]) / 255.0;
    }
  }

  internal::dft2_strided(img, M, N, internal::Dir::Forward);

  const auto dc = img[0];

  std::vector<double> mags(M * N);

  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      mags[i * N + j] = std::abs(img[i * N + j]);
    }
  }

  const int idx =
      std::min(static_cast<int>(std::floor((1.0 - ratio) * mags.size())),
               static_cast<int>(mags.size()) - 1);
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

  internal::dft2_strided(img, M, N, internal::Dir::Inverse);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      double v = std::clamp(img[y * N + x].real(), 0.0, 1.0);
      data[y * width + x] = static_cast<unsigned char>(v * 255.0);
    }
  }
}

} // namespace dft2
