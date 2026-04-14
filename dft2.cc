#include "dft2.h"
#include "util.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <numbers>
#include <thread>

namespace dft2 {

namespace internal {

namespace {

void dft1_impl(std::vector<std::complex<float>> &data,
               std::vector<std::complex<float>> &scratch,
               std::vector<std::complex<float>> &table, int start, int N) {
  for (int i = 0; i < N; ++i) {
    scratch[i] = data[start + i];
  }

  int ki = 0;
  for (int k = start; k < start + N; ++k) {
    std::complex<float> F = 0;
    int idx = 0;
    for (int n = 0; n < N; ++n) {
      F += scratch[n] * table[idx];
      idx += ki;
      if (idx >= N) {
        idx -= N;
      }
    }
    data[k] = F;
    ++ki;
  }
}

void batch_dft1(std::vector<std::complex<float>> &data,
                std::vector<std::complex<float>> &table, int length, int start,
                int end, int stride) {
  std::vector<std::complex<float>> scratch(length);
  for (int i = start; i < end; i += stride) {
    dft1_impl(data, scratch, table, i, length);
  }
}

} // namespace

void dft2_separated(std::vector<std::complex<float>> &data, int M, int N,
                    Dir dir) {
  std::vector<std::complex<float>> scratch(std::max(M, N));

  std::vector<std::complex<float>> table(N);
  for (int k = 0; k < N; ++k) {
    table[k] = std::exp(std::complex<float>(0, -static_cast<int>(dir) * 2.0f *
                                                   k * std::numbers::pi / N));
  }

  for (int y = 0; y < M; ++y) {
    dft1_impl(data, scratch, table, y * N, N);
  }

  table.resize(M);
  for (int k = 0; k < M; ++k) {
    table[k] = std::exp(std::complex<float>(0, -static_cast<int>(dir) * 2.0f *
                                                   k * std::numbers::pi / M));
  }

  util::transpose_flattened(data, M, N);

  for (int x = 0; x < N; ++x) {
    dft1_impl(data, scratch, table, x * M, M);
  }

  util::transpose_flattened(data, N, M);

  if (dir == Dir::Forward) {
    const float scale = 1.0f / (M * N);
    for (auto &n : data) {
      n *= scale;
    }
  }
}

void dft2_separated_threaded(std::vector<std::complex<float>> &data, int M,
                             int N, Dir dir) {

  std::vector<std::complex<float>> table(N);
  for (int k = 0; k < N; ++k) {
    table[k] = std::exp(std::complex<float>(0, -static_cast<int>(dir) * 2.0f *
                                                   k * std::numbers::pi / N));
  }

  int num_threads = std::max(1u, std::thread::hardware_concurrency());
  int rows_per_thread = std::ceil(1.0 * M / num_threads);
  int cols_per_thread = std::ceil(1.0 * N / num_threads);

  // Row pass
  {
    std::vector<std::jthread> pool;

    int num_active = std::min(num_threads, M);
    for (int i = 0; i < num_active; ++i) {
      int start = i * rows_per_thread * N;
      int end = std::min((i + 1) * rows_per_thread, M) * N;

      pool.emplace_back(batch_dft1, std::ref(data), std::ref(table), N, start,
                        end, N);
    }
  }

  table.resize(M);
  for (int k = 0; k < M; ++k) {
    table[k] = std::exp(std::complex<float>(0, -static_cast<int>(dir) * 2.0f *
                                                   k * std::numbers::pi / M));
  }

  util::transpose_flattened(data, M, N);

  // Col pass
  {
    std::vector<std::jthread> pool;

    int num_active = std::min(num_threads, N);
    for (int i = 0; i < num_active; ++i) {
      int start = i * cols_per_thread * M;
      int end = std::min((i + 1) * cols_per_thread, N) * M;

      pool.emplace_back(batch_dft1, std::ref(data), std::ref(table), M, start,
                        end, M);
    }
  }

  util::transpose_flattened(data, N, M);

  if (dir == Dir::Forward) {
    const float scale = 1.0f / (M * N);
    for (auto &n : data) {
      n *= scale;
    }
  }
}

[[deprecated("use dft2_separated instead")]]
void dft2(std::vector<std::complex<float>> &data, int M, int N, Dir dir) {
  const int direction = static_cast<int>(dir);
  const float scale = direction == 1 ? 1.0f : 1.0f / (M * N);
  const std::complex<float> W_M = std::exp(
      std::complex<float>(0, -direction * 2.0f * std::numbers::pi / M));
  const std::complex<float> W_N = std::exp(
      std::complex<float>(0, -direction * 2.0f * std::numbers::pi / N));

  const auto input = data;

  std::complex<float> W_M_step = 1.0f;
  for (int k = 0; k < M; ++k) {
    std::complex<float> W_N_step = 1.0f;

    for (int l = 0; l < N; ++l) {
      std::complex<float> W_M_cur = 1.0f;
      std::complex<float> F = 0;

      for (int n = 0; n < M; ++n) {
        std::complex<float> W_N_cur = 1.0f;

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

void transform(unsigned char *data, int width, int height, float quality,
               bool threaded) {
  const int M = height;
  const int N = width;

  std::vector<std::complex<float>> img(M * N, 0);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      img[y * N + x] = static_cast<float>(data[y * width + x]) / 255.0f;
    }
  }

  if (threaded) {
    internal::dft2_separated_threaded(img, M, N, internal::Dir::Forward);
  } else {
    internal::dft2_separated(img, M, N, internal::Dir::Forward);
  }

  const auto dc = img[0];

  std::vector<float> mags(M * N);

  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      mags[i * N + j] = std::abs(img[i * N + j]);
    }
  }

  const int idx =
      std::min(static_cast<int>(std::floor((1.0f - quality) * mags.size())),
               static_cast<int>(mags.size()) - 1);
  std::nth_element(mags.begin(), mags.begin() + idx, mags.end());
  const float thresh = mags[idx];

  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      if (std::abs(img[i * N + j]) < thresh) {
        img[i * N + j] = 0;
      }
    }
  }

  img[0] = dc;

  if (threaded) {
    internal::dft2_separated_threaded(img, M, N, internal::Dir::Inverse);
  } else {
    internal::dft2_separated(img, M, N, internal::Dir::Inverse);
  }

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      float v = std::clamp(img[y * N + x].real(), 0.0f, 1.0f);
      data[y * width + x] = static_cast<unsigned char>(v * 255.0f);
    }
  }
}

} // namespace dft2
