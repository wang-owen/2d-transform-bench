#include "fft2.h"
#include "util.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <numbers>
#include <utility>

namespace fft2 {

namespace internal {

namespace {

struct TwiddleCache {
  int max_size = 1;
  std::vector<std::vector<std::complex<float>>> forward;
  std::vector<std::vector<std::complex<float>>> inverse;
};

TwiddleCache &twiddle_cache() {
  static TwiddleCache cache;
  return cache;
}

int log2_pow2(int n) { return std::countr_zero(static_cast<unsigned int>(n)); }

void ensure_twiddles(int max_size) {
  auto &cache = twiddle_cache();
  if (max_size <= cache.max_size) {
    return;
  }

  const int new_max = std::bit_ceil(static_cast<unsigned int>(max_size));
  const int max_log2 = log2_pow2(new_max);
  cache.forward.resize(max_log2 + 1);
  cache.inverse.resize(max_log2 + 1);

  for (int size = 2; size <= new_max; size <<= 1) {
    const int lg = log2_pow2(size);
    if (!cache.forward[lg].empty()) {
      continue;
    }

    cache.forward[lg].resize(size >> 1);
    cache.inverse[lg].resize(size >> 1);
    for (int k = 0; k < (size >> 1); ++k) {
      const float angle =
          2.0f * std::numbers::pi_v<float> * static_cast<float>(k) / size;
      cache.forward[lg][k] = {std::cos(-angle), std::sin(-angle)};
      cache.inverse[lg][k] = {std::cos(angle), std::sin(angle)};
    }
  }

  cache.max_size = new_max;
}

const std::vector<std::complex<float>> &twiddles_for_size(int size, Dir dir) {
  const auto &cache = twiddle_cache();
  const int lg = log2_pow2(size);
  return dir == Dir::Forward ? cache.forward[lg] : cache.inverse[lg];
}

void bit_reversal(std::vector<std::complex<float>> &a, int length, int start) {
  int j = 0;
  for (int i = 1; i < length; i++) {
    int bit = length >> 1;
    while (j & bit) {
      j ^= bit;
      bit >>= 1;
    }
    j |= bit;

    if (i < j) {
      std::swap(a[start + i], a[start + j]);
    }
  }
}

void fft_recur_impl(std::vector<std::complex<float>> &data, int direction,
                    int start, int N, Dir dir, float scale) {
  if (N <= 1) {
    return;
  }

  const auto &twiddles = twiddles_for_size(N, dir);
  for (int n = 0; n < N / 2; ++n) {
    const int i1 = start + n;
    const int i2 = start + (n + N / 2);

    auto a = data[i1];
    auto b = data[i2];

    data[i1] = scale * (a + b);
    data[i2] = scale * (a - b) * twiddles[n];
  }

  fft_recur_impl(data, direction, start, N / 2, dir, scale);
  fft_recur_impl(data, direction, start + (N / 2), N / 2, dir, scale);
}

void fft_iter_impl(std::vector<std::complex<float>> &data, int start, int N,
                   Dir dir) {
  const int direction = static_cast<int>(dir);
  const float scale = direction == 1 ? 1.0f : 0.5f;

  for (int size = N; size > 1; size >>= 1) {
    const auto &twiddles = twiddles_for_size(size, dir);

    for (int block = 0; block < N / size; ++block) {
      int n1 = start + block * size;
      for (int n = 0; n < (size >> 1); ++n) {
        const int i1 = n1 + n;
        const int i2 = n1 + (n + (size >> 1));

        auto a = data[i1];
        auto b = data[i2];

        data[i1] = scale * (a + b);
        data[i2] = scale * (a - b) * twiddles[n];
      }
    }
  }

  bit_reversal(data, N, start);
}

} // namespace

void fft_recur(std::vector<std::complex<float>> &data, int M, int N, Dir dir) {
  ensure_twiddles(std::max(M, N));

  const int direction = static_cast<int>(dir);
  const float scale = direction == 1 ? 1.0f : 0.5f;

  for (int y = 0; y < M; ++y) {
    fft_recur_impl(data, direction, y * N, N, dir, scale);
    bit_reversal(data, N, y * N);
  }

  util::transpose_flattened(data, M, N);

  for (int x = 0; x < N; ++x) {
    fft_recur_impl(data, direction, x * M, M, dir, scale);
    bit_reversal(data, M, x * M);
  }

  util::transpose_flattened(data, N, M);
}

void fft_iter(std::vector<std::complex<float>> &data, int M, int N, Dir dir) {
  ensure_twiddles(std::max(M, N));

  for (int y = 0; y < M; ++y) {
    fft_iter_impl(data, y * N, N, dir);
  }

  util::transpose_flattened(data, M, N);

  for (int x = 0; x < N; ++x) {
    fft_iter_impl(data, x * M, M, dir);
  }

  util::transpose_flattened(data, N, M);
}

} // namespace internal

void transform(unsigned char *data, int width, int height, float quality,
               internal::Method method) {
  const int M = std::bit_ceil(static_cast<unsigned int>(height));
  const int N = std::bit_ceil(static_cast<unsigned int>(width));

  std::vector<std::complex<float>> img(M * N, 0);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      img[y * N + x] = static_cast<float>(data[y * width + x]) / 255.0f;
    }
  }

  if (method == internal::Method::ITER) {
    internal::fft_iter(img, M, N, internal::Dir::Forward);
  } else {
    internal::fft_recur(img, M, N, internal::Dir::Forward);
  }

  std::vector<float> mags;
  mags.reserve(img.size());
  for (auto &v : img) {
    mags.push_back(std::abs(v));
  }

  const int idx =
      std::min(static_cast<int>(std::floor((1.0f - quality) * mags.size())),
               static_cast<int>(mags.size()) - 1);
  std::nth_element(mags.begin(), mags.begin() + idx, mags.end());
  const float thresh = mags[idx];

  for (size_t i = 1; i < img.size(); ++i) {
    if (std::abs(img[i]) < thresh) {
      img[i] = 0;
    }
  }

  if (method == internal::Method::ITER) {
    internal::fft_iter(img, M, N, internal::Dir::Inverse);
  } else {
    internal::fft_recur(img, M, N, internal::Dir::Inverse);
  }

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      float v = std::clamp(img[y * N + x].real(), 0.0f, 1.0f);
      data[y * width + x] = static_cast<unsigned char>(v * 255.0f);
    }
  }
}

} // namespace fft2
