#include "dct2.h"
#include "util.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <thread>

namespace dct2 {

namespace internal {

namespace {

// COS_TABLE[k][n] = cos(π * k * (2n + 1) / 16): the DCT-II basis values for an
// 8-point transform. Row k is the k-th frequency's basis vector over 8 samples.
constexpr std::array<std::array<float, 8>, 8> COS_TABLE = {
    {{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
     {0.980785, 0.83147, 0.55557, 0.19509, -0.19509, -0.55557, -0.83147,
      -0.980785},
     {0.92388, 0.382683, -0.382683, -0.92388, -0.92388, -0.382683, 0.382683,
      0.92388},
     {0.83147, -0.19509, -0.980785, -0.55557, 0.55557, 0.980785, 0.19509,
      -0.83147},
     {0.707107, -0.707107, -0.707107, 0.707107, 0.707107, -0.707107, -0.707107,
      0.707107},
     {0.55557, -0.980785, 0.19509, 0.83147, -0.83147, -0.19509, 0.980785,
      -0.55557},
     {0.382683, -0.92388, 0.92388, -0.382683, -0.382683, 0.92388, -0.92388,
      0.382683},
     {0.19509, -0.55557, 0.83147, -0.980785, 0.980785, -0.83147, 0.55557,
      -0.19509}}};

constexpr float C_0 = 0.353553f; // sqrt(1.0f / N);
constexpr float C_k = 0.5f;      // sqrt(2.0f / N);

// Forward DCT on each 8-element block within [start, start+length).
// scratch holds a copy of the input so in-place writes don't corrupt reads.
// blockIdx tracks which 8-element block within the row we're processing;
// blockIdx * 8 is its offset into the scratch buffer.
void dct1_impl(std::vector<float> &data, int start, int length,
               std::vector<float> &scratch) {
  assert(length % 8 == 0);

  const int N = 8;

  size_t j = 0;
  for (int i = start; i < start + length; ++i) {
    scratch[j++] = data[i];
  }

  int blockIdx = 0;
  for (int s = start; s < start + length; s += 8) {
    // DC coefficient (k=0) uses C_0 = sqrt(1/N) normalisation.
    data[s] = 0;
    for (int n = 0; n < N; ++n) {
      data[s] += scratch[blockIdx * 8 + n] * COS_TABLE[0][n];
    }
    data[s] *= C_0;

    // AC coefficients (k=1..7) use C_k = sqrt(2/N) normalisation.
    for (int k = 1; k < N; ++k) {
      int i = s + k;
      data[i] = 0;

      for (int n = 0; n < N; ++n) {
        data[i] += scratch[blockIdx * 8 + n] * COS_TABLE[k][n];
      }

      data[i] *= C_k;
    }
    ++blockIdx;
  }
}

void idct1_impl(std::vector<float> &data, int start, int length,
                std::vector<float> &scratch) {
  assert(length % 8 == 0);

  const int N = 8;

  size_t j = 0;
  for (int i = start; i < start + length; ++i) {
    scratch[j++] = data[i];
  }

  int blockIdx = 0;
  for (int s = start; s < start + length; s += 8) {
    for (int n = 0; n < N; ++n) {
      int i = s + n;
      data[i] = C_0 * scratch[blockIdx * 8] * COS_TABLE[0][n];

      for (int k = 1; k < N; ++k) {
        data[i] += C_k * scratch[blockIdx * 8 + k] * COS_TABLE[k][n];
      }
    }
    ++blockIdx;
  }
}

} // namespace

void dct2(std::vector<float> &data, int M, int N, Dir dir) {
  auto dct_ptr = dir == Dir::Forward ? dct1_impl : idct1_impl;

  std::vector<float> scratch(std::max(M, N));

  for (int y = 0; y < M; ++y) {
    dct_ptr(data, y * N, N, scratch);
  }

  util::transpose_flattened(data, M, N);

  for (int x = 0; x < N; ++x) {
    dct_ptr(data, x * M, M, scratch);
  }

  util::transpose_flattened(data, N, M);
}

void dct2_threaded(std::vector<float> &data, int M, int N, Dir dir) {
  const int blocks_per_row = N / 8;
  const int total_blocks = (M / 8) * blocks_per_row;

  const int num_threads = std::max(1u, std::thread::hardware_concurrency());
  const int blocks_per_thread = (total_blocks + num_threads - 1) / num_threads;
  const int num_active = std::min(num_threads, total_blocks);

  auto dct_ptr = (dir == Dir::Forward) ? &dct1_impl : &idct1_impl;

  auto process_range = [&](int b_start, int b_end) {
    std::vector<float> local_buf(64);
    std::vector<float> scratch(64);

    // Transpose the 8×8 local buffer in-place so the column pass can
    // reuse dct1_impl, which operates on contiguous rows.
    auto transpose8 = [&]() {
      for (int r = 0; r < 8; ++r)
        for (int c = r + 1; c < 8; ++c)
          std::swap(local_buf[r * 8 + c], local_buf[c * 8 + r]);
    };

    for (int b = b_start; b < b_end; ++b) {
      // Convert flat block index to top-left pixel coordinates of the block.
      const int br = (b / blocks_per_row) * 8;
      const int bc = (b % blocks_per_row) * 8;

      for (int r = 0; r < 8; ++r)
        for (int c = 0; c < 8; ++c)
          local_buf[r * 8 + c] = data[(br + r) * N + (bc + c)];

      dct_ptr(local_buf, 0, 64, scratch);

      transpose8();
      dct_ptr(local_buf, 0, 64, scratch);
      transpose8();

      for (int r = 0; r < 8; ++r)
        for (int c = 0; c < 8; ++c)
          data[(br + r) * N + (bc + c)] = local_buf[r * 8 + c];
    }
  };

  std::vector<std::jthread> pool;
  pool.reserve(num_active);
  for (int i = 0; i < num_active; ++i) {
    int b_start = i * blocks_per_thread;
    int b_end = std::min(b_start + blocks_per_thread, total_blocks);
    pool.emplace_back(process_range, b_start, b_end);
  }
}

constexpr std::array<std::array<int, 8>, 8> std_luminance_table = {
    {{{16, 11, 10, 16, 24, 40, 51, 61}},
     {{12, 12, 14, 19, 26, 58, 60, 55}},
     {{14, 13, 16, 24, 40, 57, 69, 56}},
     {{14, 17, 22, 29, 51, 87, 80, 62}},
     {{18, 22, 37, 56, 68, 109, 103, 77}},
     {{24, 35, 55, 64, 81, 104, 113, 92}},
     {{49, 64, 78, 87, 103, 121, 120, 101}},
     {{72, 92, 95, 98, 112, 100, 103, 99}}}};

// Quantizes DCT coefficients using the JPEG standard luminance table scaled by
// quality. Follows the IJG/libjpeg convention: quality ∈ [0,1] is mapped to a
// q-value in [1,100], then to a scale factor S that multiplies each table
// entry. Lower quality → larger step sizes → more aggressive rounding.
void quantize(std::vector<float> &data, int M, int N, float quality) {
  float q = quality * 99.0f + 1.0f;
  float S = q < 50.0f ? 5000.0f / q : 200.0f - 2.0f * q;
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float step = std::clamp(
          (std_luminance_table[i % 8][j % 8] * S + 50) / 100, 1.0f, 255.0f);
      data[i * N + j] = std::round(data[i * N + j] / step);
    }
  }
}

void dequantize(std::vector<float> &data, int M, int N, float quality) {
  float q = quality * 99.0f + 1.0f;
  float S = q < 50.0f ? 5000.0f / q : 200.0f - 2.0f * q;
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float step = std::clamp(
          (std_luminance_table[i % 8][j % 8] * S + 50) / 100, 1.0f, 255.0f);
      data[i * N + j] *= step;
    }
  }
}

} // namespace internal

void transform(unsigned char *data, int width, int height, float quality,
               bool threaded) {
  const int M = ((height + 7) / 8) * 8;
  const int N = ((width + 7) / 8) * 8;

  std::vector<float> img(N * M, 0);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      img[y * N + x] = (static_cast<float>(data[y * width + x]) - 128);
    }
  }

  if (threaded) {
    internal::dct2_threaded(img, M, N, internal::Dir::Forward);
  } else {
    internal::dct2(img, M, N, internal::Dir::Forward);
  }

  internal::quantize(img, M, N, quality);

  internal::dequantize(img, M, N, quality);

  if (threaded) {
    internal::dct2_threaded(img, M, N, internal::Dir::Inverse);
  } else {
    internal::dct2(img, M, N, internal::Dir::Inverse);
  }

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      float v = std::clamp(img[y * N + x], -128.0f, 127.0f);
      data[y * width + x] = static_cast<unsigned char>(v + 128);
    }
  }
}

} // namespace dct2
