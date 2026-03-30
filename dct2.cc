#include "dct2.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>

namespace dct2 {

namespace internal {

namespace {

constexpr std::array<std::array<double, 8>, 8> COS_TABLE = {
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

void dct1_impl(std::vector<double> &data, int start, int length, int stride) {
  assert(length % 8 == 0);

  const int N = 8;

  std::vector<double> temp;
  temp.reserve(length);
  for (int i = start; i < start + length * stride; i += stride) {
    temp.push_back(data[i]);
  }

  const double C_k = sqrt(2.0 / N);

  int blockIdx = 0;
  for (int s = start; s < start + length * stride; s += 8 * stride) {
    for (int k = 0; k < N; ++k) {
      int i = s + k * stride;
      data[i] = 0;

      for (int n = 0; n < N; ++n) {
        data[i] += temp[blockIdx * 8 + n] * COS_TABLE[k][n];
      }

      data[i] *= k == 0 ? sqrt(1.0 / N) : C_k;
    }
    ++blockIdx;
  }
}

void idct1_impl(std::vector<double> &data, int start, int length, int stride) {
  assert(length % 8 == 0);

  const int N = 8;

  std::vector<double> temp;
  temp.reserve(length);
  for (int i = start; i < start + length * stride; i += stride) {
    temp.push_back(data[i]);
  }

  const double C_k = sqrt(2.0 / N);

  int blockIdx = 0;
  for (int s = start; s < start + length * stride; s += 8 * stride) {
    for (int n = 0; n < N; ++n) {
      int i = s + n * stride;
      data[i] = 0;

      for (int k = 0; k < N; ++k) {
        data[i] += (k == 0 ? sqrt(1.0 / N) : C_k) * temp[blockIdx * 8 + k] *
                   COS_TABLE[k][n];
      }
    }
    ++blockIdx;
  }
}

} // namespace

void dct2(std::vector<double> &data, int M, int N, Dir dir) {
  auto dct_ptr = dir == Dir::Forward ? dct1_impl : idct1_impl;

  for (int y = 0; y < M; ++y) {
    dct_ptr(data, y * N, N, 1);
  }
  for (int x = 0; x < N; ++x) {
    dct_ptr(data, x, M, N);
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

void quantize(std::vector<double> &data, int M, int N, double ratio) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      data[i * N + j] = std::round(data[i * N + j] /
                                   (ratio * std_luminance_table[i % 8][j % 8]));
    }
  }
}

void dequantize(std::vector<double> &data, int M, int N, double ratio) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      data[i * N + j] *= (ratio * std_luminance_table[i % 8][j % 8]);
    }
  }
}

} // namespace internal

void transform(unsigned char *data, int width, int height, double ratio) {
  const int M = ((height + 7) / 8) * 8;
  const int N = ((width + 7) / 8) * 8;

  std::vector<double> img(N * M, 0);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      img[y * N + x] = (static_cast<double>(data[y * width + x]) - 128);
    }
  }

  internal::dct2(img, M, N, internal::Dir::Forward);

  internal::quantize(img, M, N, ratio);

  internal::dequantize(img, M, N, ratio);

  internal::dct2(img, M, N, internal::Dir::Inverse);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      double v = std::clamp(img[y * N + x], -128.0, 127.0);
      data[y * width + x] = static_cast<unsigned char>(v + 128);
    }
  }
}

} // namespace dct2
