#pragma once

#include <vector>

namespace dct2 {

namespace internal {

enum class Dir { Forward = 1, Inverse = -1 };

void dct2(std::vector<float> &data, int M, int N, Dir dir = Dir::Forward);

void quantize(std::vector<float> &data, int M, int N, float quality);

void dequantize(std::vector<float> &data, int M, int N, float quality);

} // namespace internal

void transform(unsigned char *data, int width, int height,
               float quality = 1.0f);

} // namespace dct2
