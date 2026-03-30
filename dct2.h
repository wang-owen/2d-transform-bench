#pragma once

#include <vector>

namespace dct2 {

namespace internal {

enum class Dir { Forward = 1, Inverse = -1 };

void dct2(std::vector<double> &data, int M, int N, Dir dir = Dir::Forward);

void quantize(std::vector<double> &data, int M, int N, double ratio);

void dequantize(std::vector<double> &data, int M, int N, double ratio);

} // namespace internal

void transform(unsigned char *data, int width, int height, double ratio = 1);

} // namespace dct2
