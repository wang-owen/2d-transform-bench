#pragma once

#include "common.h"

#include <vector>

namespace dct2 {

namespace internal {

using Dir = common::Dir;

void dct2(std::vector<float> &data, int M, int N, Dir dir);

void dct2_threaded(std::vector<float> &data, int M, int N, Dir dir);

void quantize(std::vector<float> &data, int M, int N, float quality);

void dequantize(std::vector<float> &data, int M, int N, float quality);

} // namespace internal

void transform(unsigned char *data, int width, int height, float quality = 0.5f,
               bool threaded = false);

} // namespace dct2
