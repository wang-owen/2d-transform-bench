#pragma once

#include "common.h"

#include <complex>
#include <vector>

namespace fft2 {

namespace internal {

using Dir = common::Dir;

enum class Method { ITER, RECUR };

void fft_iter(std::vector<std::complex<float>> &data, int M, int N, Dir dir);

void fft_recur(std::vector<std::complex<float>> &data, int M, int N, Dir dir);

void fft_threaded(std::vector<std::complex<float>> &data, int M, int N,
                  Dir dir);

} // namespace internal

void transform(unsigned char *data, int width, int height, float quality = 0.1f,
               bool threaded = false);

} // namespace fft2
