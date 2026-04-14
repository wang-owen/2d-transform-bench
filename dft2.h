#pragma once

#include "common.h"

#include <complex>
#include <vector>

namespace dft2 {

namespace internal {

using Dir = common::Dir;

void dft2_separated(std::vector<std::complex<float>> &data, int M, int N,
                    Dir dir);

void dft2_separated_threaded(std::vector<std::complex<float>> &data, int M,
                             int N, Dir dir);

[[deprecated]]
void dft2(std::vector<std::complex<float>> &data, int M, int N, Dir dir);

} // namespace internal

void transform(unsigned char *data, int width, int height, float quality = 0.1f,
               bool threaded = false);

} // namespace dft2
