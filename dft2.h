#pragma once

#include <complex>
#include <vector>

namespace dft2 {

namespace internal {

void dft2(std::vector<std::complex<double>> &data, int N, int M);

void idft2(std::vector<std::complex<double>> &data, int N, int M);

} // namespace internal

void transform(unsigned char *data, int width, int height, int channels,
               double ratio = 0.1);

} // namespace dft2