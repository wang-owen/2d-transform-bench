#pragma once

#include <complex>
#include <vector>

namespace dft2 {

namespace internal {

void dft2(std::vector<std::complex<double>> &data, int M, int N);

void idft2(std::vector<std::complex<double>> &data, int M, int N);

} // namespace internal

void transform(unsigned char *data, int width, int height, int channels,
               double ratio = 0.1);

} // namespace dft2