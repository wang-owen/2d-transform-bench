#pragma once

#include <complex>
#include <vector>

namespace fft2 {

namespace internal {

enum class Dir { Forward = 1, Inverse = -1 };

enum class Method { ITER, RECUR };

void fft_strided_iter(std::vector<std::complex<double>> &data, int start, int N,
                      int stride, Dir dir = Dir::Forward);

void fft_strided_recur(std::vector<std::complex<double>> &data, int start,
                       int N, int stride, Dir dir = Dir::Forward);

} // namespace internal

void transform(unsigned char *data, int width, int height, double ratio = 0.1,
               internal::Method method = internal::Method::ITER);

} // namespace fft2
