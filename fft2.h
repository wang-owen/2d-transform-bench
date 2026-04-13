#pragma once

#include <complex>
#include <vector>

namespace fft2 {

namespace internal {

enum class Dir { Forward = 1, Inverse = -1 };

enum class Method { ITER, RECUR };

void fft_iter(std::vector<std::complex<float>> &data, int M, int N,
              Dir dir = Dir::Forward);

void fft_recur(std::vector<std::complex<float>> &data, int M, int N,
               Dir dir = Dir::Forward);

} // namespace internal

void transform(unsigned char *data, int width, int height, float quality = 0.1f,
               internal::Method method = internal::Method::ITER);

} // namespace fft2
