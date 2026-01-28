#pragma once

#include <complex>
#include <vector>

namespace fft2 {

namespace internal {

enum class FFTDir { Forward = 1, Inverse = -1 };

void fft_strided_iter(std::vector<std::complex<double>> &data, FFTDir dir,
                      size_t start, size_t N, size_t stride);

void fft_strided_recur(std::vector<std::complex<double>> &data, FFTDir dir,
                       size_t start, size_t N, size_t stride);

} // namespace internal

void transform(unsigned char *data, int width, int height, int channels,
               double ratio = 0.1, bool recur = false);

} // namespace fft2
