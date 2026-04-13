#pragma once

#include <complex>
#include <vector>

namespace dft2 {

namespace internal {

enum class Dir { Forward = 1, Inverse = -1 };

void dft2_strided(std::vector<std::complex<float>> &data, int M, int N,
                  Dir dir = Dir::Forward);

[[deprecated]]
void dft2(std::vector<std::complex<float>> &data, int M, int N,
          Dir dir = Dir::Forward);

} // namespace internal

void transform(unsigned char *data, int width, int height,
               float quality = 0.1f);

} // namespace dft2
