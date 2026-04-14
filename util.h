#pragma once

#include <utility>
#include <vector>

namespace util {

namespace internal {

template <typename T>
void transpose_flattened_square(std::vector<T> &arr, int N) {
  for (int i = 0; i < N; ++i) {
    for (int j = i + 1; j < N; ++j) {
      std::swap(arr[i * N + j], arr[j * N + i]);
    }
  }
}

} // namespace internal

template <typename T>
void transpose_flattened(std::vector<T> &arr, int M, int N) {
  if (M == N) {
    internal::transpose_flattened_square(arr, N);
    return;
  }

  std::vector<T> buffer(arr.size());

  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      buffer[j * M + i] = arr[i * N + j];
    }
  }

  arr = std::move(buffer);
}

} // namespace util
