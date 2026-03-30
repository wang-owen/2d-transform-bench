#include "dct2.h"
#include "dft2.h"
#include "fft2.h"
#include "stb/stb_image.h"
#include "stb/stb_image_write.h"

#include <format>
#include <iostream>
#include <string>
#include <string_view>

constexpr int GRAYSCALE = 1;

int main(int argc, char *argv[]) {
  const std::string usage = std::format(
      "usage: {} --dft|--fft [iter|recur]|--dct <input> <output> [quality]\n"
      "  --fft defaults to iter if method is omitted\n"
      "  quality: [0, 1] for DFT/FFT, unrestricted for DCT\n",
      argv[0]);

  if (argc < 4) {
    std::cerr << usage;
    exit(1);
  }

  const std::string_view algorithm = argv[1];

  // Resolve FFT method and determine where positional args start.
  auto fft_method = fft2::internal::Method::ITER;
  int pos = 2;
  if (algorithm == "--fft" && pos < argc) {
    const std::string_view method = argv[pos];
    if (method == "iter") {
      fft_method = fft2::internal::Method::ITER;
      ++pos;
    } else if (method == "recur") {
      fft_method = fft2::internal::Method::RECUR;
      ++pos;
    }
  }

  if (pos + 2 > argc || pos + 3 < argc) {
    std::cerr << usage;
    exit(1);
  }

  const char *input_path = argv[pos];
  const char *output_path = argv[pos + 1];

  double quality = -1;
  if (pos + 2 < argc) {
    try {
      quality = std::stod(argv[pos + 2]);
    } catch (const std::invalid_argument &e) {
      std::cerr << e.what() << '\n';
      std::cerr << usage;
      exit(1);
    }
  }

  if (algorithm != "--dct" && quality != -1 && (quality < 0 || quality > 1)) {
    std::cerr << "quality must be in [0, 1] for non-DCT transforms\n";
    std::cerr << usage;
    exit(1);
  }

  if (quality == -1) {
    quality = algorithm == "--dct" ? 1.0 : 0.5;
  }

  int width, height, channels;
  unsigned char *data =
      stbi_load(input_path, &width, &height, &channels, GRAYSCALE);

  if (data == nullptr) {
    std::cerr << "Failed to open image at " << input_path << '\n';
    exit(1);
  }

  if (algorithm == "--dft") {
    dft2::transform(data, width, height, quality);
  } else if (algorithm == "--fft") {
    fft2::transform(data, width, height, quality, fft_method);
  } else if (algorithm == "--dct") {
    dct2::transform(data, width, height, quality);
  } else {
    std::cerr << "unknown algorithm: " << algorithm << '\n';
    std::cerr << usage;
    stbi_image_free(data);
    exit(1);
  }

  stbi_write_png(output_path, width, height, GRAYSCALE, data, width);

  stbi_image_free(data);
}
