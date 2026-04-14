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

enum class Algorithm { DFT, FFT, DCT };

int main(int argc, char *argv[]) {
  const std::string usage =
      std::format("usage: {} [-t] --dft|--fft|--dct <input> "
                  "<output> [quality]\n"
                  "  -t: run transform using multiple threads\n"
                  "  quality: [0, 1], higher is better (default 0.5)\n",
                  argv[0]);

  bool threaded = false;
  bool has_algorithm = false;
  Algorithm algorithm = Algorithm::DFT;

  int pos = 1;
  while (pos < argc) {
    const std::string_view arg = argv[pos];

    if (arg == "-t") {
      threaded = true;
      ++pos;
      continue;
    }

    if (arg == "--dft" || arg == "--fft" || arg == "--dct") {
      if (has_algorithm) {
        std::cerr << "multiple algorithms specified\n";
        std::cerr << usage;
        return 1;
      }

      has_algorithm = true;
      if (arg == "--dft") {
        algorithm = Algorithm::DFT;
      } else if (arg == "--fft") {
        algorithm = Algorithm::FFT;
      } else {
        algorithm = Algorithm::DCT;
      }

      ++pos;
      continue;
    }

    break;
  }

  if (!has_algorithm) {
    std::cerr << "missing algorithm flag\n";
    std::cerr << usage;
    return 1;
  }

  const int remaining = argc - pos;
  if (remaining != 2 && remaining != 3) {
    std::cerr << usage;
    return 1;
  }

  const char *input_path = argv[pos];
  const char *output_path = argv[pos + 1];

  float quality = 0.5f;
  if (remaining == 3) {
    try {
      quality = std::stof(argv[pos + 2]);
    } catch (const std::exception &e) {
      std::cerr << e.what() << '\n';
      std::cerr << usage;
      return 1;
    }
  }

  if (quality < 0 || quality > 1) {
    std::cerr << "quality must be in [0, 1]\n";
    std::cerr << usage;
    return 1;
  }

  int width, height, channels;
  unsigned char *data =
      stbi_load(input_path, &width, &height, &channels, GRAYSCALE);

  if (data == nullptr || width == 0 || height == 0) {
    std::cerr << "Failed to open image at " << input_path << '\n';
    return 1;
  }

  if (algorithm == Algorithm::DFT) {
    dft2::transform(data, width, height, quality, threaded);
  } else if (algorithm == Algorithm::FFT) {
    fft2::transform(data, width, height, quality, threaded);
  } else {
    dct2::transform(data, width, height, quality, threaded);
  }

  stbi_write_png(output_path, width, height, GRAYSCALE, data, width);

  stbi_image_free(data);

  return 0;
}
