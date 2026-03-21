#include "fft2.h"
#include "stb/stb_image.h"
#include "stb/stb_image_write.h"

#include <format>
#include <iostream>
#include <string>

constexpr int GRAYSCALE = 1;
constexpr double DEFAULT_RATIO = 0.1;

int main(int argc, char *argv[]) {
  const std::string usage =
      std::format("usage: {} <input path> <output path> [ratio]\n", argv[0]);

  if (argc < 3 || argc > 4) {
    std::cerr << usage;
    exit(1);
  }

  double ratio = DEFAULT_RATIO;
  if (argc == 4) {
    try {
      ratio = std::stod(argv[3]);
      if (ratio < 0 || ratio > 1) {
        throw std::invalid_argument("Ratio must be in [0, 1]\n");
      }
    } catch (const std::invalid_argument &e) {
      std::cerr << e.what() << '\n';
      std::cerr << usage;
      exit(1);
    }
  }

  int width, height, channels;
  unsigned char *data =
      stbi_load(argv[1], &width, &height, &channels, GRAYSCALE);

  if (data == nullptr) {
    std::cerr << "Failed to open image at " << argv[1] << '\n';
    exit(1);
  }

  fft2::transform(data, width, height, GRAYSCALE, ratio);
  // dft2::transform(data, width, height, GRAYSCALE, ratio);

  stbi_write_png(argv[2], width, height, GRAYSCALE, data, width);

  stbi_image_free(data);
}
