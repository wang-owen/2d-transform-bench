CXX      := clang++
CXXFLAGS := -std=c++20 -O3 -Wall -Wextra

# Source modules
LIB_SRCS := dft2.cc fft2.cc dct2.cc stb_impl.cc
LIB_OBJS := $(LIB_SRCS:.cc=.o)

# Targets
all: main benchmark equality_test

main: main.o dct2.o dft2.o fft2.o stb_impl.o
	$(CXX) $(CXXFLAGS) -o $@ $^

benchmark: benchmark.o dct2.o dft2.o fft2.o
	$(CXX) $(CXXFLAGS) -o $@ $^

equality_test: equality_test.o dft2.o fft2.o stb_impl.o
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cc
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f *.o main benchmark equality_test

.PHONY: all clean
