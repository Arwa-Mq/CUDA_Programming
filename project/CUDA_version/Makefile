COSMOSIS_SRC ?= $(shell cosmosis-config --src-dir)
NVCC = nvcc
CXXFLAGS = -O3 -Xcompiler -fPIC -shared -std=c++14

TARGET = boost_factor_likelihood.so

all: $(TARGET)

$(TARGET): boost_factor_likelihood.cu
	$(NVCC) $(CXXFLAGS) -I$(COSMOSIS_SRC)/cosmosis $< -o $@

clean:
	rm -f *.so *.o
