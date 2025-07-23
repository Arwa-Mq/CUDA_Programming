# Simple Makefile for building CosmoSIS C++ module
COSMOSIS_SRC ?= $(shell cosmosis-config --src-dir)
CXX = g++
CXXFLAGS = -O3 -fPIC -shared -std=c++11
TARGET = boost_factor_likelihood.so

all: $(TARGET)

$(TARGET): boost_factor_likelihood.cpp
	$(CXX) $(CXXFLAGS) -I$(COSMOSIS_SRC)/cosmosis $< -o $@

clean:
	rm -f *.so *.o
