CXX = g++
OPENMP_FLAGS = $(shell if g++ -fopenmp -x c++ /dev/null -o /dev/null 2>/dev/null; then echo "-fopenmp"; elif clang++ -Xpreprocessor -fopenmp -lomp -x c++ /dev/null -o /dev/null 2>/dev/null; then echo "-Xpreprocessor -fopenmp -lomp"; else echo ""; fi)
CXXFLAGS = -std=c++11 -Wall -Wextra -O2 $(OPENMP_FLAGS)
LDFLAGS = -lm

TARGET = experiment
SOURCES = experiment.cpp
HEADERS = npy_loader.h logistic_regression.h

all: $(TARGET)

$(TARGET): $(SOURCES) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCES) $(LDFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: clean all
