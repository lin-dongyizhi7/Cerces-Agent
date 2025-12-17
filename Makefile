CXX = g++
CXXFLAGS = -std=c++11 -pthread -Wall -O2
TARGETS = anomaly_detector

all: $(TARGETS)

anomaly_detector: anomaly_detector.cpp metrics_common.h
	$(CXX) $(CXXFLAGS) anomaly_detector.cpp -o anomaly_detector

clean:
	rm -f $(TARGETS)

.PHONY: all clean

