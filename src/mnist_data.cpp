#include "mnist_data.hpp"
#include "log.hpp"
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>

static double leakyRelu(double x) { return ((x < 0) ? x / 10 : x); }

static double leakyReluDerivative(double x) { return ((x < 0) ? 0.1 : 1); }

static double sigmoidApprox(double x) {
  return ((x < -2.5) ? 0 : ((x < 2.5) ? 0.2 * x + 0.5 : 1));
}

static double sigmoidApproxDerivative(double x) {
  return ((x < -2.5) ? 0.05 : ((x < 2.5) ? 0.2 : 0.05));
}

MnistData::MnistData()
    : networkLayout_({784, 30, 30, 11}),
      activationFunctions_({leakyRelu, leakyRelu, leakyRelu, sigmoidApprox}),
      activationFunctionDerivatives_({leakyReluDerivative, leakyReluDerivative,
                                      leakyReluDerivative,
                                      sigmoidApproxDerivative}) {
  skip_ = new char[16];
  trainingInputArray_.resize(784);
  trainingOutputArray_.resize(11);
  testInputArray_.resize(784);
  testOutputArray_.resize(11);
}

MnistData::~MnistData() {
  trainingInputs_.close();
  trainingOutputs_.close();
  delete[] skip_;
}
bool MnistData::GetNextTrainingData(std::vector<double> &inputs,
                                    std::vector<double> &outputs) {

  if (trainingInputs_.eof() || trainingOutputs_.eof() ||
      !trainingInputs_.good() || !trainingOutputs_.good()) {
    return false;
  }
  if (inputs.size() < 784)
    axon::Log(axon::Log.error, "The input vector is too small");
  if (outputs.size() < 11)
    axon::Log(axon::Log.error, "The output vector is too small");
  if (rand() % 15 == 0) {
    for (int i = 0; i < 784; i++) {
      inputs[i] = (double)(rand() % 256) / 255.0;
      trainingInputArray_[i] = inputs[i];
    }
    for (int i = 0; i < 11; i++) {
      outputs[i] = 0;
      trainingOutputArray_[i] = outputs[i];
    }
    outputs[10] = 1;
    trainingOutputArray_[10] = 1;
  } else {
    char byte;
    for (int i = 0; i < 784; i++) {
      trainingInputs_.read(&byte, 1);
      inputs[i] = (double)((unsigned char)byte + (rand() % 50 - 25)) / 255.0;
      inputs[i] = ((inputs[i] < 0) ? 0 : inputs[i]);
      inputs[i] = ((inputs[i] > 1) ? 1 : inputs[i]);
      trainingInputArray_[i] = inputs[i];
    }
    for (int i = 0; i < 11; i++) {
      outputs[i] = 0;
      trainingOutputArray_[i] = outputs[i];
    }
    if (!trainingOutputs_.read(&byte, 1))
      return false;
    outputs[(int)byte] = 1;
    trainingOutputArray_[(int)byte] = 1;
  }
  return true;
}

void MnistData::GetSameTrainingData(std::vector<double> &inputs,
                                    std::vector<double> &outputs) {
  for (int i = 0; i < 784; i++) {
    inputs[i] = trainingInputArray_[i];
  }
  for (int i = 0; i < 11; i++) {
    outputs[i] = trainingOutputArray_[i];
  }
}

bool MnistData::GetNextTestData(std::vector<double> &inputs,
                                std::vector<double> &outputs) {
  if (!testInputs_ || !testOutputs_) {
    return false;
  }
  if (rand() % 15 == 0) {
    for (int i = 0; i < 784; i++) {
      inputs[i] = (double)(rand() % 256) / 255.0;
      testInputArray_[i] = inputs[i];
    }
    for (int i = 0; i < 11; i++) {
      outputs[i] = 0;
      testOutputArray_[i] = outputs[i];
    }
    outputs[10] = 1;
    testOutputArray_[10] = 1;
  } else {
    char byte;
    for (int i = 0; i < 784; i++) {
      testInputs_.read(&byte, 1);
      inputs[i] = (double)((unsigned char)byte) / 255.0;
      testInputArray_[i] = inputs[i];
    }
    for (int i = 0; i < 11; i++) {
      outputs[i] = 0;
      testOutputArray_[i] = outputs[i];
    }
    if (!testOutputs_.read(&byte, 1))
      return false;
    outputs[(int)byte] = 1;
    testOutputArray_[(int)byte] = 1;
  }
  return true;
}

void MnistData::GetSameTestData(std::vector<double> &inputs,
                                std::vector<double> &outputs) {
  for (int i = 0; i < 784; i++) {
    inputs[i] = testInputArray_[i];
  }
  for (int i = 0; i < 11; i++) {
    outputs[i] = testOutputArray_[i];
  }
}

bool MnistData::LoadTrainingData() {
  trainingInputs_.close();
  trainingOutputs_.close();
  trainingInputs_.open("Assets/data/train-images.ubyte",
                       std::ios::in | std::ios::binary);
  if (!trainingInputs_)
    return false;
  trainingOutputs_.open("Assets/data/train-labels.ubyte",
                        std::ios::in | std::ios::binary);
  if (!trainingOutputs_)
    return false;

  trainingInputs_.read(skip_, 16);
  trainingOutputs_.read(skip_, 8);
  return true;
}

bool MnistData::LoadTestData() {
  testInputs_.close();
  testOutputs_.close();
  testInputs_.open("Assets/data/test-images.ubyte",
                   std::ios::in | std::ios::binary);
  if (!testInputs_)
    return false;
  testOutputs_.open("Assets/data/test-labels.ubyte",
                    std::ios::in | std::ios::binary);
  if (!testOutputs_)
    return false;
  testInputs_.read(skip_, 16);
  testOutputs_.read(skip_, 8);
  return true;
}

const std::vector<int> &MnistData::GetNetworkLayout() const {
  return networkLayout_;
}

const int &MnistData::GetNumberOfLayers() const { return numberOfLayers_; }

const std::vector<std::function<double(double)>> &
MnistData::GetActivationFunctions() const {
  return activationFunctions_;
}

const std::vector<std::function<double(double)>> &
MnistData::GetActivationFunctionDerivatives() const {
  return activationFunctionDerivatives_;
}
