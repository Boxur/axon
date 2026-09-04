#include "layer.hpp"
#include <cassert>
#include <functional>

namespace axon {
Layer::Layer(int inputs, int outputs,
             std::function<double(double)> activationFunction,
             std::function<double(double)> activationFunctionDerivative)
    : inputCount_(inputs), outputCount_(outputs),
      activationFunction_(activationFunction),
      activationFunctionDerivative_(activationFunctionDerivative) {
  srand(time(NULL));
  outputs_.resize(outputs);
  biases_.resize(outputs);
  weights_.resize(inputs * outputs);
  deltas_.resize(outputs);

  InitWeights();
  return;
}

Layer::~Layer() {}

void Layer::InitWeights() {
  for (int i = 0; i < inputCount_; i++) {
    for (int j = 0; j < outputCount_; j++) {
      weights_[i * outputCount_ + j] = Random() - 0.5;
    }
  }
  for (int i = 0; i < outputCount_; i++) {
    biases_[i] = 0; // Utility::Random();
  }
}

std::vector<double> Layer::Compute(const std::vector<double> &inputs) {
  double activation;
  for (int i = 0; i < outputCount_; i++) {
    activation = biases_[i];
    for (int j = 0; j < inputCount_; j++) {
      assert(i < outputs_.size());
      assert(j * outputCount_ + i < weights_.size());
      activation += inputs[j] * weights_[j * outputCount_ + i];
    }
    outputs_[i] = activationFunction_(activation);
  }
  return outputs_;
  ;
}

void Layer::Delta(std::vector<double> &output, DeltaMode mode) {
  if (mode == DeltaMode::Diffrence) {
    for (int i = 0; i < outputCount_; i++) {
      double error = (output[i] - outputs_[i]);
      deltas_[i] = error * activationFunctionDerivative_(outputs_[i]);
    }
  } else {
    for (int i = 0; i < outputCount_; i++) {
      deltas_[i] = output[i] * activationFunctionDerivative_(outputs_[i]);
    }
  }
  // if(output.size()<inputCount_)
  //	output.resize(inputCount_);
  double error;
  for (int i = 0; i < inputCount_; i++) {
    error = 0.0f;
    for (int j = 0; j < outputCount_; j++) {
      error += deltas_[j] * weights_[i * outputCount_ + j];
    }
    output[i] = error;
  }
}

void Layer::Descent(const std::vector<double> &inputs, double learningRate) {
  for (int i = 0; i < outputCount_; i++) {
    biases_[i] += deltas_[i] * learningRate;
    for (int j = 0; j < inputCount_; j++) {
      weights_[j * outputCount_ + i] += inputs[j] * deltas_[i] * learningRate;
    }
  }
}

double Layer::Random() { return (double)rand() / RAND_MAX; }

void Layer::SaveLayer(std::fstream &file) {
  file.write((char *)&inputCount_, sizeof(int));
  file.write((char *)&outputCount_, sizeof(int));
  file.write(reinterpret_cast<char *>(weights_.data()),
             sizeof(double) * weights_.size());
  file.write(reinterpret_cast<char *>(biases_.data()),
             outputCount_ * sizeof(double));
}

void Layer::LoadLayer(std::fstream &file) {
  file.read((char *)&inputCount_, sizeof(int));
  file.read((char *)&outputCount_, sizeof(int));
  file.read(reinterpret_cast<char *>(weights_.data()),
            sizeof(double) * weights_.size());
  file.read(reinterpret_cast<char *>(biases_.data()),
            outputCount_ * sizeof(double));
}

std::vector<double> &Layer::GetOutputs() { return outputs_; }
} // namespace axon
