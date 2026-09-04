#include "network.hpp"
#include "log.hpp"
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>

namespace axon {
Network::Network(std::shared_ptr<NetworkData> data, double learningRate) {

  learningRate_ = learningRate;
  const int layers = data->GetNumberOfLayers();
  const std::vector<int> &layout = data->GetNetworkLayout();
  const std::vector<std::function<double(double)>> &activationFunctions =
      data->GetActivationFunctions();
  const std::vector<std::function<double(double)>>
      &activationFunctionDerivatives = data->GetActivationFunctionDerivatives();
  biggestLayer_ = 0;
  for (int i = 0; i < layout.size(); i++) {
    if (layout[i] > biggestLayer_)
      biggestLayer_ = layout[i];
  }
  layers_.resize(layers - 1);
  for (int i = 0; i < layers - 1; i++) {
    layers_[i] = std::make_unique<Layer>(layout[i], layout[i + 1],
                                         activationFunctions[i],
                                         activationFunctionDerivatives[i]);
  }
  inputCount_ = layout[0];
  outputCount_ = layout[layers - 1];
  layerCount_ = layers - 1;
  data_ = data;
  precission_ = 0;
  srand((int)time(NULL));
}

Network::~Network() {}

void Network::Train(int epochs) {
  for (int epoch = 0; epoch < epochs; epoch++) {
    Train();
  }
}

void Network::Train() {
  precission_ = TestNetwork_();

  Log(std::to_string(precission_));
  std::vector<double> inputs(biggestLayer_);
  std::vector<double> outputs(biggestLayer_);
  data_->LoadTrainingData();
  while (data_->GetNextTrainingData(inputs, outputs)) {
    Train_(inputs, outputs);
  }
}

double Network::CalculateError_(const std::vector<double> &expectedOutputs,
                                const std::vector<double> &calculatedOutputs,
                                int outputCount) {
  double finalError = 0;
  double currentError;
  for (int i = 0; i < outputCount; i++) {
    currentError = expectedOutputs[i] - calculatedOutputs[i];
    currentError *= currentError;
    finalError += currentError;
  }
  return finalError;
}

void Network::SaveNetworkWeights(const std::string &path) {
  std::fstream file;
  file.open(path, std::ios::out | std::ios::binary);
  for (int i = 0; i < layerCount_; i++)
    layers_[i]->SaveLayer(file);
}

bool Network::LoadNetworkWeights(const std::string &path) {
  std::fstream file;
  file.open(path, std::ios::in | std::ios::binary);
  if (!file.is_open())
    return false;
  for (int i = 0; i < layerCount_; i++) {
    layers_[i]->LoadLayer(file);
  }
  return true;
}

void Network::Train_(std::vector<double> &inputs,
                     std::vector<double> &outputs) {
  std::vector<double> comp = Compute(inputs);
  for (int i = 0; i < comp.size(); i++)
    inputs[i] = comp[i];
  Backpropagation_(inputs, outputs);
}

std::vector<double> Network::Compute(const std::vector<double> &inputs) {
  std::vector ret = inputs;
  for (int j = 0; j < layerCount_; j++) {
    ret = layers_[j]->Compute(ret);
  }
  return ret;
}

void Network::Test() {
  double prec = TestNetwork_();

  Log(std::to_string(prec) + " " + std::to_string(learningRate_));
}

void Network::SetLearningRate(double lr) { learningRate_ = lr; }

double Network::TestNetwork_() {
  std::vector<double> inputs(biggestLayer_);
  std::vector<double> outputs(biggestLayer_);
  if (!data_->LoadTestData())
    Log(Log.error, "Failed to load test data");
  double error = 0;
  int i = 0;
  while (data_->GetNextTestData(inputs, outputs)) {
    error += CalculateError_(Compute(inputs), outputs, outputCount_);
    i++;
  }
  return error / i;
}

void Network::Backpropagation_(std::vector<double> &inputs,
                               std::vector<double> &outputs) {

  layers_[layerCount_ - 1]->Delta(outputs, Layer::DeltaMode::Diffrence);
  for (int i = layerCount_ - 2; i >= 0; i--) {
    layers_[i]->Delta(outputs, Layer::DeltaMode::Multiplication);
  }

  for (int i = layerCount_ - 1; i >= 1; i--) {
    layers_[i]->Descent(layers_[i - 1]->GetOutputs(), learningRate_);
  }
  data_->GetSameTrainingData(inputs, outputs);
  layers_[0]->Descent(inputs, learningRate_);
}
} // namespace axon
