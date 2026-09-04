#pragma once
#include "layer.hpp"
#include "network_data.hpp"

#include <assert.h>
#include <memory>
#include <vector>

namespace axon {
class Network {
private:
  double learningRate_;
  int inputCount_;
  int outputCount_;
  int layerCount_;
  int biggestLayer_;
  std::vector<std::unique_ptr<Layer>> layers_;
  std::unique_ptr<NetworkData> data_;

  double precission_;

public:
  Network(double learningRate);

  ~Network();

  template <class T> void Data() {
    data_ = std::make_unique<T>();
    const int layers = data_->GetNumberOfLayers();
    const std::vector<int> &layout = data_->GetNetworkLayout();
    const std::vector<std::function<double(double)>> &activationFunctions =
        data_->GetActivationFunctions();
    const std::vector<std::function<double(double)>>
        &activationFunctionDerivatives =
            data_->GetActivationFunctionDerivatives();
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
  }

  void Train(int epochs);

  void Train();

  std::vector<double> Compute(const std::vector<double> &inputs);

  void SaveNetworkWeights(const std::string &path);

  bool LoadNetworkWeights(const std::string &path);

  void Test();

  void SetLearningRate(double lr);

private:
  void Train_(std::vector<double> &inputs, std::vector<double> &outputs);

  static double CalculateError_(const std::vector<double> &expectedOutputs,
                                const std::vector<double> &calculatedOutputs,
                                int outputCount);

  double TestNetwork_();

  void Backpropagation_(std::vector<double> &inputs,
                        std::vector<double> &outputs);
};
} // namespace axon
