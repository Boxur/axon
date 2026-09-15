#pragma once
#include "axon/layer.hpp"
#include "axon/network_data.hpp"

#include <algorithm>
#include <assert.h>
#include <memory>
#include <vector>

namespace axon {
/**
 * @brief Network manages the neural net learning process
 *
 * Loads data from a NetworkData implementation, trains the network on said
 * data using regression, can calculate outputs based on inputs and is
 * responsible for saving and loading the data.
 */
class Network {
private:
  double learningRate_;
  int inputCount_;
  int outputCount_;
  int layerCount_;
  int biggestLayer_;
  std::vector<Layer> layers_;
  std::unique_ptr<NetworkData> data_;

  double precission_;

public:
  /**
   * @brief Constructor setting up some default values
   *
   * Please call the Data<T>(); method with T beeing an implementation of
   * the NetworkData class to fully finish setting up.
   *
   * @param learningRate The base learning rate for the network.
   */
  Network(double learningRate);

  /**
   * @brief Just the destructor to clean up.
   */
  ~Network();

  /**
   * @brief Setting up the data for the network to learn from.
   *
   * @tparam T Implementation of the NetworkData class to get the data from.
   */
  template <class T> void Data() {
    data_ = std::make_unique<T>();
    const int layers = data_->GetNumberOfLayers();
    const std::vector<int> &layout = data_->GetNetworkLayout();
    const std::vector<std::function<double(double)>> &activationFunctions =
        data_->GetActivationFunctions();
    const std::vector<std::function<double(double)>>
        &activationFunctionDerivatives =
            data_->GetActivationFunctionDerivatives();
    biggestLayer_ = *std::max_element(layout.begin(), layout.end());
    layers_.clear();
    for (int i = 0; i < layers - 1; i++) {
      layers_.push_back(Layer(layout[i], layout[i + 1], biggestLayer_,
                              activationFunctions[i],
                              activationFunctionDerivatives[i]));
    }
    inputCount_ = layout[0];
    outputCount_ = layout[layers - 1];
    layerCount_ = layers - 1;
  }

  /**
   * @brief Train for epochs iterations
   *
   * @param epochs The number of epochs to train for
   */
  void Train(int epochs);

  /**
   * @brief Train for a single epoch
   */
  void Train();

  /**
   * @brief Compute outputs for said inputs
   *
   * @param inputs The inputs to Compute the values for
   *
   * @returns a vector with outputs
   */
  [[nodiscard]] std::vector<double> Compute(const std::vector<double> &inputs);

  /**
   * @brief Save weights to a file
   *
   * @param path Path to the file
   */
  void SaveNetworkWeights(const std::string &path);

  /**
   * @brief Load weights to a file
   *
   * @param path Path to the file
   */
  bool LoadNetworkWeights(const std::string &path);

  /**
   * @brief Test the network efficiency
   *
   * Loads the test data and runs it through the network logging the average
   * MSE (mean squared error)
   */
  void Test();

  /**
   * @brief set the learning rate
   *
   * @param lr Learning rate to set to
   */
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
