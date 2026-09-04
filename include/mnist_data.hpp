#pragma once
#include "network_data.hpp"
#include <fstream>
#include <functional>
#include <stdlib.h>
#include <vector>
class MnistData : public axon::NetworkData {
private:
  char *skip_;
  // 11 outputs for the 10 digits and noise
  const std::vector<int> networkLayout_;
  static constexpr int numberOfLayers_ = 4;
  static constexpr int trainingSetCount_ = 60000;
  const std::vector<std::function<double(double)>> activationFunctions_;
  const std::vector<std::function<double(double)>>
      activationFunctionDerivatives_;

  std::vector<double> trainingInputArray_;
  std::vector<double> trainingOutputArray_;
  std::vector<double> testInputArray_;
  std::vector<double> testOutputArray_;

  std::fstream trainingInputs_;
  std::fstream trainingOutputs_;
  std::fstream testInputs_;
  std::fstream testOutputs_;

public:
  MnistData();

  ~MnistData();

  bool GetNextTrainingData(std::vector<double> &inputs,
                           std::vector<double> &outputs) override;

  void GetSameTrainingData(std::vector<double> &inputs,
                           std::vector<double> &outputs) override;

  bool GetNextTestData(std::vector<double> &inputs,
                       std::vector<double> &outputs) override;

  void GetSameTestData(std::vector<double> &inputs,
                       std::vector<double> &outputs) override;

  bool LoadTrainingData() override;

  bool LoadTestData() override;

  constexpr const int GetNumberOfLayers() const override;

  const std::vector<int> GetNetworkLayout() const override;

  const std::vector<std::function<double(double)>> &
  GetActivationFunctions() const override;

  const std::vector<std::function<double(double)>> &
  GetActivationFunctionDerivatives() const override;
};
