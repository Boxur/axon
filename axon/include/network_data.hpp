#pragma once
#include <functional>
#include <vector>

namespace axon {
class NetworkData {
public:
  virtual bool GetNextTrainingData(std::vector<double> &inputs,
                                   std::vector<double> &outputs) = 0;
  virtual void GetSameTrainingData(std::vector<double> &inputs,
                                   std::vector<double> &outputs) = 0;

  virtual bool GetNextTestData(std::vector<double> &inputs,
                               std::vector<double> &outputs) = 0;
  virtual void GetSameTestData(std::vector<double> &inputs,
                               std::vector<double> &outputs) = 0;

  virtual bool LoadTrainingData() = 0;
  virtual bool LoadTestData() = 0;

  const virtual std::vector<int> &GetNetworkLayout() const = 0;
  const virtual int &GetNumberOfLayers() const = 0;

  const virtual std::vector<std::function<double(double)>> &
  GetActivationFunctions() const = 0;
  const virtual std::vector<std::function<double(double)>> &
  GetActivationFunctionDerivatives() const = 0;
};
} // namespace axon
