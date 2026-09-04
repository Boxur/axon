#pragma once
#include <fstream>
#include <functional>
#include <vector>

namespace axon {
class Layer {
public:
  enum class DeltaMode { Diffrence, Multiplication };

private:
  int inputCount_;
  int outputCount_;
  std::vector<double> outputs_;
  std::vector<double> biases_;
  std::vector<double> weights_;
  std::vector<double> deltas_;

  const std::function<double(double)> activationFunction_;
  const std::function<double(double)> activationFunctionDerivative_;

public:
  Layer(int inputs, int outputs,
        std::function<double(double)> activationFunction,
        std::function<double(double)> activationFunctionDerivative);

  ~Layer();

  void InitWeights();

  std::vector<double> Compute(const std::vector<double> &inputs);

  void Delta(std::vector<double> &output, DeltaMode mode);

  void Descent(const std::vector<double> &inputs, double learningRate);

  double Random();

  void SaveLayer(std::fstream &file);

  void LoadLayer(std::fstream &file);

  std::vector<double> &GetOutputs();
};
} // namespace axon
