#pragma once
#include <fstream>
#include <functional>
#include <vector>

namespace axon {
/**
 * @brief Layer class responsible for matrix multiplication, adding biases and
 * applying the activation functions
 *
 * Under normal use there is no reason for the user to create an instance of
 * this class
 */
class Layer {
public:
  /**
   * @brief The delta mode to use for the Delta calculation
   */
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
  /**
   * @brief Just the constructor
   *
   * @param inputs Number of layer inputs
   * @param outputs Number of layer outputs
   * @param buffer_size The size of input/output buffer
   * @param activationFunction The activation function to use during calculation
   * @param activationFunctionDerivative The activationFunctionDerivative to
   * use during gradient descent.
   */
  Layer(int inputs, int outputs, int buffer_size,
        std::function<double(double)> activationFunction,
        std::function<double(double)> activationFunctionDerivative);

  ~Layer();

  /**
   * @brief generate random starting weights
   */
  void InitWeights();

  /**
   * @brief Compute outputs based on inputs
   *
   * Multiplies the input vector by the weight matrix, adds biases and runs
   * everything through the activation function
   *
   * @param inputs Inputs to calculate the output for
   *
   * @returns The output vector.
   */
  const std::vector<double> &Compute(const std::vector<double> &inputs);

  /**
   * @brief Calculates by how much the layer was off for the gradient descent
   *
   *
   * Uses the saved outputs from Compute so that function needs to be called
   * before this one.
   *
   * @param output The output that should have been created
   * @param mode The delta mode to use
   */
  void Delta(std::vector<double> &output, DeltaMode mode);

  /**
   * @brief Perform the gradient descent based on the deltas computed by
   * the Delta method.
   *
   * @param inputs The original inputs the layer recieved
   * @param learningRate The learningRate rate (idk what more to explain)
   */
  void Descent(const std::vector<double> &inputs, double learningRate);

  /**
   * @brief Generates a random double in range <0,1>
   *
   * @returns the random value
   */
  double Random();

  /**
   * @brief Save layer weights to an fstream file
   *
   * @param file The file to write to
   */
  void SaveLayer(std::fstream &file);

  /**
   * @brief Load layer weights from and fstream file
   *
   * @param file The file to read from
   */
  void LoadLayer(std::fstream &file);

  /**
   * @brief get the outputs previously saved by Compute
   *
   * @returns The outputs vector
   */
  const std::vector<double> &GetOutputs();
};
} // namespace axon
