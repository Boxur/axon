#pragma once
#include <functional>
#include <vector>

namespace axon {
/**
 * @brief NetworkData is a class Interface responsible for giving the network
 * the information it needs.
 *
 * NetworkData supplies:
 *  - Test data
 *  - Training data
 *  - network layout including the activation functions
 */
class NetworkData {
public:
  virtual ~NetworkData() = default;
  /**
   * @brief Responsible for providing the next set of training data
   *
   * @param inputs Vector to put the inputs in
   * @param outputs Vector to put the outputs in
   *
   * @returns
   *  - true if there was more data to give
   *  - false if there is no more data and it has to be reloaded
   */
  virtual bool GetNextTrainingData(std::vector<double> &inputs,
                                   std::vector<double> &outputs) = 0;
  /**
   * @brief Responsible for providing the exact same training data as last time
   *
   * The neural network assumes the data given by this method will be the exact
   * same as the last call of GetNextTrainingData so keep that in mind when
   * implementing some sort of noise addition
   *
   * @param inputs Vector to put the inputs in
   * @param outputs Vector to put the outputs in
   */
  virtual void GetSameTrainingData(std::vector<double> &inputs,
                                   std::vector<double> &outputs) = 0;

  /**
   * @brief Responsible for providing the next set of test data
   *
   * @param inputs Vector to put the inputs in
   * @param outputs Vector to put the outputs in
   *
   * @returns
   *  - true if there was more data to give
   *  - false if there is no more data and it has to be reloaded
   */
  virtual bool GetNextTestData(std::vector<double> &inputs,
                               std::vector<double> &outputs) = 0;
  /**
   * @brief Responsible for providing the exact same test data as last time
   *
   * The neural network assumes the data given by this method will be the exact
   * same as the last call of GetNextTestData so keep that in mind when
   * implementing some sort of noise addition
   *
   * @param inputs Vector to put the inputs in
   * @param outputs Vector to put the outputs in
   */
  virtual void GetSameTestData(std::vector<double> &inputs,
                               std::vector<double> &outputs) = 0;

  /**
   * @brief Loads the training dataset
   *
   * @returns
   *  - true if succesful
   *  - false if unsuccesful
   */
  virtual bool LoadTrainingData() { return true; };
  /**
   * @brief Loads the testing dataset
   *
   * @returns
   *  - true if succesful
   *  - false if unsuccesful
   */
  virtual bool LoadTestData() { return true; };

  /**
   * @brief Get the layout for the network to follow
   *
   * @return a vector that contains a list of layer sizes eg. a network with
   * layers 4->8->2 (with 4 beeing the input layer and 2 the output) will be
   * {4,8,2}
   */
  [[nodiscard]] const virtual std::vector<int> GetNetworkLayout() const = 0;
  /**
   * @brief Just get the number of layers
   *
   * @returns the number of layers
   */
  [[nodiscard]] const inline virtual int GetNumberOfLayers() const = 0;

  /**
   * @brief Get the list of activation functions
   *
   * @returns a vector containing the activation functions in the same order as
   * in GetNetworkLayout
   */
  const virtual std::vector<std::function<double(double)>> &
  GetActivationFunctions() const = 0;
  /**
   * @brief Get the list of activation function derivatives
   *
   * @returns a vector containing the activation function derivatives in
   * the same order as in GetNetworkLayout
   */
  const virtual std::vector<std::function<double(double)>> &
  GetActivationFunctionDerivatives() const = 0;
};
} // namespace axon
