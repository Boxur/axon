#pragma once
#include <vector>

namespace axon 
{
  class NetworkData
  {
  public:
    virtual bool GetNextTrainingData(std::vector<double> &inputs, std::vector<double> &outputs) = 0;

    virtual void GetSameTrainingData(std::vector<double> &inputs, std::vector<double> &outputs) = 0;

    virtual bool GetNextTestData(std::vector<double> &inputs, std::vector<double> &outputs) = 0;

    virtual void GetSameTestData(std::vector<double> &inputs, std::vector<double> &outputs) = 0;
    
    virtual bool LoadTrainingData() = 0;

    virtual bool LoadTestData() = 0;

    virtual std::vector<int> GetNetworkLayout() = 0;

    virtual int GetNumberOfLayers() = 0;
  };
}
