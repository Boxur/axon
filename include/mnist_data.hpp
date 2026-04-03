#pragma once
#include "network_data.hpp"
#include <fstream>
#include <stdlib.h>
#include <vector>
class MnistData : public axon::NetworkData
{
private:
	// 11 outputs for the 10 digits and noise
	std::vector<int> networkLayout_;
	int numberOfLayers_ = 4;
	int trainingSetCount_ = 60000;

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

	bool GetNextTrainingData(std::vector<double> &inputs, std::vector<double> &outputs) override;

	void GetSameTrainingData(std::vector<double> &inputs, std::vector<double> &outputs) override;

	bool GetNextTestData(std::vector<double> &inputs, std::vector<double> &outputs) override;

	void GetSameTestData(std::vector<double> &inputs, std::vector<double> &outputs) override;

 	bool LoadTrainingData() override;

	bool LoadTestData() override;

	std::vector<int> GetNetworkLayout() override;

	int GetNumberOfLayers() override;
};
