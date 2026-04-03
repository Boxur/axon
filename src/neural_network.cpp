#include "mnist_data.hpp"
#include "network.hpp"
#include "log.hpp"
#include <memory>

int main()
{
	std::shared_ptr<axon::NetworkData> data = std::make_shared<MnistData>();
  axon::Network network(data, 0.000001f);
	network.LoadNetwork("Assets/network.nn");
  axon::Log.Level(axon::Log.error | axon::Log.info);
	network.Train(10);
	network.Test();
	network.SaveNetwork("Assets/network.nn");
}

