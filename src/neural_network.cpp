#include "log.hpp"
#include "mnist_data.hpp"
#include "network.hpp"
#include <memory>
#include <unistd.h>

int main() {
  std::shared_ptr<axon::NetworkData> data = std::make_shared<MnistData>();
  axon::Network network(data, 0.000001f);
  network.LoadNetworkWeights("Assets/network.nn");
  axon::Log.Level(axon::Log.error | axon::Log.info);
  network.Train(1);
  network.Test();
  network.SaveNetworkWeights("Assets/network.nn");
}
