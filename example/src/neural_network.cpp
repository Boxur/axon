#include "axon/network.hpp"
#include "lg/logger.hpp"
#include "mnist_data.hpp"
#include <memory>
#include <unistd.h>

int main() {
  std::shared_ptr<axon::NetworkData> data = std::make_shared<MnistData>();
  axon::Network network(0.000001f);
  network.Data<MnistData>();
  network.LoadNetworkWeights("Assets/network.nn");
  lg::log.SetLevel((short)lg::Log::LogLevel::error |
                   (short)lg::Log::LogLevel::info);
  network.Train(10);
  network.Test();
  network.SaveNetworkWeights("Assets/network.nn");
}
