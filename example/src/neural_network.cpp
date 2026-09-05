#include "logger.hpp"
#include "mnist_data.hpp"
#include "network.hpp"
#include <memory>
#include <unistd.h>

int main() {
  std::shared_ptr<axon::NetworkData> data = std::make_shared<MnistData>();
  axon::Network network(0.000001f);
  network.Data<MnistData>();
  network.LoadNetworkWeights("Assets/network.nn");
  logger::Logger::SetLogLevel((short)logger::Logger::LogLevel::error |
                              (short)logger::Logger::LogLevel::info);
  network.Train(10);
  network.Test();
  network.SaveNetworkWeights("Assets/network.nn");
}
