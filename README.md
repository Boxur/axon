# axon
A neural network implemented from scratch designed in a way to make using it
straight forward

[![Build](https://github.com/Boxur/axon/actions/workflows/documentation.yml/badge.svg)](https://github.com/boxur/axon/actions)
[![Documentation](https://img.shields.io/badge/docs-online-blue)](https://axon.documentation.boxur.org/)
[![License](https://img.shields.io/github/license/boxur/axon)](https://github.com/Boxur/axon/blob/main/LICENSE)
[![Top Language](https://img.shields.io/github/languages/top/boxur/axon)](https://github.com/boxur/axon)

## Features

 - Calculating outputs based on inputs, weights, and activation functions
 - Performing gradient descent to learn
 - Loading and saving weights to a file

## Requirements

 - C++23 compiler
 - CMake 3.14+
 - Git

## Instalation

Clone from github:
```bash
git clone https://github.com/Boxur/axon external/axon
```

### CMake

Add to your project in CMake
```cmake
add_subdirectory(external/axon)

target_link_libraries(project PRIVATE axon)
```

## Usage

Create an implementation of the NetworkData class in [include/axon/NetworkData.hpp](https://github.com/Boxur/axon/blob/main/include/axon/network_data.hpp).
Create a Network class instance and pass in the implementation as a template
argument of Network::Data\<T\>\(\).

This can be see in the fully working example in [example/](https://github.com/Boxur/axon/blob/main/example).

## Contributing

Contributions, bug reports, and suggestions are welcome.

Before submitting a pull request, please ensure that the project builds
successfully

## License

This project is licensed under the MIT License.
See [LICENSE](https://github.com/Boxur/axon/blob/main/LICENSE) for details.
