# Error Simulations on Projected Quantum Kernel Features

This code introduces errors in TensorFlow Quantum's [Quantum Data tutorial](https://www.tensorflow.org/quantum/tutorials/quantum_data). It was used in the author's master's thesis [Error Simulations on Projected Quantum Kernels](https://helda.helsinki.fi/items/1b2ec72c-026d-48bb-9c7c-5303a05f04a3). The code may be run on different error rates.

In summary, the code simulates a quantum embedding of training data rows and computes their classical projections. The resulting dataset is used to produce new binary labeling such that a classical neural network learns the labeling with fewer training data rows if it uses the quantum-processed data instead of the original dataset as training data. The error simulation allows us to observe how this quantum prediction advantage holds when the error rates of contemporary quantum computers are introduced to the setup. For a more detailed description, please refer to the Quantum Data tutorial or Chapter 3 in the thesis linked above.

The simulation is heavy on RAM and SWAP memory. The number of qubits is parametrized so that the memory usage may be controlled. In addition, there are methods to save intermediate simulation and model training results. These earlier results may be used again in case of a code crash due to insufficient memory.

The packages in the Quantum Data tutorial have been updated, and the packages used in this code are older. For more details, please see Version and Environment below.

## Version and Environment

We ran the simulation code on a standard PC with Nvidia GeForce GTX 1080 GPU running Ubuntu 22.04 LTS. The simulation code uses an older version of TensorFlow and TensorFlow Quantum (0.7.2 and 2.7.0, respectively), and the simulation code does not necessarily run on the newest TensorFlow installations. We used Python 3.8, for which the packages were available. We point out that the Quantum Data Tutorial of TensorFlow was updated to use newer package versions (2.15.0 and 0.7.3) after our experiment.

TensorFlow should be able to run on CPU resources alone without GPU support. That was the case when we ran the original Quantum Data tutorial. Nevertheless, after we introduced the noise simulations, the code started to require Nvidia developer packages. We used the TensorFlow installation with GPU support and installed the Nvidia CUDA Toolkit of generation 12 and a Nvidia neural network package cuDNN version 8.9.7. The code explicitly required the 8th generation cuDNN to be used.

We installed the packages through pip. There is a Conda channel that provides the 0.7.2 version of TensorFlow with GPU support but not TensorFlow Quantum. The combination of installing TensorFlow with Conda and TensorFlow Quantum with pip did not work.
