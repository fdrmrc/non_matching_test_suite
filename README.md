# Non Matching Test Suite


Collection of C++ finite element programs showing and testing different coupling strategies for Non matching techniques.

## How to compile and run

All the tests are based on the branch https://github.com/luca-heltai/dealii/tree/Non_matching_tests and require C++17 and `CGAL`.

Move to the folder of one test (for instance, `/DistributedLagrangeMultiplier`)

- Set the `DEAL_II_DIR` to the directory where you built the branch above. By default we currently have in `CMakeLists.txt`: `SET(DEAL_II_DIR "/workspace/dealii/build")`

- `cmake .` 

- Compile with `make ` (or `make -jN`)

- Run `./distributed_lagrange parameters.prm`

## Using Docker image [TODO]