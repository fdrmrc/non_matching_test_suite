# Non Matching Test Suite

Collection of C++ finite element programs testing different coupling strategies with Non Matching techniques and showing benefits of composite quadrature rules on mesh intersections. This is done by integrating the C++ library `CGAL` (https://www.cgal.org/) into `deal.II` (www.dealii.org). Currently, all examples are based on the developement branch https://github.com/luca-heltai/dealii/tree/Non_matching_tests and require C++17 standard and the latest version of `CGAL` installed.

![Screenshot](grids/Flower_interface.png)

## How to compile and run
We provide parameter files (.prm) for test cases, allowing the user to change rhs, boundary conditions, number of refinement cycles and mesh related parameters. To compile and run, move into the desired folder (e.g. `/LagrangeMultiplier/1d2d`).

- Set the `DEAL_II_DIR` to the directory where you built the branch above. Alternatively, you can pass it as one of the CMake flags

- `cmake .` , or `cmake -DDEAL_II_DIR=/your/local/path/to/deal` if you want to pass it as a flag

- Compile with `make ` (or `make -jN`)

- Run `./lagrange_multiplier disk_parameters.prm`

## Using Docker image [TODO]