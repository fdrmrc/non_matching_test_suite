# Non Matching Test Suite

Collection of C++ finite element programs testing different coupling strategies with Non Matching techniques using composite quadrature rules on mesh intersections. This is done by integrating the C++ library `CGAL` (https://www.cgal.org/) into `deal.II` (www.dealii.org). Currently, all examples require a `deal.II` version greater or equal to `9.4.2`, a C++17-compliant compiler and a `CGAL` version greater than 5.5.2 installed.

![Screenshot](grids/Flower_interface.png)

## How to compile and run
We provide parameter files (.prm) for test cases, allowing the user to change rhs, boundary conditions, number of refinement cycles and mesh related parameters. To compile and run, move into the desired folder (e.g. `/LagrangeMultiplier/1d2d`).

- Set the `DEAL_II_DIR` to the directory where you built the branch above. Alternatively, you can pass it as one of the CMake flags

- `cmake .` , or `cmake -DDEAL_II_DIR=/your/local/path/to/deal` if you want to pass it as a flag

- Compile with `make ` (or `make -jN`)

- Run `./lagrange_multiplier disk_parameters.prm`

## Using Docker image [TODO]