# Non Matching Test Suite

C++ application consisting of finite element programs testing different coupling strategies with via non matching techniques such as using quadrature rules on mesh intersections. Part of this work builds on top of the integration of the C++ library `CGAL` (https://www.cgal.org/) into `deal.II` (www.dealii.org). Currently, a `deal.II` version greater or equal than `9.4.2`, and `CGAL` versions greater than 5.5.2 are required, along with a C++17 compliant compiler.

![Screenshot](grids/mesh_3d.png)

![Screenshot](grids/iso_contour_3D_ns.png)

## How to compile and run

- `mkdir build && cd build`
- `cmake .` , or `cmake -DDEAL_II_DIR=/your/local/path/to/deal` 
- `make`, or `make -j N`, begin `N` the number of make jobs you may want to ask.


## Using Docker image [TODO]