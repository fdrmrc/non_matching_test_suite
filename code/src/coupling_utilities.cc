// ---------------------------------------------------------------------
//
// Copyright (C) 2018 - 2022 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.md at
// the top level directory of deal.II.
//
// ---------------------------------------------------------------------

#include "coupling_utilities.h"
#include <deal.II/lac/sparse_matrix.h>

namespace dealii {
namespace NonMatching {

template <int dim0, int dim1, int spacedim>
std::vector<std::tuple<typename Triangulation<dim0, spacedim>::cell_iterator,
                       typename Triangulation<dim1, spacedim>::cell_iterator,
                       Quadrature<spacedim>>>
collect_quadratures_on_overlapped_grids(
    const GridTools::Cache<dim0, spacedim> &space_cache,
    const GridTools::Cache<dim1, spacedim> &immersed_cache,
    const unsigned int degree, const double tol) {
  AssertThrow(dim1 <= dim0, ExcMessage("Intrinsic dimension of the immersed "
                                       "object must be smaller than dim0."));
  AssertThrow(degree > 0, ExcMessage("Invalid quadrature degree."));
  Assert((dim1 <= dim0) && (dim0 <= spacedim),
         ExcMessage("This function can only work if dim1<=dim0<=spacedim"));
  std::vector<std::tuple<typename Triangulation<dim0, spacedim>::cell_iterator,
                         typename Triangulation<dim1, spacedim>::cell_iterator,
                         Quadrature<spacedim>>>
      cells_with_quadratures;

  const auto &space_tree =
      space_cache.get_locally_owned_cell_bounding_boxes_rtree();

  // The immersed tree *must* contain all cells, also the non-locally owned
  // ones.
  const auto &immersed_tree = immersed_cache.get_cell_bounding_boxes_rtree();

  // references to triangulations' info (cp cstrs marked as delete)
  const auto &mapping0 = space_cache.get_mapping();
  const auto &mapping1 = immersed_cache.get_mapping();
  namespace bgi = boost::geometry::index;
  // Whenever the BB space_cell intersects the BB of an embedded cell,
  // store the space_cell in the set of intersected_cells
  for (const auto &[immersed_box, immersed_cell] : immersed_tree) {
    for (const auto &[space_box, space_cell] :
         space_tree | bgi::adaptors::queried(bgi::intersects(immersed_box))) {
      const auto test_intersection = compute_quadrature_on_intersection(
          space_cell, immersed_cell, degree, mapping0, mapping1, tol);

      const auto &weights = test_intersection.get_weights();
      const double area = std::accumulate(weights.begin(), weights.end(), 0.0);
      if (area > tol) // non-trivial intersection
      {
        cells_with_quadratures.push_back(
            std::make_tuple(space_cell, immersed_cell, test_intersection));
      }
    }
  }
  return cells_with_quadratures;
}

// Explicit instantiations
template std::vector<
    std::tuple<typename Triangulation<2, 2>::cell_iterator,
               typename Triangulation<1, 2>::cell_iterator, Quadrature<2>>>
collect_quadratures_on_overlapped_grids(
    const GridTools::Cache<2, 2> &space_cache,
    const GridTools::Cache<1, 2> &immersed_cache, const unsigned int degree,
    const double tol);

template std::vector<
    std::tuple<typename Triangulation<2, 2>::cell_iterator,
               typename Triangulation<2, 2>::cell_iterator, Quadrature<2>>>
collect_quadratures_on_overlapped_grids(
    const GridTools::Cache<2, 2> &space_cache,
    const GridTools::Cache<2, 2> &immersed_cache, const unsigned int degree,
    const double tol);

template std::vector<
    std::tuple<typename Triangulation<3, 3>::cell_iterator,
               typename Triangulation<2, 3>::cell_iterator, Quadrature<3>>>
collect_quadratures_on_overlapped_grids(
    const GridTools::Cache<3, 3> &space_cache,
    const GridTools::Cache<2, 3> &immersed_cache, const unsigned int degree,
    const double tol);

template std::vector<
    std::tuple<typename Triangulation<3, 3>::cell_iterator,
               typename Triangulation<3, 3>::cell_iterator, Quadrature<3>>>
collect_quadratures_on_overlapped_grids(
    const GridTools::Cache<3, 3> &space_cache,
    const GridTools::Cache<3, 3> &immersed_cache, const unsigned int degree,
    const double tol);

} // namespace NonMatching
} // namespace dealii
