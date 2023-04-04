// ---------------------------------------------------------------------
//
// Copyright (C) 2022 by the deal.II authors
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

#ifndef dealii_cgal_intersections_h
#define dealii_cgal_intersections_h

#include <deal.II/base/config.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/base/array_view.h>
#include <deal.II/base/point.h>

#include <deal.II/fe/mapping.h>

#include <deal.II/grid/tria.h>

DEAL_II_NAMESPACE_OPEN
namespace CGALWrappers {

/**
 * Get vertices of cell in CGAL ordering.
 *
 * @param cell A cell_iterator to a deal.II cell.
 * @param mapping Mapping object for the cell.
 * @return  Array of vertices in CGAL order.
 */
template <int dim, int spacedim>
std::vector<Point<spacedim>> get_vertices_in_cgal_order(
    const typename dealii::Triangulation<dim, spacedim>::cell_iterator &cell,
    const Mapping<dim, spacedim> &mapping) {
  // Elements have to be rectangular or simplices
  const unsigned int n_vertices = cell->n_vertices();
  Assert((n_vertices == ReferenceCells::get_hypercube<dim>().n_vertices()) ||
             (n_vertices == ReferenceCells::get_simplex<dim>().n_vertices()),
         ExcNotImplemented());

  std::vector<Point<spacedim>> ordered_vertices(n_vertices);
  std::copy_n(mapping.get_vertices(cell).begin(), n_vertices,
              ordered_vertices.begin());

  if (ReferenceCell::n_vertices_to_type(dim, n_vertices) ==
      ReferenceCells::Quadrilateral)
    std::swap(ordered_vertices[2], ordered_vertices[3]);

  return ordered_vertices;
}

/**
 * Get vertices of face in CGAL ordering
 *
 * @param cell A cell_iterator to a deal.II cell.
 * @param face_no The face number within the given cell.
 * @param mapping Mapping object for the cell.
 * @return Array of vertices in CGAL order.
 */
template <int dim, int spacedim>
std::vector<Point<spacedim>> get_vertices_in_cgal_order(
    const typename dealii::Triangulation<dim, spacedim>::cell_iterator &cell,
    const unsigned int face_no, const Mapping<dim, spacedim> &mapping) {
  // Elements have to be rectangular or simplices
  const unsigned int n_vertices = cell->face(face_no)->n_vertices();
  Assert(
      (n_vertices == ReferenceCells::get_hypercube<dim - 1>().n_vertices()) ||
          (n_vertices == ReferenceCells::get_simplex<dim - 1>().n_vertices()),
      ExcNotImplemented());

  std::vector<Point<spacedim>> ordered_vertices(n_vertices);
  std::copy_n(mapping.get_vertices(cell, face_no).begin(), n_vertices,
              ordered_vertices.begin());

  if (ReferenceCell::n_vertices_to_type(dim - 1, n_vertices) ==
      ReferenceCells::Quadrilateral)
    std::swap(ordered_vertices[2], ordered_vertices[3]);

  return ordered_vertices;
}

/**
 * Given two deal.II affine cells, compute their intersection and return a
 * vector of simplices, each one identified by an array of deal.II Points. All
 * the simplices together are a subdivision of the intersection. If cells are
 * non-affine, a geometrical error is introduced. If the
 * measure of one of the simplices is below a certain threshold which defaults
 * to 1e-9, then it is discarded. In case the two cells are disjoint, an empty
 * array is returned.
 *
 * @note If the two cells have different intrinsic dimensions, than the
 * first iterator is assumed to have with higher intrinsic dimension.
 * The same holds for the two Mapping objects.
 *
 *
 * @param cell0 Iterator to the first cell.
 * @param cell1 Iterator to the second cell.
 * @param mapping0 Mapping for the first cell.
 * @param mapping1 Mapping for the second cell.
 * @param tol Threshold to decide whether or not a simplex is included.
 * @return Vector of arrays, where each array identify a simplex by its
 * vertices.
 */
template <int dim0, int dim1, int spacedim>
std::vector<std::array<Point<spacedim>, dim1 + 1>>
compute_intersection_of_cells(
    const typename Triangulation<dim0, spacedim>::cell_iterator &cell0,
    const typename Triangulation<dim1, spacedim>::cell_iterator &cell1,
    const Mapping<dim0, spacedim> &mapping0,
    const Mapping<dim1, spacedim> &mapping1, const double tol = 1e-9);

/**
 * Same function as above, but working directly with vertices. It's assumed
 * that the first vertices are coming from a Triangulation<dim0,spacedim>,
 * while the other vertices are from a Triangulation<dim1,spacedim>, with
 * @p dim0 > @p dim1.
 *
 * @note The vertices have to be given in CGAL order.
 */
template <int dim0, int dim1, int spacedim>
std::vector<std::array<Point<spacedim>, dim1 + 1>>
compute_intersection_of_cells(const ArrayView<const Point<spacedim>> &vertices0,
                              const ArrayView<const Point<spacedim>> &vertices1,
                              const double tol = 1e-9);

} // namespace CGALWrappers

/**
 *
 * Given two cell_iterator objects to two intersecting cells, computes a
 * Quadrature rule of degree `degree` formula by splitting the intersection
 * into simplices.
 *
 * @param [in] cell0 A cell_iterator to the first deal.II cell.
 * @param [in] cell1 A cell_iterator to the second deal.II cell.
 * @param [in] degree The degree of accuracy you wish to get for the global
 * quadrature formula.
 * @param [in] mapping0 Mapping object for the first cell.
 * @param [in] mapping1 Mapping object for the first cell.
 * @return [out] Quadrature<spacedim> The global quadrature rule on the
 * polygon/polyhedron.
 */
template <int dim0, int dim1, int spacedim>
Quadrature<spacedim> compute_quadrature_on_intersection(
    const typename Triangulation<dim0, spacedim>::cell_iterator &cell0,
    const typename Triangulation<dim1, spacedim>::cell_iterator &cell1,
    const unsigned int degree, const Mapping<dim0, spacedim> &mapping0,
    const Mapping<dim1, spacedim> &mapping1, const double tol = 1e-9);

DEAL_II_NAMESPACE_CLOSE

#endif