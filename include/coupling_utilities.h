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

#ifndef coupling_utilities
#define coupling_utilities

#include "intersections.h"
#include "non_matching_utilities.h"
#include <deal.II/grid/grid_tools.h>
#include <deal.II/non_matching/coupling.h>

namespace dealii {
namespace NonMatching {

template <int dim0, int dim1, int spacedim, typename Sparsity, typename number>
void create_coupling_sparsity_pattern_with_exact_intersections(
    const std::vector<std::tuple<
        typename dealii::Triangulation<dim0, spacedim>::cell_iterator,
        typename dealii::Triangulation<dim1, spacedim>::cell_iterator,
        dealii::Quadrature<spacedim>>> &intersections_info,
    const DoFHandler<dim0, spacedim> &space_dh,
    const DoFHandler<dim1, spacedim> &immersed_dh, Sparsity &sparsity,
    const AffineConstraints<number> &constraints,
    const ComponentMask &space_comps, const ComponentMask &immersed_comps,
    const AffineConstraints<number> &immersed_constraints) {
#ifdef DEAL_II_WITH_CGAL
  AssertDimension(sparsity.n_rows(), space_dh.n_dofs());
  AssertDimension(sparsity.n_cols(), immersed_dh.n_dofs());
  Assert(dim1 <= dim0,
         ExcMessage("This function can only work if dim1 <= dim0"));
  Assert((dim1 <= dim0) && (dim0 <= spacedim),
         ExcMessage("This function can only work if dim1<=dim0<=spacedim"));
  Assert((dynamic_cast<
              const parallel::distributed::Triangulation<dim1, spacedim> *>(
              &immersed_dh.get_triangulation()) == nullptr),
         ExcNotImplemented());

  const auto &space_fe = space_dh.get_fe();
  const auto &immersed_fe = immersed_dh.get_fe();
  const unsigned int n_dofs_per_space_cell = space_fe.n_dofs_per_cell();
  const unsigned int n_dofs_per_immersed_cell = immersed_fe.n_dofs_per_cell();
  const unsigned int n_space_fe_components = space_fe.n_components();
  const unsigned int n_immersed_fe_components = immersed_fe.n_components();
  std::vector<types::global_dof_index> space_dofs(n_dofs_per_space_cell);
  std::vector<types::global_dof_index> immersed_dofs(n_dofs_per_immersed_cell);

  const ComponentMask space_c =
      (space_comps.size() == 0 ? ComponentMask(n_space_fe_components, true)
                               : space_comps);

  const ComponentMask immersed_c =
      (immersed_comps.size() == 0
           ? ComponentMask(n_immersed_fe_components, true)
           : immersed_comps);

  AssertDimension(space_c.size(), n_space_fe_components);
  AssertDimension(immersed_c.size(), n_immersed_fe_components);

  // Global 2 Local indices
  std::vector<unsigned int> space_gtl(n_space_fe_components);
  std::vector<unsigned int> immersed_gtl(n_immersed_fe_components);
  for (unsigned int i = 0, j = 0; i < n_space_fe_components; ++i) {
    if (space_c[i])
      space_gtl[i] = j++;
  }

  for (unsigned int i = 0, j = 0; i < n_immersed_fe_components; ++i) {
    if (immersed_c[i])
      immersed_gtl[i] = j++;
  }

  Table<2, bool> dof_mask(n_dofs_per_space_cell, n_dofs_per_immersed_cell);
  dof_mask.fill(false); // start off by assuming they don't couple

  for (unsigned int i = 0; i < n_dofs_per_space_cell; ++i) {
    const auto comp_i = space_fe.system_to_component_index(i).first;
    if (space_gtl[comp_i] != numbers::invalid_unsigned_int) {
      for (unsigned int j = 0; j < n_dofs_per_immersed_cell; ++j) {
        const auto comp_j = immersed_fe.system_to_component_index(j).first;
        if (immersed_gtl[comp_j] == space_gtl[comp_i]) {
          dof_mask(i, j) = true;
        }
      }
    }
  }

  const bool dof_mask_is_active = dof_mask.n_rows() == n_dofs_per_space_cell &&
                                  dof_mask.n_cols() == n_dofs_per_immersed_cell;

  // Whenever the BB space_cell intersects the BB of an embedded cell, those
  // DoFs have to be recorded

  for (const auto &it : intersections_info) {
    const auto &space_cell = std::get<0>(it);
    const auto &immersed_cell = std::get<1>(it);
    typename DoFHandler<dim0, spacedim>::cell_iterator space_cell_dh(
        *space_cell, &space_dh);
    typename DoFHandler<dim1, spacedim>::cell_iterator immersed_cell_dh(
        *immersed_cell, &immersed_dh);

    space_cell_dh->get_dof_indices(space_dofs);
    immersed_cell_dh->get_dof_indices(immersed_dofs);

    if (dof_mask_is_active) {
      for (unsigned int i = 0; i < n_dofs_per_space_cell; ++i) {
        const unsigned int comp_i =
            space_dh.get_fe().system_to_component_index(i).first;
        if (space_gtl[comp_i] != numbers::invalid_unsigned_int) {
          for (unsigned int j = 0; j < n_dofs_per_immersed_cell; ++j) {
            const unsigned int comp_j =
                immersed_dh.get_fe().system_to_component_index(j).first;
            if (space_gtl[comp_i] == immersed_gtl[comp_j]) {
              // local_cell_matrix(i, j) +=
              constraints.add_entries_local_to_global(
                  {space_dofs[i]}, immersed_constraints, {immersed_dofs[j]},
                  sparsity, true);
            }
          }
        }
      }
    } else {
      constraints.add_entries_local_to_global(space_dofs, immersed_constraints,
                                              immersed_dofs, sparsity, true,
                                              dof_mask);
    }
  }

#else
  (void)intersections_info;
  (void)space_dh;
  (void)immersed_dh;
  (void)sparsity;
  (void)constraints;
  (void)space_comps;
  (void)immersed_comps;
  (void)immersed_constraints;
  Assert(false, ExcMessage("This function needs CGAL installed to work."));
#endif
}

template <int dim0, int dim1, int spacedim, typename Matrix>
void create_coupling_mass_matrix_with_exact_intersections(
    const DoFHandler<dim0, spacedim> &space_dh,
    const DoFHandler<dim1, spacedim> &immersed_dh,
    const std::vector<
        std::tuple<typename Triangulation<dim0, spacedim>::cell_iterator,
                   typename Triangulation<dim1, spacedim>::cell_iterator,
                   Quadrature<spacedim>>> &cells_and_quads,
    Matrix &matrix,
    const AffineConstraints<typename Matrix::value_type> &space_constraints,
    const ComponentMask &space_comps, const ComponentMask &immersed_comps,
    const Mapping<dim0, spacedim> &space_mapping,
    const Mapping<dim1, spacedim> &immersed_mapping,
    const AffineConstraints<typename Matrix::value_type>
        &immersed_constraints) {
#ifdef DEAL_II_WITH_CGAL
  AssertDimension(matrix.m(), space_dh.n_dofs());
  AssertDimension(matrix.n(), immersed_dh.n_dofs());
  Assert((dim1 <= dim0) && (dim0 <= spacedim),
         ExcMessage("This function can only work if dim1<=dim0<=spacedim"));
  Assert((dynamic_cast<
              const parallel::distributed::Triangulation<dim1, spacedim> *>(
              &immersed_dh.get_triangulation()) == nullptr),
         ExcMessage("The immersed triangulation can only be a "
                    "parallel::shared::triangulation"));

  const auto &space_fe = space_dh.get_fe();
  const auto &immersed_fe = immersed_dh.get_fe();

  const unsigned int n_dofs_per_space_cell = space_fe.n_dofs_per_cell();
  const unsigned int n_dofs_per_immersed_cell = immersed_fe.n_dofs_per_cell();

  const unsigned int n_space_fe_components = space_fe.n_components();
  const unsigned int n_immersed_fe_components = immersed_fe.n_components();

  FullMatrix<typename Matrix::value_type> local_cell_matrix(
      n_dofs_per_space_cell, n_dofs_per_immersed_cell);
  // DoF indices
  std::vector<types::global_dof_index> local_space_dof_indices(
      n_dofs_per_space_cell);
  std::vector<types::global_dof_index> local_immersed_dof_indices(
      n_dofs_per_immersed_cell);

  const ComponentMask space_c =
      (space_comps.size() == 0 ? ComponentMask(n_space_fe_components, true)
                               : space_comps);
  const ComponentMask immersed_c =
      (immersed_comps.size() == 0
           ? ComponentMask(n_immersed_fe_components, true)
           : immersed_comps);

  AssertDimension(space_c.size(), n_space_fe_components);
  AssertDimension(immersed_c.size(), n_immersed_fe_components);

  std::vector<unsigned int> space_gtl(n_space_fe_components,
                                      numbers::invalid_unsigned_int);
  std::vector<unsigned int> immersed_gtl(n_immersed_fe_components,
                                         numbers::invalid_unsigned_int);
  for (unsigned int i = 0, j = 0; i < n_space_fe_components; ++i) {
    if (space_c[i])
      space_gtl[i] = j++;
  }

  for (unsigned int i = 0, j = 0; i < n_immersed_fe_components; ++i) {
    if (immersed_c[i])
      immersed_gtl[i] = j++;
  }

  Table<2, bool> dof_mask(n_dofs_per_space_cell, n_dofs_per_immersed_cell);
  dof_mask.fill(false); // start off by assuming they don't couple

  for (unsigned int i = 0; i < n_dofs_per_space_cell; ++i) {
    const auto comp_i = space_fe.system_to_component_index(i).first;
    if (space_gtl[comp_i] != numbers::invalid_unsigned_int) {
      for (unsigned int j = 0; j < n_dofs_per_immersed_cell; ++j) {
        const auto comp_j = immersed_fe.system_to_component_index(j).first;
        if (immersed_gtl[comp_j] == space_gtl[comp_i]) {
          dof_mask(i, j) = true;
        }
      }
    }
  }

  // Loop over vector of tuples, and gather everything together
  for (const auto &infos : cells_and_quads) {
    const auto &[space_cell, embedded_cell, quad_formula] = infos;
    // std::cout << "Space cell: " << space_cell->active_cell_index()
    //           << std::endl;
    // std::cout << "Immersed cell: " << embedded_cell->active_cell_index()
    //           << "on the boundary? " << embedded_cell->at_boundary()
    //           << std::endl;

    local_cell_matrix = typename Matrix::value_type();

    const unsigned int n_quad_pts = quad_formula.size();
    const auto &real_qpts = quad_formula.get_points();
    std::vector<Point<dim0>> ref_pts_space(n_quad_pts);
    std::vector<Point<dim1>> ref_pts_immersed(n_quad_pts);

    space_mapping.transform_points_real_to_unit_cell(space_cell, real_qpts,
                                                     ref_pts_space);
    // std::cout << "Space indietro fatto" << std::endl;
    // if (!(dynamic_cast<const MappingQ<dim1, spacedim> *>(
    //         &immersed_mapping) == nullptr))
    //   {
    //     std::cout << "dynamic cast passato" << std::endl;
    //     immersed_mapping.transform_points_real_to_unit_cell(
    //       embedded_cell, real_qpts, ref_pts_immersed);
    //   }
    // else
    //   {
    // std::cout << "dynamic cast NON passato" << std::endl;
    // for (unsigned int i = 0; i < 2; ++i)
    //   {
    //     std::cout << "cella number: " <<
    //     embedded_cell->active_cell_index()
    //               << " has vertex " << embedded_cell->vertex(i)
    //               << std::endl;
    //   }

    // for (const auto &cell_test :
    //      immersed_dh.get_triangulation().active_cell_iterators())
    //   {
    //     std::cout << "Cella number: " << cell_test->active_cell_index()
    //               << std::endl;
    //     for (const auto &x : immersed_mapping.get_vertices(cell_test))
    //       {
    //         std::cout << "Mapped vertex: " << x << std::endl;

    //         std::cout
    //           << "Mapped BACK vertex: "
    //           << immersed_mapping.transform_real_to_unit_cell(cell_test,
    //           x)
    //           << std::endl;
    //       }
    //   }

    for (unsigned int q = 0; q < n_quad_pts; ++q) {
      ref_pts_immersed[q] = immersed_mapping.transform_real_to_unit_cell(
          embedded_cell, real_qpts[q]);
    }
    // }

    // std::cout << "Show unit points embedded" << std::endl;
    // for (const auto &p : ref_pts_immersed)
    //   std::cout << p << std::endl;

    // std::cout << "Immersed indietro fatto" << std::endl;
    const auto &JxW = quad_formula.get_weights();
    // std::cout << "Jacobiani presi con size:" << JxW.size() << std::endl;
    for (unsigned int q = 0; q < n_quad_pts; ++q) {
      for (unsigned int i = 0; i < n_dofs_per_space_cell; ++i) {
        const unsigned int comp_i =
            space_dh.get_fe().system_to_component_index(i).first;
        if (space_gtl[comp_i] != numbers::invalid_unsigned_int) {
          for (unsigned int j = 0; j < n_dofs_per_immersed_cell; ++j) {
            const unsigned int comp_j =
                immersed_dh.get_fe().system_to_component_index(j).first;
            if (space_gtl[comp_i] == immersed_gtl[comp_j]) {
              local_cell_matrix(i, j) +=
                  space_fe.shape_value(i, ref_pts_space[q]) *
                  immersed_fe.shape_value(j, ref_pts_immersed[q]) * JxW[q];
            }
          }
        }
      }
    }
    // std::cout << "Assemblato" << std::endl;
    typename DoFHandler<dim0, spacedim>::cell_iterator space_cell_dh(
        *space_cell, &space_dh);
    // std::cout << "DoFHandler space fatto" << std::endl;
    typename DoFHandler<dim1, spacedim>::cell_iterator immersed_cell_dh(
        *embedded_cell, &immersed_dh);
    // std::cout << "DoFHandler immerso fatto" << std::endl;

    space_cell_dh->get_dof_indices(local_space_dof_indices);
    immersed_cell_dh->get_dof_indices(local_immersed_dof_indices);

    // std::cout << "DoFIndices fatti" << std::endl;
    space_constraints.distribute_local_to_global(
        local_cell_matrix, local_space_dof_indices, immersed_constraints,
        local_immersed_dof_indices, matrix);
    // std::cout << "Distribuiti" << std::endl;
  }
  matrix.compress(VectorOperation::add);
#else
  (void)space_dh;
  (void)immersed_dh;
  (void)cells_and_quads;
  (void)matrix;
  (void)space_constraints;
  (void)space_comps;
  (void)immersed_comps;
  (void)space_mapping;
  (void)immersed_mapping;
  (void)immersed_constraints;
  Assert(false, ExcMessage("This function needs CGAL installed to work."));

#endif
}

template <int dim0, int dim1, int spacedim, typename Matrix>
void assemble_nitsche_with_exact_intersections(
    const DoFHandler<dim0, spacedim> &space_dh,
    const std::vector<std::tuple<
        typename dealii::Triangulation<dim0, spacedim>::cell_iterator,
        typename dealii::Triangulation<dim1, spacedim>::cell_iterator,
        dealii::Quadrature<spacedim>>> &cells_and_quads,
    Matrix &matrix,
    const AffineConstraints<typename Matrix::value_type> &space_constraints,
    const ComponentMask &space_comps,
    const Mapping<dim0, spacedim> &space_mapping,
    const Function<spacedim, typename Matrix::value_type> &nitsche_coefficient,
    const double penalty) {
#ifdef DEAL_II_WITH_CGAL
  AssertDimension(matrix.m(), space_dh.n_dofs());
  AssertDimension(matrix.n(), space_dh.n_dofs());
  Assert((dim1 <= dim0) && (dim0 <= spacedim),
         ExcMessage("This function can only work if dim1<=dim0<=spacedim"));
  Assert(cells_and_quads.size() > 0,
         ExcMessage("The background and immersed mesh must overlap in order to "
                    "use this function."));
  const auto &space_fe = space_dh.get_fe();
  const unsigned int n_dofs_per_space_cell = space_fe.n_dofs_per_cell();
  const unsigned int n_space_fe_components = space_fe.n_components();
  std::vector<unsigned int> space_gtl(n_space_fe_components,
                                      numbers::invalid_unsigned_int);
  // DoF indices
  std::vector<types::global_dof_index> local_space_dof_indices(
      n_dofs_per_space_cell);

  const ComponentMask space_c =
      (space_comps.size() == 0 ? ComponentMask(n_space_fe_components, true)
                               : space_comps);
  AssertDimension(space_c.size(), n_space_fe_components);

  for (unsigned int i = 0, j = 0; i < n_space_fe_components; ++i) {
    if (space_c[i])
      space_gtl[i] = j++;
  }

  FullMatrix<typename Matrix::value_type> local_cell_matrix(
      n_dofs_per_space_cell, n_dofs_per_space_cell);
  // Loop over vector of tuples, and gather everything together
  double h;
  for (const auto &infos : cells_and_quads) {
    const auto &[space_cell, embedded_cell, quad_formula] = infos;
    if (space_cell->is_active() && space_cell->is_locally_owned()) {
      local_cell_matrix = typename Matrix::value_type();

      const unsigned int n_quad_pts = quad_formula.size();
      const auto &real_qpts = quad_formula.get_points();
      std::vector<typename Matrix::value_type> nitsche_coefficient_values(
          n_quad_pts);
      nitsche_coefficient.value_list(real_qpts, nitsche_coefficient_values);

      std::vector<Point<dim0>> ref_pts_space(n_quad_pts);

      space_mapping.transform_points_real_to_unit_cell(space_cell, real_qpts,
                                                       ref_pts_space);

      h = space_cell->diameter();
      const auto &JxW = quad_formula.get_weights();
      for (unsigned int q = 0; q < n_quad_pts; ++q) {
        const auto &q_ref_point = ref_pts_space[q];
        for (unsigned int i = 0; i < n_dofs_per_space_cell; ++i) {
          const unsigned int comp_i =
              space_dh.get_fe().system_to_component_index(i).first;
          if (comp_i != numbers::invalid_unsigned_int) {
            for (unsigned int j = 0; j < n_dofs_per_space_cell; ++j) {
              const unsigned int comp_j =
                  space_dh.get_fe().system_to_component_index(j).first;
              if (space_gtl[comp_i] == space_gtl[comp_j]) {
                local_cell_matrix(i, j) +=
                    nitsche_coefficient_values[q] * (penalty / h) *
                    space_fe.shape_value(i, q_ref_point) *
                    space_fe.shape_value(j, q_ref_point) * JxW[q];
              }
            }
          }
        }
      }
      typename DoFHandler<dim0, spacedim>::cell_iterator space_cell_dh(
          *space_cell, &space_dh);

      space_cell_dh->get_dof_indices(local_space_dof_indices);
      space_constraints.distribute_local_to_global(
          local_cell_matrix, local_space_dof_indices, matrix);
    }
  }
#else
  (void)space_dh;
  (void)cells_and_quads;
  (void)matrix;
  (void)space_constraints;
  (void)space_comps;
  (void)space_mapping;
  (void)nitsche_coefficient;
  (void)penalty;
  Assert(false, ExcMessage("This function needs CGAL installed to work."));
#endif
}

template <int dim0, int dim1, int spacedim, typename Vector>
void create_nitsche_rhs_with_exact_intersections(
    const DoFHandler<dim0, spacedim> &space_dh,
    const std::vector<std::tuple<
        typename dealii::Triangulation<dim0, spacedim>::cell_iterator,
        typename dealii::Triangulation<dim1, spacedim>::cell_iterator,
        dealii::Quadrature<spacedim>>> &cells_and_quads,
    Vector &rhs_vector, const AffineConstraints<double> &space_constraints,
    const Mapping<dim0, spacedim> &space_mapping,
    const Function<spacedim, double> &rhs_function,
    const Function<spacedim, double> &coefficient, const double penalty) {
#ifdef DEAL_II_WITH_CGAL
  AssertDimension(rhs_vector.size(), space_dh.n_dofs());
  Assert(dim1 <= dim0, ExcMessage("This function can only work if dim1<=dim0"));

  const auto &space_fe = space_dh.get_fe();
  const unsigned int n_dofs_per_space_cell = space_fe.n_dofs_per_cell();
  dealii::Vector<double> local_rhs(n_dofs_per_space_cell);
  // DoF indices
  std::vector<types::global_dof_index> local_space_dof_indices(
      n_dofs_per_space_cell);

  // Loop over vector of tuples, and gather everything together
  double h;
  for (const auto &infos : cells_and_quads) {
    const auto &[space_cell, embedded_cell, quad_formula] = infos;

    if (space_cell->is_active()) {
      h = space_cell->diameter();
      local_rhs = 0.;
      // local_rhs = typename VectorType::value_type();

      const unsigned int n_quad_pts = quad_formula.size();
      const auto &real_qpts = quad_formula.get_points();
      std::vector<Point<dim0>> ref_pts_space(n_quad_pts);
      std::vector<double> rhs_function_values(n_quad_pts);
      rhs_function.value_list(real_qpts, rhs_function_values);

      std::vector<double> coefficient_values(n_quad_pts);
      coefficient.value_list(real_qpts, coefficient_values);

      space_mapping.transform_points_real_to_unit_cell(space_cell, real_qpts,
                                                       ref_pts_space);

      const auto &JxW = quad_formula.get_weights();
      for (unsigned int q = 0; q < n_quad_pts; ++q) {
        const auto &q_ref_point = ref_pts_space[q];
        for (unsigned int i = 0; i < n_dofs_per_space_cell; ++i) {
          local_rhs(i) += coefficient_values[q] * (penalty / h) *
                          space_fe.shape_value(i, q_ref_point) *
                          rhs_function_values[q] * JxW[q];
        }
      }
      typename DoFHandler<dim0, spacedim>::cell_iterator space_cell_dh(
          *space_cell, &space_dh);

      space_cell_dh->get_dof_indices(local_space_dof_indices);
      space_constraints.distribute_local_to_global(
          local_rhs, local_space_dof_indices, rhs_vector);
    }
  }
#else
  (void)space_dh;
  (void)cells_and_quads;
  (void)rhs_vector;
  (void)space_constraints;
  (void)space_mapping;
  (void)rhs_function;
  (void)coefficient;
  (void)penalty;
  Assert(false,
         ExcMessage("This function function needs CGAL installed to work."));
#endif
}

// Routines for cells intersections

/**
 *
 * Given two cached, arbitrarily overlapped grids, the following function
 * computes Quadrature rules on the intersection of the embedding grid with
 * the embedded one, of degree `degree`. The return type is a vector of tuples
 * `v`, where `v[i][0]` is an iterator to a cell of the embedding grid,
 * `v[i][1]` is an iterator to a cell of the embedded grid,
 * `v[i][2]` is a Quadrature formula to integrate over the intersection of the
 * two.
 *
 * The last parameter `tol` defaults to 1e-6, and can be used to discard small
 * intersections.
 *
 * @note This function calls compute_quadrature_on_intersection().
 *
 * @param [in] space_cache First cached triangulation.
 * @param [in] immersed_cache Second cached triangulation.
 * @param [in] degree The degree of accuracy of each quadrature formula.
 * @param [in] tol Tolerance used to discard small intersections.
 * @return std::vector<std::tuple<typename Triangulation<dim0,
 * spacedim>::cell_iterator, typename Triangulation<dim1,
 * spacedim>::cell_iterator, Quadrature<spacedim>>>.
 */
template <int dim0, int dim1, int spacedim>
std::vector<std::tuple<typename Triangulation<dim0, spacedim>::cell_iterator,
                       typename Triangulation<dim1, spacedim>::cell_iterator,
                       Quadrature<spacedim>>>
collect_quadratures_on_overlapped_grids(
    const GridTools::Cache<dim0, spacedim> &space_cache,
    const GridTools::Cache<dim1, spacedim> &immersed_cache,
    const unsigned int degree, const double tol = 1e-14);

} // namespace NonMatching
} // namespace dealii

#endif