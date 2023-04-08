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

#include <deal.II/grid/grid_out.h>

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/function.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_fe_field.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/linear_operator_tools.h>
#include <deal.II/numerics/error_estimator.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_tools_cache.h>
#include <deal.II/grid/tria.h>

#include "coupling_utilities.h"
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>

using namespace dealii;

// Functors-like classes to describe boundary values, right hand side,
// analytical solution, if any.
template <int dim> class RightHandSide : public Function<dim> {
public:
  virtual double value(const Point<dim> &p,
                       const unsigned int component = 0) const override;
};

template <>
double RightHandSide<3>::value(const Point<3> &p,
                               const unsigned int component) const {
  (void)p;
  (void)component;
  return 0.;
}

template <>
double RightHandSide<2>::value(const Point<2> &p,
                               const unsigned int component) const {
  // (void)p;
  (void)component;
  // return 0.;
  return 8. * numbers::PI * numbers::PI *
         (std::sin(2. * numbers::PI * p[0]) *
          std::sin(2. * numbers::PI * p[1]));
}

template <int dim> class Solution : public Function<dim> {
public:
  virtual double value(const Point<dim> &p,
                       const unsigned int component = 0) const override;

  virtual Tensor<1, dim>
  gradient(const Point<dim> &p,
           const unsigned int component = 0) const override;
};

template <>
double Solution<2>::value(const Point<2> &p,
                          const unsigned int component) const {
  (void)component;
  return std::sin(2. * numbers::PI * p[0]) * std::sin(2. * numbers::PI * p[1]);
}

template <>
Tensor<1, 2> Solution<2>::gradient(const Point<2> &p,
                                   const unsigned int component) const {
  (void)component;
  Tensor<1, 2> gradient;
  gradient[0] =
      std::cos(2. * numbers ::PI * p[0]) * std::sin(2. * numbers::PI * p[1]);

  gradient[1] =
      std::sin(2. * numbers ::PI * p[0]) * std::cos(2. * numbers::PI * p[1]);

  return 2. * numbers::PI * gradient;
}

template <int dim, int spacedim = dim> class PoissonNitscheInterface {
public:
  PoissonNitscheInterface();
  void run();

private:
  void generate_grids(const unsigned int);

  void adjust_grids();

  void setup_system();

  void assemble_system();

  void solve();

  void output_results(const unsigned cycle) const;

  /**
   * The actual triangulations. Here with "space_triangulation" we refer to
   * the original domain \Omega, also called the ambient space, while with
   * embedded we refer to the immersed domain, the one where we want to
   * impose a constraint.
   *
   */
  parallel::shared::Triangulation<spacedim> space_triangulation;
  Triangulation<dim, spacedim> embedded_triangulation;

  std::unique_ptr<GridTools::Cache<spacedim, spacedim>> space_cache;
  std::unique_ptr<GridTools::Cache<dim, spacedim>> embedded_cache;

  std::vector<
      std::tuple<typename dealii::Triangulation<spacedim>::cell_iterator,
                 typename dealii::Triangulation<dim, spacedim>::cell_iterator,
                 dealii::Quadrature<spacedim>>>
      cells_and_quads;

  FE_Q<spacedim> space_fe;

  DoFHandler<spacedim> space_dh;

  MappingQ1<spacedim> mapping;

  // Members needed to deescribe parametric surfaces
  std::unique_ptr<FiniteElement<dim, spacedim>> embedded_configuration_fe;
  std::unique_ptr<DoFHandler<dim, spacedim>> embedded_configuration_dh;
  Vector<double> embedded_configuration;
  std::unique_ptr<Mapping<dim, spacedim>> embedded_mapping;

  unsigned int embedded_configuration_finite_element_degree = 1;
  unsigned int embedded_initial_global_refinements = 8;

  MPI_Comm mpi_communicator;

  const unsigned int n_mpi_processes;
  const unsigned int this_mpi_process;

  AffineConstraints<double> space_constraints;
  SparsityPattern sparsity_pattern;

  LinearAlgebraTrilinos::MPI::SparseMatrix system_matrix;
  LinearAlgebraTrilinos::MPI::Vector solution;
  LinearAlgebraTrilinos::MPI::Vector system_rhs;

  mutable TimerOutput timer;

  mutable ConvergenceTable convergence_table;

  mutable DataOut<spacedim> data_out;

  /**
   * The penalty parameter which multiplies Nitsche's terms. In this program
   * it is defaulted to 10.0
   */

  double penalty = 10.0;

  unsigned int n_refinement_cycles = 5; // 6
};

template <int dim, int spacedim>
PoissonNitscheInterface<dim, spacedim>::PoissonNitscheInterface()
    : space_triangulation(MPI_COMM_WORLD), space_fe(1),
      space_dh(space_triangulation), mpi_communicator(MPI_COMM_WORLD),
      n_mpi_processes(Utilities::MPI::n_mpi_processes(mpi_communicator)),
      this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator)),
      timer(std::cout, TimerOutput::summary, TimerOutput::cpu_and_wall_times) {}

template <int dim, int spacedim>
void PoissonNitscheInterface<dim, spacedim>::generate_grids(
    const unsigned int cycle) {
  TimerOutput::Scope timer_section(timer, "Generate grids");
  if (cycle == 0) {
    GridGenerator::hyper_cube(space_triangulation, -1., 1.);
    space_triangulation.refine_global(4); // 2
  }

  if constexpr (dim == 1 && spacedim == 2) {
    if (cycle == 0) {

      GridIn<1, 2> grid_in;
      grid_in.attach_triangulation(embedded_triangulation);
      std::ifstream input_file("../../grids/flower_interface.vtk");
      Assert(dim == 1 && spacedim == 2, ExcInternalError());
      grid_in.read_vtk(input_file);
    }

    embedded_mapping = std::make_unique<MappingQ<dim, spacedim>>(1);

  } else {
    Assert(false, ExcMessage("Invalid dimensions."));
  }

  // We create unique pointers to cached triangulations. This This objects
  // will be necessary to compute the the Quadrature formulas on the
  // intersection of the cells.
  space_cache = std::make_unique<GridTools::Cache<spacedim, spacedim>>(
      space_triangulation);
  embedded_cache = std::make_unique<GridTools::Cache<dim, spacedim>>(
      embedded_triangulation, *embedded_mapping);
}

template <int dim, int spacedim>
void PoissonNitscheInterface<dim, spacedim>::adjust_grids() {

  std::cout << "Adjusting the grids (with two meshes)..." << std::endl;
  namespace bgi = boost::geometry::index;

  auto refine = [&]() {
    bool done = false;

    double min_embedded = 1e10;
    double max_embedded = 0;
    double min_space = 1e10;
    double max_space = 0;

    while (done == false) {
      // Bounding boxes of the space grid
      const auto &tree =
          space_cache->get_locally_owned_cell_bounding_boxes_rtree();

      // Bounding boxes of the embedded grid
      const auto &embedded_tree =
          embedded_cache->get_cell_bounding_boxes_rtree();

      // Let's check all cells whose bounding box contains an embedded
      // bounding box
      done = true;

      const bool use_space = false; // parameters.use_space;

      const bool use_embedded = false; // parameters.use_embedded;

      AssertThrow(!(use_embedded && use_space),
                  ExcMessage("You can't refine both the embedded and "
                             "the space grid at the same time."));

      for (const auto &[embedded_box, embedded_cell] : embedded_tree) {
        const auto &[p1, p2] = embedded_box.get_boundary_points();
        const auto diameter = p1.distance(p2);
        min_embedded = std::min(min_embedded, diameter);
        max_embedded = std::max(max_embedded, diameter);

        for (const auto &[space_box, space_cell] :
             tree | bgi::adaptors::queried(bgi::intersects(embedded_box))) {
          const auto &[sp1, sp2] = space_box.get_boundary_points();
          const auto space_diameter = sp1.distance(sp2);
          min_space = std::min(min_space, space_diameter);
          max_space = std::max(max_space, space_diameter);

          if (use_embedded && space_diameter < diameter) {
            embedded_cell->set_refine_flag();
            done = false;
          }
          if (use_space && diameter < space_diameter) {
            space_cell->set_refine_flag();
            done = false;
          }
        }
      }
      if (done == false) {
        if (use_embedded) {
          // Compute again the embedded displacement grid
          embedded_triangulation.execute_coarsening_and_refinement();
        }
        if (use_space) {
          // Compute again the embedded displacement grid
          space_triangulation.execute_coarsening_and_refinement();
        }
      }
    }
    return std::make_tuple(min_space, max_space, min_embedded, max_embedded);
  };

  // Do the refinement loop once, to make sure we satisfy our criterions
  refine();

  // Pre refine the space grid according to the delta refinement
  const unsigned int n_space_cycles = 4; // before it was 2.
  for (unsigned int i = 0; i < n_space_cycles; ++i) {
    const auto &tree =
        space_cache->get_locally_owned_cell_bounding_boxes_rtree();

    const auto &embedded_tree = embedded_cache->get_cell_bounding_boxes_rtree();

    for (const auto &[embedded_box, embedded_cell] : embedded_tree)
      for (const auto &[space_box, space_cell] :
           tree | bgi::adaptors::queried(bgi::intersects(embedded_box)))
        space_cell->set_refine_flag();
    space_triangulation.execute_coarsening_and_refinement();

    // Make sure again we satisfy our criterion after the space
    // refinement
    refine();
  }

  // Check once again we satisfy our criterion, and record min/max
  const auto [sm, sM, em, eM] = refine();

  std::cout << "Space local min/max diameters   : " << sm << "/" << sM
            << std::endl
            << "Embedded space min/max diameters: " << em << "/" << eM
            << std::endl;
}

template <int dim, int spacedim>
void PoissonNitscheInterface<dim, spacedim>::setup_system() {
  TimerOutput::Scope timer_section(timer, "Setup system");
  // std::cout << "System setup" << std::endl;

  // We propagate the information about the constants to all functions of
  // the problem, so that constants can be used within the functions

  space_dh.distribute_dofs(space_fe);
  std::cout << "Number of dofs in space: " << space_dh.n_dofs() << std::endl;

  space_constraints.clear();
  DoFTools::make_hanging_node_constraints(space_dh, space_constraints);

  // This is where we apply essential boundary conditions.
  VectorTools::interpolate_boundary_values(
      space_dh, 0, Solution<spacedim>(),
      space_constraints); // zero Dirichlet on the boundary

  space_constraints.close();
  DynamicSparsityPattern dsp(space_dh.n_dofs());
  DoFTools::make_sparsity_pattern(space_dh, dsp, space_constraints, false);
  sparsity_pattern.copy_from(dsp);

  const std::vector<IndexSet> locally_owned_dofs_per_proc =
      DoFTools::locally_owned_dofs_per_subdomain(space_dh);
  const IndexSet locally_owned_dofs =
      locally_owned_dofs_per_proc[this_mpi_process];

  system_matrix.reinit(locally_owned_dofs, locally_owned_dofs, sparsity_pattern,
                       mpi_communicator);

  solution.reinit(locally_owned_dofs, mpi_communicator);
  system_rhs.reinit(locally_owned_dofs, mpi_communicator);
}

template <int dim, int spacedim>
void PoissonNitscheInterface<dim, spacedim>::assemble_system() {
  {
    TimerOutput::Scope timer_section(timer, "Assemble system");
    // std::cout << "Assemble system" << std::endl;

    QGauss<spacedim> quadrature_formula(2 * space_fe.degree + 1);
    FEValues<spacedim, spacedim> fe_values(
        mapping, space_fe, quadrature_formula,
        update_values | update_gradients | update_quadrature_points |
            update_JxW_values);

    const unsigned int dofs_per_cell = space_fe.n_dofs_per_cell();
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> cell_rhs(dofs_per_cell);
    RightHandSide<spacedim> rhs;
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    for (const auto &cell : space_dh.active_cell_iterators()) {
      if (cell->is_locally_owned()) {
        fe_values.reinit(cell);
        cell_matrix = 0;
        cell_rhs = 0;
        const auto &q_points = fe_values.get_quadrature_points();
        for (const unsigned int q_index :
             fe_values.quadrature_point_indices()) {
          for (const unsigned int i : fe_values.dof_indices())
            for (const unsigned int j : fe_values.dof_indices())
              cell_matrix(i, j) +=
                  (fe_values.shape_grad(i, q_index) * // grad phi_i(x_q)
                   fe_values.shape_grad(j, q_index) * // grad phi_j(x_q)
                   fe_values.JxW(q_index));           // dx
          for (const unsigned int i : fe_values.dof_indices())
            cell_rhs(i) +=
                (fe_values.shape_value(i, q_index) * // phi_i(x_q)
                 rhs.value(q_points[q_index]) * fe_values.JxW(q_index)); // dx
        }

        cell->get_dof_indices(local_dof_indices);
        space_constraints.distribute_local_to_global(cell_matrix, cell_rhs,
                                                     local_dof_indices,
                                                     system_matrix, system_rhs);
      }
    }
  }

  std::cout << "Assemble Nitsche contributions" << std::endl;
  {
    TimerOutput::Scope timer_section(timer, "Assemble Nitsche terms");

    FE_Q<dim, spacedim> embedded_fe(1);
    DoFHandler<dim, spacedim> embedded_dh(embedded_triangulation);
    embedded_dh.distribute_dofs(embedded_fe);
    std::cout << "Embedded DoFs: " << embedded_dh.n_dofs() << std::endl;

    NonMatching::assemble_nitsche_with_exact_intersections<spacedim, dim,
                                                           spacedim>(
        space_dh, cells_and_quads, system_matrix, space_constraints,
        ComponentMask(), MappingQ1<spacedim, spacedim>(),
        Functions::ConstantFunction<spacedim>(2.0), penalty);

    // Without composite intersections
    // Add the Nitsche's contribution to the rhs. The embedded value is
    // parsed from the parameter file, while we have again the constant 2.0
    // in front of that term, parsed as above from command line. Finally, we
    // have the penalty parameter as before.
    NonMatching::create_nitsche_rhs_with_exact_intersections<spacedim, dim,
                                                             spacedim>(
        space_dh, cells_and_quads, system_rhs, space_constraints,
        MappingQ1<spacedim>(), Solution<spacedim>(),
        Functions::ConstantFunction<spacedim>(2.0), penalty);

    system_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);
  }
}

// We solve the resulting system as done in the classical Poisson example.
template <int dim, int spacedim>
void PoissonNitscheInterface<dim, spacedim>::solve() {
  TimerOutput::Scope timer_section(timer, "Solve system");
  std::cout << "Solve system" << std::endl;

  LinearAlgebraTrilinos::MPI::PreconditionAMG preconditioner;
  preconditioner.initialize(system_matrix);
  SolverControl solver_control(solution.size(), 1e-8);
  LinearAlgebraTrilinos::SolverCG solver(solver_control);
  solver.solve(system_matrix, solution, system_rhs, preconditioner);

  std::cout << "Solver converged in: " << solver_control.last_step()
            << " iterations" << std::endl;
  space_constraints.distribute(solution);
}

// Finally, we output the solution living in the embedding space, just
// like all the other programs.
template <int dim, int spacedim>
void PoissonNitscheInterface<dim, spacedim>::output_results(
    const unsigned cycle) const {
  std::cout << "Output results" << std::endl;
  TimerOutput::Scope timer_section(timer, "Output results");
  data_out.clear();
  if (cycle < 3) {
    data_out.attach_dof_handler(space_dh);
    data_out.add_data_vector(solution, "solution");
    data_out.build_patches();
    std::ofstream output("solution_nitsche" + std::to_string(dim) +
                         std::to_string(spacedim) + std::to_string(cycle) +
                         ".vtu");
    data_out.write_vtu(output);
  }
  {
    Vector<double> difference_per_cell(space_triangulation.n_active_cells());
    VectorTools::integrate_difference(
        space_dh, solution, Solution<spacedim>(), difference_per_cell,
        QGauss<spacedim>(2 * space_fe.degree + 1), VectorTools::L2_norm);
    const double L2_error = VectorTools::compute_global_error(
        space_triangulation, difference_per_cell, VectorTools::L2_norm);

    difference_per_cell.reinit(
        space_triangulation
            .n_active_cells()); // zero out again to store the H1 error
    VectorTools::integrate_difference(
        space_dh, solution, Solution<spacedim>(), difference_per_cell,
        QGauss<spacedim>(2 * space_fe.degree + 1), VectorTools::H1_norm);
    const double H1_error = VectorTools::compute_global_error(
        space_triangulation, difference_per_cell, VectorTools::H1_norm);

    convergence_table.add_value("cycle", cycle);
    convergence_table.add_value("cells", space_triangulation.n_active_cells());
    convergence_table.add_value("dofs", space_dh.n_dofs() -
                                            space_constraints.n_constraints());
    convergence_table.add_value("L2", L2_error);
    convergence_table.add_value("H1", H1_error);
  }

  {
    if (cycle == 3) {
      std::ofstream output_test_space("space_grid_cycle3.vtk");
      GridOut().write_vtk(space_triangulation, output_test_space);
      std::ofstream output_test_embedded("embedded_grid_cycle3.vtk");
      GridOut().write_vtk(embedded_triangulation, output_test_embedded);
    }
  }
}

// The run() method here differs only in the call to
// NonMatching::compute_intersection().
template <int dim, int spacedim>
void PoissonNitscheInterface<dim, spacedim>::run() {
  for (unsigned int cycle = 0; cycle < n_refinement_cycles; ++cycle) {
    std::cout << "Cycle: " << cycle << std::endl;
    generate_grids(cycle);
    if (spacedim == 2 && cycle == 0)
      adjust_grids();
    // else if (spacedim == 3/* && cycle == 0*/)
    //   adjust_grids();

    // Compute all the things we need to assemble the Nitsche's
    // contributions, namely the two cached triangulations and a degree to
    // integrate over the intersections.
    {
      TimerOutput::Scope timer_section(timer, "Total time cycle " +
                                                  std::to_string(cycle));
      std::cout << "Start collecting quadratures" << std::endl;
      cells_and_quads = NonMatching::collect_quadratures_on_overlapped_grids(
          *space_cache, *embedded_cache, 2 * space_fe.degree + 1);
      std::cout << "Collected quadratures" << std::endl;

      double sum = 0.;
      for (const auto &p : cells_and_quads) {
        auto quad = std::get<2>(p);
        sum += std::accumulate(quad.get_weights().begin(),
                               quad.get_weights().end(), 0.);
      }
      std::cout << "Area/Measure: " << sum << std::endl;

      setup_system();
      assemble_system();
      solve();
    }

    // error_table.error_from_exact(space_dh, solution, exact_solution);
    output_results(cycle);

    if (cycle < n_refinement_cycles - 1) {
      space_triangulation.refine_global(1);
      embedded_triangulation.refine_global(1);
    }
    cells_and_quads.clear();
  }

  convergence_table.set_precision("L2", 3);
  convergence_table.set_precision("H1", 3);
  convergence_table.set_scientific("L2", true);
  convergence_table.set_scientific("H1", true);
  convergence_table.evaluate_convergence_rates(
      "L2", "dofs", ConvergenceTable::reduction_rate_log2, spacedim);
  convergence_table.evaluate_convergence_rates(
      "H1", "dofs", ConvergenceTable::reduction_rate_log2, spacedim);
  convergence_table.write_text(std::cout);
}

int main(int argc, char *argv[]) {
  try {
    {
      std::cout << "Solving in 1D/2D" << std::endl;
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
      PoissonNitscheInterface<1, 2> problem;
      problem.run();
      return 0;
    }
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
