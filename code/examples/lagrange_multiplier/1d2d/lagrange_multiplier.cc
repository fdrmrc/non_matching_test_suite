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

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/function.h>
#include <deal.II/base/function_lib.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/parsed_function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_fe_field.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_tools_cache.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/linear_operator_tools.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/trilinos_linear_operator.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>

#include "coupling_utilities.h"

using namespace dealii;

template <int dim, int spacedim = dim>
class PoissonLM {
 public:
  class Parameters : public ParameterAcceptor {
   public:
    Parameters();

    unsigned int n_refinement_cycles = 6;

    bool apply_delta_refinements;

    bool adjust_grids_ratio = false;

    unsigned int space_pre_refinement_cycles = 2;

    unsigned int space_initial_global_refinements = 4;

    unsigned int embedded_initial_global_refinements = 2;

    bool initialized = false;

    bool use_space;

    bool use_embedded;

    int embedded_post_refinement_cycles = 0;

    std::string coupling_strategy;

    unsigned int fe_space_degree = 1;

    unsigned int fe_embedded_degree = 1;

    unsigned int embedded_configuration_finite_element_degree = 1;
  };

  PoissonLM(const Parameters &parameters, const bool);

  void run();

 private:
  const Parameters &parameters;

  void setup_grids_and_dofs();

  void setup_coupling();

  void setup_embedded_dofs();

  void setup_space_dofs();

  void adjust_grids();

  void assemble_system();

  void solve();

  void output_results(const unsigned cycle) const;

  parallel::shared::Triangulation<spacedim> space_triangulation;
  Triangulation<dim, spacedim> embedded_triangulation;

  std::unique_ptr<GridTools::Cache<spacedim, spacedim>> space_cache;
  std::unique_ptr<GridTools::Cache<dim, spacedim>> embedded_cache;

  std::vector<
      std::tuple<typename dealii::Triangulation<spacedim>::cell_iterator,
                 typename dealii::Triangulation<dim, spacedim>::cell_iterator,
                 dealii::Quadrature<spacedim>>>
      cells_and_quads;

  std::unique_ptr<FE_Q<spacedim>> space_fe;
  std::unique_ptr<FE_DGQ<dim, spacedim>> embedded_fe;

  /**
   * The actual DoFHandler class.
   */
  std::unique_ptr<DoFHandler<spacedim>> space_dh;
  std::unique_ptr<DoFHandler<dim, spacedim>> embedded_dh;
  MappingQ1<spacedim> space_mapping;

  // The next members are needed to generate a mesh out of a parametrized curve
  std::unique_ptr<FiniteElement<dim, spacedim>> embedded_configuration_fe;
  std::unique_ptr<DoFHandler<dim, spacedim>> embedded_configuration_dh;
  Vector<double> embedded_configuration;
  std::unique_ptr<Mapping<dim, spacedim>> embedded_mapping;
  // MappingQ1<dim, spacedim> embedded_mapping;

  // End of members for immersed grid generation

  AffineConstraints<double> space_constraints;
  AffineConstraints<double> embedded_constraints;
  SparsityPattern stiffness_sparsity_pattern;
  SparsityPattern mass_sparsity_pattern;
  SparsityPattern coupling_sparsity_pattern;

  MPI_Comm mpi_communicator;
  const unsigned int n_mpi_processes;
  const unsigned int this_mpi_process;

  TrilinosWrappers::SparseMatrix stiffness_matrix;
  TrilinosWrappers::SparseMatrix mass_matrix;
  TrilinosWrappers::SparseMatrix coupling_matrix;
  TrilinosWrappers::MPI::Vector space_rhs;
  TrilinosWrappers::MPI::Vector embedded_rhs;

  TrilinosWrappers::MPI::Vector solution;
  TrilinosWrappers::MPI::Vector lambda;

  ParameterAcceptorProxy<Functions::ParsedFunction<spacedim>> rhs_function;

  ParameterAcceptorProxy<Functions::ParsedFunction<spacedim>> solution_function;

  ParameterAcceptorProxy<Functions::ParsedFunction<spacedim>>
      multiplier_function;

  ParameterAcceptorProxy<Functions::ParsedFunction<spacedim>>
      boundary_condition_function;

  ParameterAcceptorProxy<Functions::ParsedFunction<spacedim>>
      embedded_configuration_function;

  mutable TimerOutput timer;

  mutable ConvergenceTable convergence_table;

  bool smoothness;

  mutable DataOut<spacedim> data_out;

  unsigned int cycle;

  mutable unsigned int iter;
};

template <int dim, int spacedim>
PoissonLM<dim, spacedim>::Parameters::Parameters()
    : ParameterAcceptor("/Distributed Lagrange<" +
                        Utilities::int_to_string(dim) + "," +
                        Utilities::int_to_string(spacedim) + ">/") {
  add_parameter("Adjust grids", adjust_grids_ratio);

  add_parameter("Number of refinement cycles", n_refinement_cycles);

  add_parameter("Number of space initial refinement cycles",
                space_initial_global_refinements);

  add_parameter("Number of embedded initial refinement cycles",
                embedded_initial_global_refinements);

  add_parameter("Space pre refinements cycles", space_pre_refinement_cycles);

  add_parameter("Embedded post refinement cycles",
                embedded_post_refinement_cycles);

  add_parameter("Apply space refinements steps near embedded domain",
                apply_delta_refinements);

  add_parameter("Use space refinement", use_space);

  add_parameter("Use embedded refinement", use_embedded);

  add_parameter("Coupling strategy", coupling_strategy);

  add_parameter("Finite element space degree", fe_space_degree);

  add_parameter("Finite element embedded degree", fe_embedded_degree);

  add_parameter("Embedded configuration finite element degree",
                embedded_configuration_finite_element_degree);

  parse_parameters_call_back.connect([&]() -> void { initialized = true; });
}

template <int dim, int spacedim>
PoissonLM<dim, spacedim>::PoissonLM(const Parameters &parameters,
                                    const bool smoothness_solution)
    : parameters(parameters),
      space_triangulation(MPI_COMM_WORLD),
      mpi_communicator(MPI_COMM_WORLD),
      n_mpi_processes(Utilities::MPI::n_mpi_processes(mpi_communicator)),
      this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator)),
      rhs_function("Right hand side"),
      solution_function("Solution"),
      multiplier_function("Solution multiplier"),
      boundary_condition_function("Boundary condition"),
      embedded_configuration_function("Immersed configuration", spacedim),
      timer(std::cout, TimerOutput::every_call_and_summary,
            TimerOutput::cpu_times) {
  rhs_function.declare_parameters_call_back.connect(
      []() -> void { ParameterAcceptor::prm.set("Function expression", "0"); });

  solution_function.declare_parameters_call_back.connect(
      []() -> void { ParameterAcceptor::prm.set("Function expression", "1"); });

  multiplier_function.declare_parameters_call_back.connect(
      []() -> void { ParameterAcceptor::prm.set("Function expression", "0"); });

  boundary_condition_function.declare_parameters_call_back.connect(
      []() -> void { ParameterAcceptor::prm.set("Function expression", "0"); });

  embedded_configuration_function.declare_parameters_call_back.connect(
      []() -> void {
        ParameterAcceptor::prm.set("Function constants", "R=.3, Cx=.4, Cy=.4");

        ParameterAcceptor::prm.set("Function expression",
                                   "R*cos(2*pi*x)+Cx; R*sin(2*pi*x)+Cy");
      });
  iter = numbers::invalid_unsigned_int;
  smoothness = smoothness_solution;
}

template <int dim, int spacedim>
void PoissonLM<dim, spacedim>::setup_grids_and_dofs() {
  // TimerOutput::Scope timer_section(timer, "Generate grids");
  if (cycle == 0) {
    GridGenerator::hyper_cube(space_triangulation, -1., 1.);

    if constexpr (dim == 1 && spacedim == 2) {
      space_triangulation.refine_global(
          parameters.space_initial_global_refinements);  // 4
      if (parameters.coupling_strategy == "inexact") {
        // Use a level set to generate the actual domain.
        GridGenerator::hyper_cube(embedded_triangulation, 0.,
                                  1.);  // parametric space for the curve
        embedded_triangulation.refine_global(
            parameters.embedded_initial_global_refinements);  // 2

        embedded_configuration_fe = std::make_unique<FESystem<dim, spacedim>>(
            FE_Q<dim, spacedim>(
                parameters.embedded_configuration_finite_element_degree),
            spacedim);
      } else {
        const double Cx = .5;
        const double Cy = .5;
        const double R = .3;
        GridGenerator::hyper_sphere(embedded_triangulation, {Cx, Cy}, R);
        embedded_triangulation.refine_global(
            parameters.embedded_initial_global_refinements);  // 2

        embedded_mapping = std::make_unique<MappingQ<dim, spacedim>>(1);
      }
    } else {
      Assert(false, ExcMessage("Wrong dim-spacedim dimensions for this test."));
    }
  }

  space_fe = std::make_unique<FE_Q<spacedim>>(parameters.fe_space_degree);
  embedded_fe =
      std::make_unique<FE_DGQ<dim, spacedim>>(parameters.fe_embedded_degree);

  if (parameters.coupling_strategy == "inexact") {
    embedded_configuration_dh =
        std::make_unique<DoFHandler<dim, spacedim>>(embedded_triangulation);

    embedded_configuration_dh->distribute_dofs(*embedded_configuration_fe);

    embedded_configuration.reinit(embedded_configuration_dh->n_dofs());

    VectorTools::interpolate(*embedded_configuration_dh,
                             embedded_configuration_function,
                             embedded_configuration);

    embedded_mapping =
        std::make_unique<MappingFEField<dim, spacedim, Vector<double>>>(
            *embedded_configuration_dh, embedded_configuration);
  }

  space_cache = std::make_unique<GridTools::Cache<spacedim, spacedim>>(
      space_triangulation);
  embedded_cache = std::make_unique<GridTools::Cache<dim, spacedim>>(
      embedded_triangulation, *embedded_mapping);

  // Embedded DoFs can be already distributed
  setup_embedded_dofs();

  // Adjust the grid during for the first cycle
  if (parameters.adjust_grids_ratio == true && cycle == 0) {
    adjust_grids();
  }

  setup_space_dofs();

  const double embedded_space_maximal_diameter =
      GridTools::maximal_cell_diameter(embedded_triangulation,
                                       *embedded_mapping);
  double embedding_space_minimal_diameter =
      GridTools::minimal_cell_diameter(space_triangulation, space_mapping);

  std::cout << "Space minimal diameter: " << embedding_space_minimal_diameter
            << ", embedded maximal diameter: "
            << embedded_space_maximal_diameter << ", ratio: "
            << embedded_space_maximal_diameter /
                   embedding_space_minimal_diameter
            << std::endl;
}

template <int dim, int spacedim>
void PoissonLM<dim, spacedim>::adjust_grids() {
  std::cout << "Adjusting the grids..." << std::endl;
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

      const bool use_space = parameters.use_space;

      const bool use_embedded = parameters.use_embedded;

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
  if (parameters.apply_delta_refinements &&
      parameters.space_pre_refinement_cycles != 0)
    for (unsigned int i = 0; i < parameters.space_pre_refinement_cycles; ++i) {
      const auto &tree =
          space_cache->get_locally_owned_cell_bounding_boxes_rtree();

      const auto &embedded_tree =
          embedded_cache->get_cell_bounding_boxes_rtree();

      for (const auto &[embedded_box, embedded_cell] : embedded_tree)
        for (const auto &[space_box, space_cell] :
             tree | bgi::adaptors::queried(bgi::intersects(embedded_box)))
          space_cell->set_refine_flag();
      space_triangulation.execute_coarsening_and_refinement();

      // Make sure again we satisfy our criterion after the space
      // refinement
      refine();
    }

  // Post refinement on embedded grid is easy
  if (parameters.apply_delta_refinements &&
      parameters.embedded_post_refinement_cycles != 0) {
    embedded_triangulation.refine_global(
        parameters.embedded_post_refinement_cycles);
  }

  // Check once again we satisfy our criterion, and record min/max
  const auto [sm, sM, em, eM] = refine();

  std::cout << "Space local min/max diameters   : " << sm << "/" << sM
            << std::endl
            << "Embedded space min/max diameters: " << em << "/" << eM
            << std::endl;
}

template <int dim, int spacedim>
void PoissonLM<dim, spacedim>::setup_space_dofs() {
  // Setup space DoFs
  space_dh = std::make_unique<DoFHandler<spacedim>>(space_triangulation);
  space_dh->distribute_dofs(*space_fe);
  std::cout << "Number of dofs in space: " << space_dh->n_dofs() << std::endl;
  space_constraints.clear();
  DoFTools::make_hanging_node_constraints(*space_dh, space_constraints);

  // This is where we apply essential boundary conditions.
  VectorTools::interpolate_boundary_values(
      *space_dh, 0, boundary_condition_function,
      space_constraints);  // Dirichlet on the exterior (fictitious) boundary

  space_constraints.close();

  DynamicSparsityPattern dsp(space_dh->n_dofs(), space_dh->n_dofs());
  DoFTools::make_sparsity_pattern(*space_dh, dsp, space_constraints);
  stiffness_sparsity_pattern.copy_from(dsp);

  const std::vector<IndexSet> locally_owned_dofs_per_proc =
      DoFTools::locally_owned_dofs_per_subdomain(*space_dh);
  const IndexSet locally_owned_dofs =
      locally_owned_dofs_per_proc[this_mpi_process];

  stiffness_matrix.reinit(locally_owned_dofs, locally_owned_dofs, dsp,
                          mpi_communicator);

  solution.reinit(locally_owned_dofs, mpi_communicator);
  space_rhs.reinit(locally_owned_dofs, mpi_communicator);
}

template <int dim, int spacedim>
void PoissonLM<dim, spacedim>::setup_embedded_dofs() {
  embedded_dh =
      std::make_unique<DoFHandler<dim, spacedim>>(embedded_triangulation);
  embedded_dh->distribute_dofs(*embedded_fe);
  std::cout << "Embedded_DOFs: " << embedded_dh->n_dofs() << std::endl;

  // Mass matrix for the preconditioner
  DynamicSparsityPattern mass_dsp(embedded_dh->n_dofs(), embedded_dh->n_dofs());
  DoFTools::make_sparsity_pattern(*embedded_dh, mass_dsp, embedded_constraints);
  mass_sparsity_pattern.copy_from(mass_dsp);

  const std::vector<IndexSet> locally_owned_dofs_per_proc =
      DoFTools::locally_owned_dofs_per_subdomain(*embedded_dh);
  const IndexSet locally_owned_dofs =
      locally_owned_dofs_per_proc[this_mpi_process];

  mass_matrix.reinit(locally_owned_dofs, locally_owned_dofs, mass_dsp,
                     mpi_communicator);

  embedded_rhs.reinit(locally_owned_dofs, mpi_communicator);
  lambda.reinit(locally_owned_dofs, mpi_communicator);
}

template <int dim, int spacedim>
void PoissonLM<dim, spacedim>::setup_coupling() {
  QGauss<dim> quad(2 * parameters.fe_space_degree + 1);

  DynamicSparsityPattern dsp(space_dh->n_dofs(), embedded_dh->n_dofs());

  {
    TimerOutput::Scope timer_section(timer, "Setup coupling");

    if (parameters.coupling_strategy == "inexact") {
      NonMatching::create_coupling_sparsity_pattern(
          *space_cache, *space_dh, *embedded_dh,
          QGauss<dim>(2 * parameters.fe_space_degree + 1), dsp,
          space_constraints, ComponentMask(), ComponentMask(),
          *embedded_mapping);
    } else if (parameters.coupling_strategy == "exact") {
      NonMatching::create_coupling_sparsity_pattern_with_exact_intersections(
          cells_and_quads, *space_dh, *embedded_dh, dsp, space_constraints,
          ComponentMask(), ComponentMask(), embedded_constraints);
    } else {
      Assert(false, ExcMessage("Please select a valid strategy."));
    }
  }

  coupling_sparsity_pattern.copy_from(dsp);

  const std::vector<IndexSet> locally_owned_dofs_per_proc_space =
      DoFTools::locally_owned_dofs_per_subdomain(*space_dh);
  const IndexSet locally_owned_dofs_space =
      locally_owned_dofs_per_proc_space[this_mpi_process];

  const std::vector<IndexSet> locally_owned_dofs_per_proc_emb =
      DoFTools::locally_owned_dofs_per_subdomain(*embedded_dh);
  const IndexSet locally_owned_dofs_emb =
      locally_owned_dofs_per_proc_emb[this_mpi_process];

  coupling_matrix.reinit(locally_owned_dofs_space, locally_owned_dofs_emb,
                         coupling_sparsity_pattern, mpi_communicator);
}

template <int dim, int spacedim>
void PoissonLM<dim, spacedim>::assemble_system() {
  {
    TimerOutput::Scope timer_section(timer, "Assemble system");
    std::cout << "Assemble system" << std::endl;

    QGauss<dim> quadrature_formula_gamma(2 * parameters.fe_embedded_degree + 1);

    FEValues<dim, spacedim> fe_values_gamma(
        *embedded_mapping, *embedded_fe, quadrature_formula_gamma,
        update_values | update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_gamma_cell = embedded_fe->n_dofs_per_cell();
    FullMatrix<double> cell_mass_matrix(dofs_per_gamma_cell,
                                        dofs_per_gamma_cell);

    std::vector<types::global_dof_index> local_dof_gamma_indices(
        dofs_per_gamma_cell);
    for (const auto &cell : embedded_dh->active_cell_iterators()) {
      fe_values_gamma.reinit(cell);
      cell_mass_matrix = 0;
      for (const unsigned int q_index :
           fe_values_gamma.quadrature_point_indices()) {
        for (const unsigned int i : fe_values_gamma.dof_indices())
          for (const unsigned int j : fe_values_gamma.dof_indices())
            cell_mass_matrix(i, j) +=
                fe_values_gamma.shape_value(i, q_index) *  //  q_i(x_q)
                fe_values_gamma.shape_value(j, q_index) *  //  q_j(x_q)
                fe_values_gamma.JxW(q_index);              // dx
      }

      cell->get_dof_indices(local_dof_gamma_indices);
      embedded_constraints.distribute_local_to_global(
          cell_mass_matrix, local_dof_gamma_indices, mass_matrix);
    }

    QGauss<spacedim> quadrature_formula(2 * parameters.fe_space_degree + 1);
    FEValues<spacedim, spacedim> fe_values(
        space_mapping, *space_fe, quadrature_formula,
        update_values | update_gradients | update_quadrature_points |
            update_JxW_values);

    const unsigned int dofs_per_cell = space_fe->n_dofs_per_cell();
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    for (const auto &cell : space_dh->active_cell_iterators()) {
      fe_values.reinit(cell);
      cell_matrix = 0;
      cell_rhs = 0;
      const auto &q_points = fe_values.get_quadrature_points();
      for (const unsigned int q_index : fe_values.quadrature_point_indices()) {
        for (const unsigned int i : fe_values.dof_indices())
          for (const unsigned int j : fe_values.dof_indices())
            cell_matrix(i, j) +=
                (fe_values.shape_grad(i, q_index) *  // grad phi_i(x_q)
                 fe_values.shape_grad(j, q_index) *  // grad phi_j(x_q)
                 fe_values.JxW(q_index));            // dx
        for (const unsigned int i : fe_values.dof_indices())
          cell_rhs(i) += (fe_values.shape_value(i, q_index) *  // phi_i(x_q)
                          /*  forcing_term.value(
                              fe_values.quadrature_point(q_index)) * // f(x_q)*/
                          rhs_function.value(q_points[q_index]) *
                          fe_values.JxW(q_index));  // dx
      }

      cell->get_dof_indices(local_dof_indices);
      space_constraints.distribute_local_to_global(cell_matrix, cell_rhs,
                                                   local_dof_indices,
                                                   stiffness_matrix, space_rhs);
    }

    VectorTools::create_right_hand_side(
        *embedded_mapping, *embedded_dh,
        QGauss<dim>(2 * parameters.fe_embedded_degree + 1), solution_function,
        embedded_rhs);
  }

  std::cout << "Assemble coupling term" << std::endl;
  {
    TimerOutput::Scope timer_section(timer, "Assemble coupling term");

    if (parameters.coupling_strategy == "inexact") {
      NonMatching::create_coupling_mass_matrix(
          *space_dh, *embedded_dh,
          QGauss<dim>(2 * parameters.fe_space_degree + 1), coupling_matrix,
          space_constraints, ComponentMask(), ComponentMask(), space_mapping,
          *embedded_mapping, embedded_constraints);
    } else if (parameters.coupling_strategy == "exact") {
      // Coupling mass matrix
      NonMatching::create_coupling_mass_matrix_with_exact_intersections(
          *space_dh, *embedded_dh, cells_and_quads, coupling_matrix,
          space_constraints, ComponentMask(), ComponentMask(), space_mapping,
          *embedded_mapping, embedded_constraints);
    }
  }

  stiffness_matrix.compress(VectorOperation::add);
  coupling_matrix.compress(VectorOperation::add);
  mass_matrix.compress(VectorOperation::add);
  space_rhs.compress(VectorOperation::add);
  embedded_rhs.compress(VectorOperation::add);
}

// We solve the resulting system as done in the classical Poisson example.
template <int dim, int spacedim>
void PoissonLM<dim, spacedim>::solve() {
  // TimerOutput::Scope timer_section(timer, "Solve system");
  std::cout << "Solve system" << std::endl;

  // SparseDirectUMFPACK K_inv_umfpack;
  // K_inv_umfpack.initialize(stiffness_matrix);

  TrilinosWrappers::PreconditionAMG prec_stiffness;
  prec_stiffness.initialize(stiffness_matrix);

  auto K = linear_operator<TrilinosWrappers::MPI::Vector>(stiffness_matrix);

  ReductionControl reduction_control_K(solution.size(), 1.0e-14, 1e-12);
  SolverCG<TrilinosWrappers::MPI::Vector> solver_cg_K(reduction_control_K);
  auto K_inv = inverse_operator(K, solver_cg_K, prec_stiffness);

  auto M = linear_operator<TrilinosWrappers::MPI::Vector>(mass_matrix);
  auto Ct = linear_operator<TrilinosWrappers::MPI::Vector>(coupling_matrix);
  auto C = transpose_operator<TrilinosWrappers::MPI::Vector>(Ct);

  // auto K_inv = linear_operator<TrilinosWrappers::MPI::Vector>(K, K_inv);

  auto preconditioner = C * K * Ct + M;

  auto S = C * K_inv * Ct;
  // ReductionControl reduction_control(2000, 1.0e-12, 1.0e-10);

  //
  ReductionControl reduction_control(2000, 1.0e-10, 1.0e-2);
  // ReductionControl reduction_control(2000, 1.0e-12, 1.0e-2);
  // SolverCG<Vector<double>> solver_cg(reduction_control);
  // SolverFGMRES<TrilinosWrappers::MPI::Vector> solver_cg(reduction_control);
  SolverGMRES<TrilinosWrappers::MPI::Vector> solver_cg(reduction_control);
  // SolverCG<TrilinosWrappers::MPI::Vector> solver_cg(reduction_control);

  auto S_inv = inverse_operator(S, solver_cg, preconditioner);
  // auto S_inv = inverse_operator(S, solver_cg, PreconditionIdentity());

  lambda = S_inv * (C * K_inv * space_rhs - embedded_rhs);
  solution = K_inv * (space_rhs - Ct * lambda);
  std::cout << "Norm of the multiplier: " << lambda.norm_sqr() << std::endl;

  std::cout << "Solved with Schur in : " << reduction_control.last_step()
            << "iterations." << std::endl;
  iter = reduction_control.last_step();

  std::cout << "Solved with CG in : " << reduction_control_K.last_step()
            << "iterations." << std::endl;

  std::cout << "Total number of iterations: "
            << reduction_control.last_step() + reduction_control_K.last_step()
            << std::endl;

  space_constraints.distribute(solution);
}

// Finally, we output the solution living in the embedding space
template <int dim, int spacedim>
void PoissonLM<dim, spacedim>::output_results(const unsigned cycle) const {
  TimerOutput::Scope timer_section(timer, "Output results");
  std::cout << "Output results" << std::endl;
  data_out.clear();
  // if (cycle < 3) {
  //   std::ofstream data_out_file("space_solution_lm_1d2d.vtu");
  //   data_out.attach_dof_handler(*space_dh);
  //   data_out.add_data_vector(solution, "solution");
  //   data_out.build_patches();
  //   data_out.write_vtu(data_out_file);
  // }

  {
    Vector<double> difference_per_cell(space_triangulation.n_active_cells());
    VectorTools::integrate_difference(
        *space_dh, solution, solution_function, difference_per_cell,
        QGauss<spacedim>(2 * parameters.fe_space_degree + 1),
        VectorTools::L2_norm);
    const double L2_error = VectorTools::compute_global_error(
        space_triangulation, difference_per_cell, VectorTools::L2_norm);

    difference_per_cell.reinit(
        space_triangulation
            .n_active_cells());  // zero out again to store the H1 error
    VectorTools::integrate_difference(
        *space_dh, solution, solution_function, difference_per_cell,
        QGauss<spacedim>(2 * parameters.fe_space_degree + 1),
        VectorTools::H1_norm);
    const double H1_error = VectorTools::compute_global_error(
        space_triangulation, difference_per_cell, VectorTools::H1_norm);

    convergence_table.add_value("cycle", cycle);
    convergence_table.add_value("cells", space_triangulation.n_active_cells());
    convergence_table.add_value(
        "dofs", space_dh->n_dofs() - space_constraints.n_constraints());
    convergence_table.add_value("dofs_emb", embedded_dh->n_dofs());
    convergence_table.add_value("L2", L2_error);
    convergence_table.add_value("H1", H1_error);

    // Multiplier rate

    Vector<double> difference_per_cell_multiplier(
        embedded_triangulation.n_active_cells());

    VectorTools::integrate_difference(
        *embedded_mapping, *embedded_dh, lambda, multiplier_function,
        difference_per_cell_multiplier,
        QGauss<dim>(2 * parameters.fe_embedded_degree + 1),
        VectorTools::L2_norm);

    const double L2_error_multiplier = VectorTools::compute_global_error(
        embedded_triangulation, difference_per_cell_multiplier,
        VectorTools::L2_norm);
    std::cout << "L2 error multiplier:\n" << std::scientific;
    std::cout << L2_error_multiplier << std::endl;
    convergence_table.add_value("L2_multiplier", L2_error_multiplier);

    const double H12error_multiplier =
        NonMatchingUtilities::LM::compute_H_12_norm(
            *embedded_cache, *embedded_dh, *embedded_fe, multiplier_function,
            lambda, QGauss<dim>(2 * parameters.fe_embedded_degree + 1));
    std::cout << "H^1/2 error multiplier: " << std::scientific;
    std::cout << H12error_multiplier << std::endl;
    convergence_table.add_value("H12_multiplier", H12error_multiplier);

    // iteration numbers
    convergence_table.add_value("Iter.", iter);
    iter = 0;  // reset iteration number
  }
}

template <int dim, int spacedim>
void PoissonLM<dim, spacedim>::run() {
  for (cycle = 0; cycle < parameters.n_refinement_cycles; ++cycle) {
    std::cout << "Cycle: " << cycle << std::endl;
    {
      TimerOutput::Scope timer_section(
          timer, "Total time cycle " + std::to_string(cycle));
      setup_grids_and_dofs();

      if (parameters.coupling_strategy == "exact") {
        std::cout << "Start collecting quadratures" << std::endl;

        {
          TimerOutput::Scope timer_section(
              timer, "Compute quadratures on mesh intersections");
          cells_and_quads =
              NonMatching::collect_quadratures_on_overlapped_grids(
                  *space_cache, *embedded_cache,
                  2 * parameters.fe_space_degree + 1, 1e-15);

          double sum = 0.;
          for (const auto &info : cells_and_quads) {
            const auto &q = std::get<2>(info);
            sum += std::accumulate(q.get_weights().begin(),
                                   q.get_weights().end(), 0.);
          }
          std::cout << "Area: " << sum << std::endl;
        }
      }
      std::cout << "Area expected: "
                << GridTools::volume(embedded_triangulation, *embedded_mapping)
                << std::endl;

      setup_coupling();
      assemble_system();
      solve();
    }
    output_results(cycle);
    if (cycle < parameters.n_refinement_cycles - 1) {
      space_triangulation.refine_global(1);
      embedded_triangulation.refine_global(1);
    }
  }
  cells_and_quads.clear();

  convergence_table.set_precision("L2", 3);
  convergence_table.set_precision("H1", 3);
  convergence_table.set_precision("L2_multiplier", 3);
  convergence_table.set_precision("H12_multiplier", 3);
  convergence_table.set_scientific("L2", true);
  convergence_table.set_scientific("H1", true);
  convergence_table.set_scientific("L2_multiplier", true);
  convergence_table.set_scientific("H12_multiplier", true);
  convergence_table.set_scientific("H12_multiplier", true);
  convergence_table.set_scientific("Iter.", false);  // fixed point notation
  convergence_table.evaluate_convergence_rates(
      "L2", "dofs", ConvergenceTable::reduction_rate_log2, spacedim);
  convergence_table.evaluate_convergence_rates(
      "H1", "dofs", ConvergenceTable::reduction_rate_log2, spacedim);
  convergence_table.evaluate_convergence_rates(
      "L2_multiplier", "dofs_emb", ConvergenceTable::reduction_rate_log2, dim);
  convergence_table.evaluate_convergence_rates(
      "H12_multiplier", "dofs_emb", ConvergenceTable::reduction_rate_log2, dim);

  std::string conv_filename = "table_0";
  if (smoothness)
    conv_filename += "1.txt";
  else
    conv_filename += "7.txt";

  std::ofstream table_file(conv_filename);
  convergence_table.write_text(table_file);
}

int main(int argc, char **argv) {
  try {
    {
      std::cout << "Solving in 1D/2D" << std::endl;
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
      bool smoothness = false;
      std::string parameter_file;
      if (argc > 1) {
        parameter_file = argv[1];
        if (parameter_file.find("lm_1d2d_disk.smooth") != std::string::npos)
          smoothness = true;
      } else {
        parameter_file = "parameters.prm";
      }

      PoissonLM<1, 2>::Parameters parameters;
      PoissonLM<1, 2> problem(parameters, smoothness);

      ParameterAcceptor::initialize(parameter_file, "used_parameters.prm");
      problem.run();
    }
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
