#include "plot.hpp"
#include "verification.hpp"

#include <dftlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <print>

using namespace dft;

// Stationary states of the one-dimensional square-gradient model
//
//   -kappa rho'' + f0'(rho) - mu = 0 on (0, L),  rho'(0) = rho'(L) = 0,
//   f0(rho) = (rho^2 - 1)^2 / 4,
//
// traced in mu by pseudo-arclength continuation. The uniform branch
// rho^3 - rho = mu folds twice; branches with n interfaces bifurcate from its
// unstable arc and are followed from the bifurcation points.

static void write_branch(const std::string& path, const utils::Branch& b) {
  std::ofstream csv(path);
  csv << "mu,N,Omega,amplitude,a_n,n_minus\n";
  for (std::size_t k = 0; k < b.mu.size(); ++k) {
    csv << std::format(
        "{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{}\n",
        b.mu[k],
        b.mass[k],
        b.omega[k],
        b.amplitude[k],
        b.modal[k],
        b.index[k]
    );
  }
}

int main() {
#ifdef DOC_SOURCE_DIR
  std::filesystem::current_path(DOC_SOURCE_DIR);
#endif
  std::filesystem::create_directories("exports");

#ifdef DFT_HAS_MATPLOTLIB
  matplotlibcpp::backend("Agg");
#endif

  console::info("Canonical continuation: stationary states of the square-gradient model");

  const utils::Problem problem{.length = 20.0, .kappa = 1.0, .nodes = 201};

  const algorithms::continuation::Continuation continuation{
      .initial_step = 0.05,
      .max_step = 0.3,
      .min_step = 1e-6,
      .newton = {.max_iterations = 20, .tolerance = 1e-9},
  };

  std::println(
      std::cout,
      "  L = {}, kappa = {}, K = {} nodes, h = {}",
      problem.length,
      problem.kappa,
      problem.nodes,
      problem.spacing()
  );

  auto results = utils::run(problem, continuation, 3);
  const auto& uniform = results.uniform;

  // Verification against the closed forms.

  console::info("Verification");
  auto rows = utils::verification(problem, continuation, results);
  utils::print_rows(rows);
  const auto failed = std::ranges::count_if(rows, [](const auto& r) { return !r.passed(); });
  std::println(std::cout, "\n  {} / {} checks passed", rows.size() - static_cast<std::size_t>(failed), rows.size());

  write_branch("exports/uniform.csv", uniform);
  for (const auto& b : results.arms)
    write_branch(std::format("exports/branch_n{}{}.csv", b.mode, b.sign > 0 ? "p" : "m"), b);
  std::println(std::cout, "  Saved branches to exports/*.csv");

#ifdef DFT_HAS_MATPLOTLIB
  plot::style();
  plot::s_curve(problem, uniform);
  plot::swallowtail(problem, results);
  plot::branches(problem, results);
  plot::pitchfork_zoom(problem, results);
  plot::walk(problem, results);
  plot::canonical(problem, results);
  plot::nested_pitchforks(problem, results);
  plot::branch_count(results);
  plot::landau(results);
#endif

  std::println(std::cout, "\nDone.");
  return failed == 0 ? 0 : 1;
}
