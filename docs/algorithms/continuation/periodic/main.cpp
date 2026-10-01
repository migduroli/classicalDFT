#include "plot.hpp"
#include "verification.hpp"

#include <dftlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <print>

using namespace dft;

static void write_grand_branch(const std::string& path, const periodic::GrandBranch& branch) {
  std::ofstream csv(path);
  csv << "mu,N,Omega,a_m,eta,n_minus\n";
  for (std::size_t k = 0; k < branch.mu.size(); ++k) {
    csv << std::format(
        "{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{}\n",
        branch.mu[k],
        branch.mass[k],
        branch.omega[k],
        branch.modal[k],
        branch.multiplier[k],
        branch.index[k]
    );
  }
}

static void write_canonical_branch(const std::string& path, const periodic::CanonicalBranch& branch) {
  std::ofstream csv(path);
  csv << "N,mu,F,a_1,eta,n_minus\n";
  for (std::size_t k = 0; k < branch.mass.size(); ++k) {
    csv << std::format(
        "{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{}\n",
        branch.mass[k],
        branch.mu[k],
        branch.helmholtz[k],
        branch.modal[k],
        branch.multiplier[k],
        branch.index[k]
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

  console::info("Periodic continuation: stationary states of the square-gradient model");

  const periodic::Problem problem{.length = 20.0, .kappa = 1.0, .nodes = 200};
  const algorithms::continuation::Continuation continuation{
      .initial_step = 0.05,
      .max_step = 0.3,
      .min_step = 1e-6,
      .newton = {.max_iterations = 24, .tolerance = 1e-9},
  };

  std::println(
      std::cout,
      "  L = {}, kappa = {}, K = {} periodic nodes, h = {}",
      problem.length,
      problem.kappa,
      problem.nodes,
      problem.spacing()
  );

  auto results = periodic::run(problem, continuation, 3);
  for (const auto& branch : results.grand)
    write_grand_branch(std::format("exports/branch_m{}.csv", branch.mode), branch);
  for (std::size_t k = 0; k < results.canonical.size(); ++k)
    write_canonical_branch(std::format("exports/canonical_{}.csv", k == 0 ? "plus" : "minus"), results.canonical[k]);
  std::println(std::cout, "  Saved branches to exports/*.csv");

  console::info("Verification");
  auto rows = periodic::verification(problem, continuation, results);
  periodic::print_rows(rows);
  const auto failed = std::ranges::count_if(rows, [](const auto& row) { return !row.passed(); });
  std::println(
      std::cout,
      "\n  {} / {} checks passed",
      rows.size() - static_cast<std::size_t>(failed),
      rows.size()
  );

#ifdef DFT_HAS_MATPLOTLIB
  periodic::plot::style();
  periodic::plot::s_curve(problem, results.uniform);
  periodic::plot::swallowtail(problem, results);
  periodic::plot::branches(problem, results);
  periodic::plot::pitchfork_zoom(problem, results);
  periodic::plot::walk(problem, results);
  periodic::plot::canonical(problem, results);
  periodic::plot::nested_pitchforks(problem, results);
  periodic::plot::branch_count(results);
  periodic::plot::translation(problem, results);
#endif

  std::println(std::cout, "\nDone.");
  return failed == 0 ? 0 : 1;
}
