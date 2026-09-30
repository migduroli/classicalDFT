#include "plot.hpp"
#include "utils.hpp"

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
  csv << "mu,N,Omega,amplitude,n_minus\n";
  for (std::size_t k = 0; k < b.mu.size(); ++k) {
    csv << std::format(
        "{:.12e},{:.12e},{:.12e},{:.12e},{}\n",
        b.mu[k],
        b.mass[k],
        b.omega[k],
        b.amplitude[k],
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

  console::info("Continuation: stationary states of the square-gradient model");

  const utils::Problem problem{.length = 20.0, .kappa = 1.0, .nodes = 201};

  const algorithms::continuation::Continuation cont{
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

  // Uniform branch.

  console::info("Tracing the uniform branch");
  auto uniform = utils::trace_uniform(problem, cont, 1.35);
  std::println(
      std::cout,
      "  {} points, {} folds, {} bifurcation points",
      uniform.curve.size(),
      uniform.folds.size(),
      uniform.bifurcations.size()
  );
  for (const auto& f : uniform.folds)
    std::println(std::cout, "  fold          rho = {:+.10f}  mu = {:+.10f}", f.rho_bar, f.mu);
  for (const auto& e : uniform.bifurcations)
    std::println(std::cout, "  bifurcation   rho = {:+.10f}  mu = {:+.10f}  n = {}", e.rho_bar, e.mu, e.mode);

  // Branches with n = 1, 2, 3 interfaces, from the bifurcation point at
  // rho < 0 to its mirror at rho > 0.

  std::vector<utils::Branch> branches;
  for (int n = 1; n <= 3; ++n) {
    const utils::Event* start = nullptr;
    const utils::Event* end = nullptr;
    for (const auto& e : uniform.bifurcations) {
      if (e.mode != n)
        continue;
      (e.rho_bar < 0.0 ? start : end) = &e;
    }
    console::info(std::format("Tracing the branch with n = {} interfaces", n));
    auto b = utils::trace_bifurcating(problem, cont, *start, *end, 0.3, 1.0, 2000);
    auto [lo, hi] = std::ranges::minmax_element(b.index);
    std::println(
        std::cout,
        "  {} points, n_minus in [{}, {}], Omega at the widest state {:.6f}",
        b.curve.size(),
        *lo,
        *hi,
        b.omega[static_cast<std::size_t>(std::ranges::max_element(b.amplitude) - b.amplitude.begin())]
    );
    branches.push_back(std::move(b));
  }

  // The n = 1 branch traced again with N as the parameter, from the centred
  // interface at mu = 0 towards both walls.

  console::info("Tracing the n = 1 branch at fixed N");
  arma::vec kink = utils::interface_state(problem);
  std::vector<utils::CanonicalBranch> canonical;
  for (double direction : {+1.0, -1.0})
    canonical.push_back(utils::trace_canonical(problem, cont, kink, 0.0, direction, 1.5 * problem.length));
  for (const auto& c : canonical) {
    auto [lo, hi] = std::ranges::minmax_element(c.index);
    std::println(
        std::cout,
        "  {} points, N from {:+.4f} to {:+.4f}, mu at the end {:+.6f}, n_minus at fixed N in [{}, {}]",
        c.mass.size(),
        c.mass.front(),
        c.mass.back(),
        c.mu.back(),
        *lo,
        *hi
    );
  }

  // Verification against the closed forms.

  console::info("Verification");
  auto rows = utils::verification(problem, uniform, branches.front());
  utils::print_rows(rows);
  const auto failed = std::ranges::count_if(rows, [](const auto& r) { return !r.passed(); });
  std::println(std::cout, "\n  {} / {} checks passed", rows.size() - static_cast<std::size_t>(failed), rows.size());

  write_branch("exports/uniform.csv", uniform);
  for (std::size_t b = 0; b < branches.size(); ++b)
    write_branch(std::format("exports/branch_n{}.csv", b + 1), branches[b]);
  std::println(std::cout, "  Saved branches to exports/*.csv");

#ifdef DFT_HAS_MATPLOTLIB
  plot::s_curve(problem, uniform);
  plot::swallowtail(problem, uniform);
  plot::branches(problem, uniform, branches);
  plot::profiles(problem, branches);
  plot::canonical(problem, uniform, branches.front(), canonical);
#endif

  std::println(std::cout, "\nDone.");
  return failed == 0 ? 0 : 1;
}
