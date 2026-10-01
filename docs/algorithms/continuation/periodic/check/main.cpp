#include "verification.hpp"

#include <dftlib>
#include <iostream>
#include <print>

using namespace dft;

int main() {
  const periodic::Problem problem{.length = 20.0, .kappa = 1.0, .nodes = 200};
  const algorithms::continuation::Continuation continuation{
      .initial_step = 0.05,
      .max_step = 0.3,
      .min_step = 1e-6,
      .newton = {.max_iterations = 24, .tolerance = 1e-9},
  };

  const auto results = periodic::run(problem, continuation, 3);
  const auto rows = periodic::verification(problem, continuation, results);
  periodic::print_rows(rows);
  const auto failed = std::ranges::count_if(rows, [](const auto& row) { return !row.passed(); });
  std::println(std::cout, "\n{} / {} checks passed", rows.size() - static_cast<std::size_t>(failed), rows.size());
  return failed == 0 ? 0 : 1;
}
