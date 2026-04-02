#include "dft.hpp"

#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

using namespace dft;

int main() {
  std::vector<double> x1 = {1.0 + 1e-14, 2.5 + 1e-14, 3.0 + 1e-14, 4.0 + 1e-14};
  std::vector<double> x2 = {1.00100001, 2.50010002, 3.00020001, 4.00010003};

  std::cout << std::setprecision(18);

  // Standard summation (accumulate).

  console::info("Standard summation");
  double trivial_1 = std::accumulate(x1.begin(), x1.end(), 0.0);
  std::cout << "  sum(x1) = " << trivial_1 << "\n";

  // Compensated summation algorithms.

  console::info("Kahan-Babuska sum");
  std::cout << "  kahan_sum(x1) = " << math::kahan_sum(x1) << "\n";

  console::info("Neumaier sum");
  std::cout << "  neumaier_sum(x1) = " << math::neumaier_sum(x1) << "\n";

  console::info("Klein sum");
  std::cout << "  klein_sum(x1) = " << math::klein_sum(x1) << "\n";

  // Combined sums.

  std::vector<double> combined;
  combined.reserve(x1.size() + x2.size());
  combined.insert(combined.end(), x1.begin(), x1.end());
  combined.insert(combined.end(), x2.begin(), x2.end());

  console::info("Combined sums");
  std::cout << "  trivial  = " << std::accumulate(combined.begin(), combined.end(), 0.0) << "\n";
  std::cout << "  kahan    = " << math::kahan_sum(combined) << "\n";
  std::cout << "  neumaier = " << math::neumaier_sum(combined) << "\n";
  std::cout << "  klein    = " << math::klein_sum(combined) << "\n";
}
