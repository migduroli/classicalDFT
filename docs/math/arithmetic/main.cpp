#include <dftlib>
#include <iostream>
#include <numeric>
#include <print>
#include <vector>

using namespace dft;

int main() {
  std::vector<double> x1 = {1.0 + 1e-14, 2.5 + 1e-14, 3.0 + 1e-14, 4.0 + 1e-14};
  std::vector<double> x2 = {1.00100001, 2.50010002, 3.00020001, 4.00010003};

  // Standard summation (accumulate).

  console::info("Standard summation");
  double trivial_1 = std::accumulate(x1.begin(), x1.end(), 0.0);
  std::println(std::cout, "  sum(x1) = {:.18}", trivial_1);

  // Compensated summation algorithms.

  console::info("Kahan-Babuska sum");
  std::println(std::cout, "  kahan_sum(x1) = {:.18}", math::kahan_sum(x1));

  console::info("Neumaier sum");
  std::println(std::cout, "  neumaier_sum(x1) = {:.18}", math::neumaier_sum(x1));

  console::info("Klein sum");
  std::println(std::cout, "  klein_sum(x1) = {:.18}", math::klein_sum(x1));

  // Combined sums.

  std::vector<double> combined;
  combined.reserve(x1.size() + x2.size());
  combined.insert(combined.end(), x1.begin(), x1.end());
  combined.insert(combined.end(), x2.begin(), x2.end());

  console::info("Combined sums");
  std::println(std::cout, "  trivial  = {:.18}", std::accumulate(combined.begin(), combined.end(), 0.0));
  std::println(std::cout, "  kahan    = {:.18}", math::kahan_sum(combined));
  std::println(std::cout, "  neumaier = {:.18}", math::neumaier_sum(combined));
  std::println(std::cout, "  klein    = {:.18}", math::klein_sum(combined));
}
