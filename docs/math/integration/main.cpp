#include "dft.hpp"

#include <iomanip>
#include <iostream>
#include <numbers>

using namespace dft;

int main() {
  std::cout << std::setprecision(12);

  // QAGS: definite integral of exp(-x).

  console::info("QAGS: definite integral");
  auto neg_exp = math::Integrator([](double x) { return std::exp(-x); });
  auto r1 = neg_exp.integrate(0.0, -std::log(0.5));
  std::cout << "  int[0, -ln(0.5)] exp(-x) dx = " << r1.value << "\n";
  std::cout << "  error = " << r1.error << "\n";

  // QNG: fast non-adaptive.

  console::info("QNG: fast non-adaptive");
  auto r2 = neg_exp.integrate_fast(0.0, -std::log(0.5));
  std::cout << "  int[0, -ln(0.5)] exp(-x) dx = " << r2.value << "\n";
  std::cout << "  error = " << r2.error << "\n";

  // QAGIU: upper semi-infinite.

  console::info("QAGIU: upper semi-infinite");
  auto r3 = neg_exp.integrate_upper_infinite(0.0);
  std::cout << "  int[0, +inf] exp(-x) dx = " << r3.value << "\n";
  std::cout << "  error = " << r3.error << "\n";

  // QAGIL: lower semi-infinite.

  console::info("QAGIL: lower semi-infinite");
  auto pos_exp = math::Integrator([](double x) { return std::exp(x); });
  auto r4 = pos_exp.integrate_lower_infinite(0.0);
  std::cout << "  int[-inf, 0] exp(x) dx = " << r4.value << "\n";
  std::cout << "  error = " << r4.error << "\n";

  // QAGI: full infinite (Gaussian).

  console::info("QAGI: full infinite (normal distribution)");
  auto gaussian = math::Integrator([](double x) {
    return std::exp(-x * x * 0.5) / std::sqrt(2.0 * std::numbers::pi);
  });
  auto r5 = gaussian.integrate_infinite();
  std::cout << "  int[-inf, +inf] normal(x) dx = " << r5.value << "\n";
  std::cout << "  error = " << r5.error << "\n";
}
