#include <classicaldft>

int main()
{
  using namespace dft_core::numerics::arithmetic;
  using namespace dft_core::io;

  // region set-up:
  auto x_input_1 = std::vector<double>{1.0 + 1E-14, 2.5 + 1E-14, 3.0 + 1E-14, 4.0 + 1E-14};
  auto x_input_2 = std::vector<double>{1.00100001, 2.50010002, 3.00020001, 4.00010003};

  auto trivial_sum = summation::standard_vector_sum(x_input_1);
  console::info("Test 1: Sum from scratch");
  std::cout << "Trivial sum [1]: " << trivial_sum << std::endl;
  // endregion

  // region free-methods:

  double kb_sum; std::vector<double> kb_err;
  std::tie(kb_sum, kb_err) = summation::kahan_babuska_sum(x_input_1);
  std::cout << "Kahan-Babuska sum [1]: " << kb_sum << std::endl;
  std::cout << "Kahan-Babuska err [1] = " << kb_err.front() << std::endl;

  double neum_sum; std::vector<double> neum_err;
  std::tie(neum_sum, neum_err) = summation::kahan_babuska_neumaier_sum(x_input_1);
  std::cout << "Kahan-Babuska-Neumaier sum [1]: " << neum_sum << std::endl;
  std::cout << "Kahan-Babuska-Neumaier err [1]: " << neum_err.front() << std::endl;

  double klein_sum; std::vector<double> klein_err;
  std::tie(klein_sum, klein_err) = summation::kahan_babuska_klein_sum(x_input_1);
  std::cout << "Kahan-Babuska-Klein sum [1]: " << klein_sum << std::endl;
  std::cout << "Kahan-Babuska-Klein err [1/a]: " << klein_err[0] << std::endl;
  std::cout << "Kahan-Babuska-Klein err [1/b]: " << klein_err[1] << std::endl;

  std::cout << std::endl;
  trivial_sum += summation::standard_vector_sum(x_input_2);
  std::tie(kb_sum, kb_err) = summation::kahan_babuska_sum(x_input_2, kb_sum, kb_err);
  std::tie(neum_sum, neum_err) = summation::kahan_babuska_neumaier_sum(x_input_2, neum_sum, neum_err);
  std::tie(klein_sum, klein_err) = summation::kahan_babuska_klein_sum(x_input_2, klein_sum, klein_err);

  console::info("Test 2: Continuation of previous sum");
  std::cout << "Trivial sum [2]: " << trivial_sum << std::endl;

  std::cout << "Kahan-Babuska sum [2]: " << kb_sum << std::endl;
  std::cout << "Kahan-Babuska err [2]: " << kb_err.front() << std::endl;

  std::cout << "Kahan-Babuska-Neumaier sum [2]: " << neum_sum << std::endl;
  std::cout << "Kahan-Babuska-Neumaier err [2]: " << neum_err.front() << std::endl;

  std::cout << "Kahan-Babuska-Klein sum [1]: " << klein_sum << std::endl;
  std::cout << "Kahan-Babuska-Klein err [1/a]: " << klein_err[0] << std::endl;
  std::cout << "Kahan-Babuska-Klein err [1/b]: " << klein_err[1] << std::endl;

  std::cout << std::endl;
  // endregion

  // region class:
  console::info("Test 3: Using CompensatedSum class");
  summation::CompensatedSum x_cs;

  x_cs += x_input_1;
  std::cout << "Testing the operator '+=' [1] = " << x_cs << std::endl;

  x_cs += x_input_2;
  std::cout << "Testing the operator '+=' [2] = " << x_cs << std::endl;
}