#include <classicaldft>

int main()
{
  using namespace dft_core::numerics::arithmetic;

  // region set-up:
  auto x_input_1 = std::vector<double>{1.0 + 1E-14, 2.5 + 1E-14, 3.0 + 1E-14, 4.0 + 1E-14};
  auto x_input_2 = std::vector<double>{1.00100001, 2.50010002, 3.00020001, 4.00010003};

  auto trivial_sum = summation::standard_vector_sum(x_input_1);
  console::write_line("Test 1: Sum from scratch");
  console::write("Trivial sum [1]: "); console::write_line(trivial_sum);
  // endregion

  // region free-methods:

  double kb_sum; std::vector<double> kb_err;
  std::tie(kb_sum, kb_err) = summation::kahan_babuska_sum(x_input_1);
  console::write("Kahan-Babuska sum [1]:"); console::write_line(kb_sum);
  console::write("Kahan-Babuska err [1] = "); console::write_line(kb_err.front());

  double neum_sum; std::vector<double> neum_err;
  std::tie(neum_sum, neum_err) = summation::kahan_babuska_neumaier_sum(x_input_1);
  console::write("Kahan-Babuska-Neumaier sum [1]: "); console::write_line(neum_sum);
  console::write("Kahan-Babuska-Neumaier err [1]: "); console::write_line(neum_err.front());

  double klein_sum; std::vector<double> klein_err;
  std::tie(klein_sum, klein_err) = summation::kahan_babuska_klein_sum(x_input_1);
  console::write("Kahan-Babuska-Klein sum [1]: "); console::write(klein_sum);
  console::write("Kahan-Babuska-Klein err [1/a]: "); console::write_line(klein_err[0]);
  console::write("Kahan-Babuska-Klein err [1/b]: "); console::write_line(klein_err[1]);

  console::new_line();
  trivial_sum += summation::standard_vector_sum(x_input_2);
  std::tie(kb_sum, kb_err) = summation::kahan_babuska_sum(x_input_2, kb_sum, kb_err);
  std::tie(neum_sum, neum_err) = summation::kahan_babuska_neumaier_sum(x_input_2, neum_sum, neum_err);
  std::tie(klein_sum, klein_err) = summation::kahan_babuska_klein_sum(x_input_2, klein_sum, klein_err);

  console::write_line("Test 2: Continuation of previous sum");
  console::write("Trivial sum [2]: "); console::write_line(trivial_sum);

  console::write("Kahan-Babuska sum [2]: "); console::write_line(kb_sum);
  console::write("Kahan-Babuska err [2]: "); console::write_line(kb_err.front());

  console::write("Kahan-Babuska-Neumaier sum [2]: "); console::write_line(neum_sum);
  console::write("Kahan-Babuska-Neumaier err [2]: "); console::write_line(neum_err.front());

  console::write("Kahan-Babuska-Klein sum [1]: "); console::write_line(klein_sum);
  console::write("Kahan-Babuska-Klein err [1/a]: "); console::write_line(klein_err[0]);
  console::write("Kahan-Babuska-Klein err [1/b]: "); console::write_line(klein_err[1]);

  console::new_line();
  // endregion

  // region class:
  console::write_line("Test 3: Using CompensatedSum class");
  summation::CompensatedSum x_cs;

  x_cs += x_input_1;
  console::write("Testing the operator '+=' [1] = "); console::write_line(x_cs);

  x_cs += x_input_2;
  console::write("Testing the operator '+=' [2] = "); console::write_line(x_cs);

  console::wait();
}