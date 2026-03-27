#include <classicaldft>

namespace example
{
class TestProblem
{
 private:
  double param_ = 0.1;

 public:
  explicit TestProblem(double param = 0.1) : param_(param) {}
  double NegativeExp(double x) const { return param_ * exp(-x); }
  double PositiveExp(double x) const { return param_ * exp(x); }
  double NormalDist(double x) const { return param_ * exp(-x * x * 0.5) / sqrt(2 * M_PI); }
};

double NegativeExpFunction(double x, const std::vector<double>& param) { return param[0] * param[1] * exp(-x); }
}

int main()
{
  using namespace dft_core::numerics::integration;
  using namespace dft_core::io;

  auto problem = example::TestProblem(1.0);
  auto integrator = Integrator<example::TestProblem>(problem, &example::TestProblem::NegativeExp);

  console::info("Testing passing object->method:");
  std::cout << "integrator.f(2*M_PI) = " << integrator.function(M_PI_2) << std::endl;
  std::cout << "Integrator<LocalProblem>::integrand_function(2*M_PI, &integrator) = "
            << Integrator<example::TestProblem>::integrand_function(M_PI_2, &integrator) << std::endl;

  auto result = integrator.definite_integral(0, -log(0.5));
  std::cout << std::endl;
  console::info("Testing integration methods: QAGS");
  std::cout << "int[0, -log(0.5)] exp(-x) dx = " << integrator.numerical_result() << std::endl;
  std::cout << "numerical_error =  " << integrator.numerical_error() << std::endl;

  auto result_fast = integrator.definite_integral_fast(0, -log(0.5));
  std::cout << std::endl;
  console::info("Testing integration methods: QNG");
  std::cout << "int[0, -log(0.5)] exp(-x) dx = " << integrator.numerical_result() << std::endl;
  std::cout << "numerical_error =  " << integrator.numerical_error() << std::endl;

  auto result_semi_inf = integrator.upper_semi_infinite_integral(0);
  std::cout << std::endl;
  console::info("Testing integration methods: QAGIU");
  std::cout << "int[0, +inf] exp(-x) dx = " << integrator.numerical_result() << std::endl;
  std::cout << "numerical_error =  " << integrator.numerical_error() << std::endl;

  auto integrator_neg = Integrator<example::TestProblem>(problem, &example::TestProblem::PositiveExp);
  auto result_semi_neg = integrator_neg.lower_semi_infinite_integral(0);
  std::cout << std::endl;
  console::info("Testing integration methods: QAGIL");
  std::cout << "int[-inf, 0] exp(x) dx = " << integrator.numerical_result() << std::endl;
  std::cout << "numerical_error =  " << integrator.numerical_error() << std::endl;

  auto integrator_gauss = Integrator<example::TestProblem>(problem, &example::TestProblem::NormalDist);
  auto result_gauss = integrator_gauss.full_infinite_integral();
  std::cout << std::endl;
  console::info("Testing integration methods: QAGI");
  std::cout << "int[-inf, +inf] normal(x) dx = " << integrator.numerical_result() << std::endl;
  std::cout << "numerical_error =  " << integrator.numerical_error() << std::endl;

  auto parameters = std::vector<double>{ 0.5, 2 };
  auto func_integrator = FunctionIntegrator<std::vector<double>>(&example::NegativeExpFunction, parameters);

  result = func_integrator.definite_integral(0, -log(0.5));
  std::cout << std::endl;
  console::info("Testing function-integration vector parameters: p = [0.5, 2]");
  std::cout << "int[0, -log(0.5)] p[0] * p[1] * exp(-x) dx = " << func_integrator.numerical_result() << std::endl;
  std::cout << "numerical_error =  " << func_integrator.numerical_error() << std::endl;
}