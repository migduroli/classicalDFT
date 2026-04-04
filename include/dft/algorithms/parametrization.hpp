#ifndef DFT_ALGORITHMS_PARAMETRIZATION_HPP
#define DFT_ALGORITHMS_PARAMETRIZATION_HPP

#include <armadillo>
#include <variant>

namespace dft::algorithms::parametrization {

  // Density parametrizations map unconstrained variables x in R
  // to physical densities rho > 0, guaranteeing positivity.
  // The chain rule transforms forces accordingly:
  //   f_x = f_rho * drho/dx.

  // Unbounded: rho = rho_min + x^2.
  // Maps R -> (rho_min, inf).

  struct Unbounded {
    double rho_min{1e-18};
  };

  // Bounded: rho = rho_min + range * x^2 / (1 + x^2).
  // Maps R -> (rho_min, rho_max).

  struct Bounded {
    double rho_min{1e-18};
    double rho_max{1.0};
  };

  using Parametrization = std::variant<Unbounded, Bounded>;

  // Convert parameters to density: rho(x).

  [[nodiscard]] inline auto to_density(
      const arma::vec& x, const Parametrization& p
  ) -> arma::vec {
    return std::visit(
        [&](const auto& t) -> arma::vec {
          using T = std::decay_t<decltype(t)>;
          if constexpr (std::is_same_v<T, Unbounded>) {
            return t.rho_min + x % x;
          } else {
            double range = t.rho_max - t.rho_min;
            arma::vec x2 = x % x;
            return t.rho_min + range * x2 / (1.0 + x2);
          }
        },
        p
    );
  }

  // Convert density to parameters: x(rho).

  [[nodiscard]] inline auto from_density(
      const arma::vec& rho, const Parametrization& p
  ) -> arma::vec {
    return std::visit(
        [&](const auto& t) -> arma::vec {
          using T = std::decay_t<decltype(t)>;
          if constexpr (std::is_same_v<T, Unbounded>) {
            return arma::sqrt(arma::clamp(rho - t.rho_min, 0.0, arma::datum::inf));
          } else {
            double range = t.rho_max - t.rho_min;
            arma::vec delta = arma::clamp(rho - t.rho_min, 0.0, range * (1.0 - 1e-14));
            return arma::sqrt(delta / (range - delta));
          }
        },
        p
    );
  }

  // Transform forces from density space to parameter space:
  //   f_x = f_rho * drho/dx.

  [[nodiscard]] inline auto transform_force(
      const arma::vec& f_rho, const arma::vec& x, const Parametrization& p
  ) -> arma::vec {
    return std::visit(
        [&](const auto& t) -> arma::vec {
          using T = std::decay_t<decltype(t)>;
          if constexpr (std::is_same_v<T, Unbounded>) {
            return f_rho % (2.0 * x);
          } else {
            double range = t.rho_max - t.rho_min;
            arma::vec x2 = x % x;
            arma::vec denom = (1.0 + x2) % (1.0 + x2);
            return f_rho % (2.0 * range * x / denom);
          }
        },
        p
    );
  }

}  // namespace dft::algorithms::parametrization

#endif  // DFT_ALGORITHMS_PARAMETRIZATION_HPP
