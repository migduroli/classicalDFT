#ifndef DFT_ALGORITHMS_ALIAS_HPP
#define DFT_ALGORITHMS_ALIAS_HPP

#include <armadillo>
#include <variant>

namespace dft::algorithms::alias {

  // Alias transforms map between an unconstrained design variable x
  // and the physical density rho, guaranteeing rho > 0 everywhere.
  // The chain rule transforms forces accordingly:
  //   f_x = f_rho * drho/dx.

  // Unbounded alias: rho = rho_min + x^2.
  // Maps all of R to (rho_min, inf), prevents negative densities.

  struct Unbounded {
    double rho_min{1e-18};
  };

  // Bounded alias: rho = rho_min + range * x^2 / (1 + x^2).
  // Maps all of R to (rho_min, rho_max), prevents both negative
  // densities and close-packing divergences (eta >= 1).

  struct Bounded {
    double rho_min{1e-18};
    double rho_max{1.0};
  };

  using AliasTransform = std::variant<Unbounded, Bounded>;

  // Density from alias: rho(x).

  [[nodiscard]] inline auto density_from_alias(
      const arma::vec& x, const AliasTransform& transform
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
        transform
    );
  }

  // Alias from density: x(rho).

  [[nodiscard]] inline auto alias_from_density(
      const arma::vec& rho, const AliasTransform& transform
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
        transform
    );
  }

  // Transform forces from density space to alias space:
  //   f_x = f_rho * drho/dx.

  [[nodiscard]] inline auto alias_force(
      const arma::vec& f_rho, const arma::vec& x, const AliasTransform& transform
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
        transform
    );
  }

}  // namespace dft::algorithms::alias

#endif  // DFT_ALGORITHMS_ALIAS_HPP
