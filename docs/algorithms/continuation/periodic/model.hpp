#pragma once

#include <armadillo>
#include <cmath>
#include <dftlib>
#include <numbers>

namespace periodic {

  using dft::algorithms::continuation::Residual;

  // Square-gradient model on a periodic interval of length L:
  //
  //   -kappa rho'' + rho^3 - rho - mu = 0,  rho(0) = rho(L),  rho'(0) = rho'(L).
  //
  // K distinct nodes x_i = i h, h = L / K, represent the periodic box. There
  // is no duplicated endpoint and every quadrature weight is one.

  struct PhaseGauge {
    arma::vec reference;
    arma::vec tangent;

    [[nodiscard]] auto condition(const arma::vec& y, double h) const -> double {
      return h * arma::dot(y - reference, tangent);
    }
  };

  struct Problem {
    double length{20.0};
    double kappa{1.0};
    arma::uword nodes{200};

    [[nodiscard]] auto spacing() const -> double { return length / static_cast<double>(nodes); }

    [[nodiscard]] auto positions() const -> arma::vec {
      return spacing() * arma::linspace<arma::vec>(0.0, static_cast<double>(nodes - 1), nodes);
    }

    [[nodiscard]] auto laplacian(const arma::vec& y) const -> arma::vec {
      const double h2 = spacing() * spacing();
      arma::vec d(nodes);
      for (arma::uword i = 0; i < nodes; ++i) {
        const arma::uword left = i == 0 ? nodes - 1 : i - 1;
        const arma::uword right = (i + 1) % nodes;
        d(i) = (y(left) - 2.0 * y(i) + y(right)) / h2;
      }
      return d;
    }

    [[nodiscard]] auto derivative(const arma::vec& y) const -> arma::vec {
      const double h = spacing();
      arma::vec d(nodes);
      for (arma::uword i = 0; i < nodes; ++i) {
        const arma::uword left = i == 0 ? nodes - 1 : i - 1;
        const arma::uword right = (i + 1) % nodes;
        d(i) = (y(right) - y(left)) / (2.0 * h);
      }
      return d;
    }

    [[nodiscard]] auto residual(const arma::vec& y, double mu) const -> arma::vec {
      return -kappa * laplacian(y) + arma::pow(y, 3) - y - mu;
    }

    [[nodiscard]] auto jacobian(const arma::vec& y) const -> arma::mat {
      const double c = kappa / (spacing() * spacing());
      arma::mat J(nodes, nodes, arma::fill::zeros);
      J.diag() = 2.0 * c + 3.0 * arma::square(y) - 1.0;
      for (arma::uword i = 0; i < nodes; ++i) {
        J(i, i == 0 ? nodes - 1 : i - 1) = -c;
        J(i, (i + 1) % nodes) = -c;
      }
      return J;
    }

    [[nodiscard]] auto mass(const arma::vec& y) const -> double { return spacing() * arma::accu(y); }

    [[nodiscard]] auto grand_potential(const arma::vec& y, double mu) const -> double {
      const double h = spacing();
      double gradient = 0.0;
      for (arma::uword i = 0; i < nodes; ++i) {
        const double difference = y((i + 1) % nodes) - y(i);
        gradient += difference * difference;
      }
      return h * arma::accu(0.25 * arma::square(arma::square(y) - 1.0) - mu * y)
             + 0.5 * kappa * gradient / h;
    }

    [[nodiscard]] auto helmholtz(const arma::vec& y) const -> double { return grand_potential(y, 0.0); }

    [[nodiscard]] auto cosine(int mode) const -> arma::vec {
      return arma::cos(2.0 * std::numbers::pi * mode * positions() / length);
    }

    [[nodiscard]] auto sine(int mode) const -> arma::vec {
      return arma::sin(2.0 * std::numbers::pi * mode * positions() / length);
    }

    [[nodiscard]] auto modal_cosine(const arma::vec& y, int mode) const -> double {
      return 2.0 / length * spacing() * arma::dot(y - mass(y) / length, cosine(mode));
    }

    [[nodiscard]] auto modal_sine(const arma::vec& y, int mode) const -> double {
      return 2.0 / length * spacing() * arma::dot(y - mass(y) / length, sine(mode));
    }

    [[nodiscard]] auto phase_gauge(const arma::vec& reference) const -> PhaseGauge {
      return {.reference = reference, .tangent = derivative(reference)};
    }

    [[nodiscard]] auto spectrum(const arma::vec& y) const -> arma::vec { return arma::eig_sym(jacobian(y)); }

    [[nodiscard]] auto index(const arma::vec& y) const -> int {
      return static_cast<int>(arma::accu(spectrum(y) < -1e-10));
    }

    // The canonical index excludes the constant mass direction and, for a
    // non-uniform periodic profile, its translational tangent rho_x.
    [[nodiscard]] auto constrained_index(const arma::vec& y) const -> int {
      arma::mat B(nodes, 2, arma::fill::zeros);
      B.col(0).fill(1.0 / std::sqrt(static_cast<double>(nodes)));
      arma::vec translation = derivative(y);
      translation -= B.col(0) * arma::dot(B.col(0), translation);
      arma::uword columns = 1;
      if (arma::norm(translation) > 1e-8) {
        B.col(1) = translation / arma::norm(translation);
        columns = 2;
      }
      const arma::mat Q = B.head_cols(columns);
      arma::mat projected = arma::eye(nodes, nodes) - Q * Q.t();
      arma::mat A = projected * jacobian(y) * projected;
      A = 0.5 * (A + A.t());
      return static_cast<int>(arma::accu(arma::eig_sym(A) < -1e-8));
    }

    [[nodiscard]] auto periodic_eigenvalue(int mode) const -> double {
      const double s = std::sin(std::numbers::pi * mode / static_cast<double>(nodes));
      return 4.0 * s * s / (spacing() * spacing());
    }

    // x = [y; mu; eta], with eta the numerical multiplier that makes the
    // phase-fixed canonical system square. A physical state has eta = 0.
    [[nodiscard]] auto canonical(const PhaseGauge& gauge) const -> Residual {
      return [this, &gauge](const arma::vec& x, double n) {
        const arma::vec y = x.head(nodes);
        const double mu = x(nodes);
        const double eta = x(nodes + 1);
        arma::vec r(nodes + 2);
        r.head(nodes) = residual(y, mu) + eta * gauge.tangent;
        r(nodes) = mass(y) - n;
        r(nodes + 1) = gauge.condition(y, spacing());
        return r;
      };
    }

    // x = [y; eta], continued in mu. The sine coefficient fixes the Fourier
    // phase of a mode-m branch; eta vanishes on the physical branch.
    [[nodiscard]] auto grand_canonical(int mode) const -> Residual {
      const arma::vec phase_tangent = sine(mode);
      return [this, mode, phase_tangent](const arma::vec& x, double mu) {
        const arma::vec y = x.head(nodes);
        const double eta = x(nodes);
        arma::vec r(nodes + 1);
        r.head(nodes) = residual(y, mu) + eta * phase_tangent;
        r(nodes) = modal_sine(y, mode);
        return r;
      };
    }
  };

  namespace exact {

    inline auto bifurcation_density(double kappa_d) -> double {
      return std::sqrt((1.0 - kappa_d) / 3.0);
    }

    inline auto surface_tension(double kappa) -> double {
      return 2.0 * std::sqrt(2.0 * kappa) / 3.0;
    }

  } // namespace exact

} // namespace periodic
