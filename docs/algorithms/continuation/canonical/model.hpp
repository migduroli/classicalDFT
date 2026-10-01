#pragma once

#include <algorithm>
#include <armadillo>
#include <cmath>
#include <dftlib>
#include <numbers>

namespace utils {

  using dft::algorithms::continuation::Residual;

  // Square-gradient model on (0, L) with Neumann walls:
  //
  //   -kappa rho'' + f0'(rho) - mu = 0,   f0(rho) = (rho^2 - 1)^2 / 4.
  //
  // Second-order differences on K nodes x_i = i h, h = L / (K - 1). The
  // mirrored end nodes y_{-1} = y_1 and y_K = y_{K-2} impose rho' = 0.

  struct Problem {
    double length{20.0};
    double kappa{1.0};
    arma::uword nodes{201};

    [[nodiscard]] auto spacing() const -> double { return length / static_cast<double>(nodes - 1); }

    [[nodiscard]] auto positions() const -> arma::vec { return arma::linspace(0.0, length, nodes); }

    // Trapezoid weights: 1/2 at the end nodes, 1 elsewhere.
    [[nodiscard]] auto weights() const -> arma::vec {
      arma::vec w(nodes, arma::fill::ones);
      w(0) = 0.5;
      w(nodes - 1) = 0.5;
      return w;
    }

    // (D2 y)_i with mirrored end nodes.
    [[nodiscard]] auto laplacian(const arma::vec& y) const -> arma::vec {
      const arma::uword k = nodes;
      const double h2 = spacing() * spacing();
      arma::vec d(k);
      d(0) = 2.0 * (y(1) - y(0)) / h2;
      d(k - 1) = 2.0 * (y(k - 2) - y(k - 1)) / h2;
      d.subvec(1, k - 2) = (y.subvec(0, k - 3) - 2.0 * y.subvec(1, k - 2) + y.subvec(2, k - 1)) / h2;
      return d;
    }

    // F_i(y, mu) = -kappa (D2 y)_i + y_i^3 - y_i - mu.
    [[nodiscard]] auto residual(const arma::vec& y, double mu) const -> arma::vec {
      return -kappa * laplacian(y) + arma::pow(y, 3) - y - mu;
    }

    // Jacobian dF/dy: tridiagonal, symmetric in the trapezoid inner product.
    [[nodiscard]] auto jacobian(const arma::vec& y) const -> arma::mat {
      const arma::uword k = nodes;
      const double c = kappa / (spacing() * spacing());
      arma::mat J(k, k, arma::fill::zeros);
      J.diag() = 2.0 * c + 3.0 * arma::square(y) - 1.0;
      J.diag(1).fill(-c);
      J.diag(-1).fill(-c);
      J(0, 1) = -2.0 * c;
      J(k - 1, k - 2) = -2.0 * c;
      return J;
    }

    // W^{1/2} J W^{-1/2}: similar to J and symmetric, so arma::eig_sym applies.
    // The discrete Hessian of Omega is h W J, which has the same inertia.
    [[nodiscard]] auto symmetric_hessian(const arma::vec& y) const -> arma::mat {
      arma::vec s = arma::sqrt(weights());
      arma::mat J = jacobian(y);
      return arma::diagmat(s) * J * arma::diagmat(1.0 / s);
    }

    [[nodiscard]] auto mass(const arma::vec& y) const -> double { return spacing() * arma::dot(weights(), y); }

    // Omega = h sum_i w_i [f0(y_i) - mu y_i] + (kappa / 2h) sum_i (y_{i+1} - y_i)^2.
    // The gradient of this sum is h W F, so its stationary points are the zeros of F.
    [[nodiscard]] auto grand_potential(const arma::vec& y, double mu) const -> double {
      const double h = spacing();
      arma::vec f0 = 0.25 * arma::square(arma::square(y) - 1.0);
      arma::vec dy = arma::diff(y);
      return h * arma::dot(weights(), f0 - mu * y) + 0.5 * kappa * arma::dot(dy, dy) / h;
    }

    // ||y - mean(y)|| in the trapezoid norm.
    [[nodiscard]] auto amplitude(const arma::vec& y) const -> double {
      arma::vec d = y - mass(y) / length;
      return std::sqrt(spacing() * arma::dot(weights(), arma::square(d)));
    }

    // Signed amplitude of the Neumann mode n:
    // a_n = (2 / L) h sum_i w_i (y_i - rho_bar) cos(n pi x_i / L).
    // The reflection x -> L - x (odd n) or the shift by L / n (even n) maps
    // a_n to -a_n, while N, Omega and ||y - rho_bar|| are unchanged.
    [[nodiscard]] auto modal_amplitude(const arma::vec& y, int n) const -> double {
      arma::vec c = arma::cos(n * std::numbers::pi * positions() / length);
      return 2.0 / length * spacing() * arma::dot(weights(), (y - mass(y) / length) % c);
    }

    // Image of y under the symmetry that exchanges the two arms of the
    // pitchfork of mode n: the reflection x -> L - x for odd n, and for even n
    // the shift by L / n of the even 2L-periodic extension of y.
    [[nodiscard]] auto arm_image(const arma::vec& y, int n) const -> arma::vec {
      if (n % 2 == 1)
        return arma::reverse(y);
      const arma::uword period = 2 * (nodes - 1);
      const arma::uword shift = (nodes - 1) / static_cast<arma::uword>(n);
      arma::vec out(nodes);
      for (arma::uword i = 0; i < nodes; ++i) {
        arma::uword j = (i + shift) % period;
        out(i) = y(j < nodes ? j : period - j);
      }
      return out;
    }

    [[nodiscard]] auto spectrum(const arma::vec& y) const -> arma::vec { return arma::eig_sym(symmetric_hessian(y)); }

    [[nodiscard]] auto index(const arma::vec& y) const -> int {
      return static_cast<int>(arma::accu(spectrum(y) < 0.0));
    }

    // Index at fixed N: negative eigenvalues of the Hessian restricted to
    // mass-preserving perturbations, in the symmetrised coordinates.
    [[nodiscard]] auto constrained_index(const arma::vec& y) const -> int {
      arma::vec c = arma::sqrt(weights());
      c /= arma::norm(c);
      arma::mat P = arma::eye(nodes, nodes) - c * c.t();
      arma::mat S = symmetric_hessian(y);
      arma::mat A = P * S * P;
      A = 0.5 * (A + A.t());
      arma::vec eigenvalues = arma::eig_sym(A);
      return static_cast<int>(arma::accu(eigenvalues < -1e-10));
    }

    // Eigenvalues of -D2 with mirrored end nodes: (4 / h^2) sin^2(n pi h / 2L).
    [[nodiscard]] auto laplacian_eigenvalue(int n) const -> double {
      const double h = spacing();
      const double s = std::sin(n * std::numbers::pi * h / (2.0 * length));
      return 4.0 * s * s / (h * h);
    }

    // Stationary state at fixed mu by Newton with the analytic Jacobian.
    [[nodiscard]] auto stationary_state(arma::vec y, double mu) const -> arma::vec {
      dft::algorithms::solvers::Newton newton{.max_iterations = 50, .tolerance = 1e-11};
      auto result = newton.solve(
          std::move(y),
          [&](const arma::vec& v) { return residual(v, mu); },
          [&](const arma::vec& v) { return jacobian(v); }
      );
      return result.solution;
    }

    // Centred interface at mu = 0, by Newton from tanh((x - L/2) / sqrt(2 kappa)).
    [[nodiscard]] auto interface_state() const -> arma::vec {
      return stationary_state(arma::tanh((positions() - 0.5 * length) / std::sqrt(2.0 * kappa)), 0.0);
    }

    [[nodiscard]] auto grand_canonical() const -> Residual {
      return [this](const arma::vec& y, double mu) {
        return residual(y, mu);
      };
    }

    // Canonical form: unknown x = [y; mu], parameter N.
    [[nodiscard]] auto canonical() const -> Residual {
      return [this](const arma::vec& x, double n) {
        arma::vec y = x.head(nodes);
        arma::vec r(nodes + 1);
        r.head(nodes) = residual(y, x(nodes));
        r(nodes) = mass(y) - n;
        return r;
      };
    }
  };

  // Closed forms of the continuum problem.

  namespace exact {

    inline auto fold_density() -> double {
      return 1.0 / std::sqrt(3.0);
    }

    inline auto fold_chemical_potential() -> double {
      return 2.0 / (3.0 * std::sqrt(3.0));
    }

    inline auto surface_tension(double kappa) -> double {
      return 2.0 * std::sqrt(2.0) / 3.0 * std::sqrt(kappa);
    }

    inline auto interface(const arma::vec& x, double x0, double kappa) -> arma::vec {
      return arma::tanh((x - x0) / std::sqrt(2.0 * kappa));
    }

    // Uniform density at which the Neumann mode n pi / L goes soft:
    // kappa q^2 = 1 - 3 rho^2 with q^2 = (n pi / L)^2 (continuum) or the
    // discrete eigenvalue of -D2.
    inline auto bifurcation_density(double kappa_q2) -> double {
      return std::sqrt((1.0 - kappa_q2) / 3.0);
    }

    inline auto bifurcation_count(double length, double kappa, double rho_bar) -> int {
      return static_cast<int>(std::floor(length / std::numbers::pi * std::sqrt((1.0 - 3.0 * rho_bar * rho_bar) / kappa))
      );
    }

    // Uniform density of the metastable state at mu (the root of rho^3 - rho = mu
    // on the side of the stable arc with the higher Omega), for |mu| below the fold.
    inline auto metastable_density(double mu) -> double {
      double rho = mu > 0.0 ? -1.0 : 1.0;
      for (int it = 0; it < 60; ++it)
        rho -= (rho * rho * rho - rho - mu) / (3.0 * rho * rho - 1.0);
      return rho;
    }

  } // namespace exact

} // namespace utils
