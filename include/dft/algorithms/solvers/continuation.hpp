#ifndef DFT_ALGORITHMS_SOLVERS_CONTINUATION_HPP
#define DFT_ALGORITHMS_SOLVERS_CONTINUATION_HPP

#include "dft/algorithms/solvers/jacobian.hpp"
#include "dft/algorithms/solvers/newton.hpp"

#include <armadillo>
#include <cmath>
#include <functional>
#include <optional>
#include <vector>

namespace dft::algorithms::continuation {

  struct CurvePoint {
    arma::vec x;
    double lambda;
    arma::vec dx_ds;
    double dlambda_ds;
  };

  struct ContinuationConfig {
    double initial_step{0.01};
    double max_step{0.1};
    double min_step{1e-5};
    double growth_factor{1.2};
    double shrink_factor{0.5};
    solvers::NewtonConfig newton;
  };

  using Residual = std::function<arma::vec(const arma::vec&, double)>;

  // Compute the tangent vector (dx/ds, dlambda/ds) at a point on the curve
  // using the null space of the extended Jacobian [dR/dx | dR/dlambda].
  // Orients the tangent to agree with the previous direction.
  inline auto tangent(const Residual& R, const arma::vec& x, double lambda, const arma::vec& prev_dx_ds,
                      double prev_dlambda_ds, double eps = 1e-7) -> std::pair<arma::vec, double> {
    const arma::uword n = x.n_elem;

    // dR/dx via central differences
    auto fx = [&](const arma::vec& xi) -> arma::vec { return R(xi, lambda); };
    arma::mat dRdx = solvers::numerical_jacobian(fx, x, eps);

    // dR/dlambda via central differences
    arma::vec dRdl = (R(x, lambda + eps) - R(x, lambda - eps)) / (2.0 * eps);

    // Extended Jacobian: [dR/dx | dR/dlambda] is m x (n+1)
    arma::mat ext(dRdx.n_rows, n + 1);
    ext.head_cols(n) = dRdx;
    ext.col(n) = dRdl;

    // Tangent is in the null space. Use SVD: last right singular vector.
    arma::mat U, V;
    arma::vec s;
    arma::svd(U, s, V, ext);

    arma::vec tau = V.col(V.n_cols - 1);

    // Orient so dot product with previous tangent is non-negative
    arma::vec prev_full(n + 1);
    prev_full.head(n) = prev_dx_ds;
    prev_full(n) = prev_dlambda_ds;
    if (arma::dot(tau, prev_full) < 0.0) {
      tau = -tau;
    }

    return {tau.head(n), tau(n)};
  }

  // One pseudo-arclength continuation step.
  // Returns the next CurvePoint or nullopt if Newton fails to converge.
  [[nodiscard]] inline auto step(const CurvePoint& current, const Residual& R, double ds,
                                 const ContinuationConfig& config) -> std::optional<CurvePoint> {
    const arma::uword n = current.x.n_elem;

    // Predictor: Euler step along tangent
    arma::vec x_pred = current.x + ds * current.dx_ds;
    double lambda_pred = current.lambda + ds * current.dlambda_ds;

    // Pack into augmented vector y = [x; lambda]
    arma::vec y(n + 1);
    y.head(n) = x_pred;
    y(n) = lambda_pred;

    // Augmented residual: physics + arclength constraint
    auto augmented_f = [&](const arma::vec& y_) -> arma::vec {
      arma::vec xi = y_.head(n);
      double lam = y_(n);

      arma::vec phys = R(xi, lam);

      // Arclength constraint: dot(dx, dx_ds) + dlambda * dlambda_ds - ds = 0
      arma::vec dx = xi - current.x;
      double dlam = lam - current.lambda;
      double arc = arma::dot(dx, current.dx_ds) + dlam * current.dlambda_ds - ds;

      arma::vec result(phys.n_elem + 1);
      result.head(phys.n_elem) = phys;
      result(phys.n_elem) = arc;
      return result;
    };

    // Solve augmented system with Newton (auto-Jacobian)
    auto result = solvers::newton(std::move(y), augmented_f, config.newton);

    if (!result.converged) {
      return std::nullopt;
    }

    arma::vec x_new = result.solution.head(n);
    double lambda_new = result.solution(n);

    // Compute tangent at the new point
    auto [dx_ds_new, dlambda_ds_new] = tangent(R, x_new, lambda_new, current.dx_ds, current.dlambda_ds);

    return CurvePoint{
        .x = std::move(x_new),
        .lambda = lambda_new,
        .dx_ds = std::move(dx_ds_new),
        .dlambda_ds = dlambda_ds_new,
    };
  }

  // Trace a curve defined by R(x, lambda) = 0 using pseudo-arclength
  // continuation with adaptive step sizing.
  [[nodiscard]] inline auto trace(CurvePoint start, const Residual& R, const ContinuationConfig& config,
                                  std::function<bool(const CurvePoint&)> stop = {}) -> std::vector<CurvePoint> {
    std::vector<CurvePoint> curve;
    curve.push_back(start);

    double ds = config.initial_step;

    while (ds >= config.min_step) {
      auto next = step(curve.back(), R, ds, config);

      if (!next) {
        // Shrink step and retry
        ds *= config.shrink_factor;
        continue;
      }

      curve.push_back(std::move(*next));

      if (stop && stop(curve.back())) {
        break;
      }

      // Grow step for next iteration
      ds = std::min(ds * config.growth_factor, config.max_step);
    }

    return curve;
  }

}  // namespace dft::algorithms::continuation

#endif  // DFT_ALGORITHMS_SOLVERS_CONTINUATION_HPP
