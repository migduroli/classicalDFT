#ifndef DFT_ALGORITHMS_SOLVERS_CONTINUATION_HPP
#define DFT_ALGORITHMS_SOLVERS_CONTINUATION_HPP

#include "dft/algorithms/solvers/jacobian.hpp"
#include "dft/algorithms/solvers/newton.hpp"

#include <algorithm>
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

  using Residual = std::function<arma::vec(const arma::vec&, double)>;

  // Scalar condition g(x, lambda) = 0 that singles out one point of a curve.
  using Constraint = std::function<double(const arma::vec&, double)>;

  // Test functions evaluated on curve points, for locating events.
  using TestFunction = std::function<double(const CurvePoint&)>;
  using Spectrum = std::function<arma::vec(const CurvePoint&)>;

  // A zero of eigenvalue number `eigenvalue` (in ascending order) on a curve.
  struct Crossing {
    CurvePoint point;
    arma::uword eigenvalue;
  };

  // Arclength of the step from a to its successor b, read from the corrector
  // hyperplane: dot(b.x - a.x, a.dx_ds) + (b.lambda - a.lambda) a.dlambda_ds.
  [[nodiscard]] inline auto arclength(const CurvePoint& a, const CurvePoint& b) -> double {
    return arma::dot(b.x - a.x, a.dx_ds) + (b.lambda - a.lambda) * a.dlambda_ds;
  }

  // Indices k at which g changes sign between curve[k] and curve[k + 1].
  // Sign changes with |g| below floor on either side are skipped, for test
  // functions that are only resolved down to a known noise level.
  [[nodiscard]] inline auto
  sign_changes(const std::vector<CurvePoint>& curve, const TestFunction& g, double floor = 0.0)
      -> std::vector<std::size_t> {
    std::vector<std::size_t> out;
    for (std::size_t k = 0; k + 1 < curve.size(); ++k) {
      const double ga = g(curve[k]);
      const double gb = g(curve[k + 1]);
      if (std::min(std::abs(ga), std::abs(gb)) > floor && (ga > 0.0) != (gb > 0.0)) {
        out.push_back(k);
      }
    }
    return out;
  }

  // Event location shared by Continuation and MatrixFreeContinuation: both
  // provide step(point, R, ds), which is all these need.
  namespace detail {

    template <typename Stepper>
    [[nodiscard]] auto locate(
        const Stepper& stepper,
        const CurvePoint& a,
        const CurvePoint& b,
        const Residual& R,
        const TestFunction& g,
        double tolerance,
        int max_iterations
    ) -> CurvePoint {
      double lo = 0.0;
      double hi = arclength(a, b);
      double g_lo = g(a);
      double g_hi = g(b);
      CurvePoint best = std::abs(g_lo) < std::abs(g_hi) ? a : b;
      int side = 0;
      for (int it = 0; it < max_iterations && std::abs(hi - lo) > tolerance; ++it) {
        const double ds = (lo * g_hi - hi * g_lo) / (g_hi - g_lo);
        auto trial = stepper.step(a, R, ds);
        if (!trial) {
          break;
        }
        const double g_ds = g(*trial);
        best = std::move(*trial);
        if (std::abs(g_ds) < tolerance) {
          break;
        }
        if ((g_ds > 0.0) == (g_hi > 0.0)) {
          hi = ds;
          g_hi = g_ds;
          if (side == -1) {
            g_lo *= 0.5;
          }
          side = -1;
        } else {
          lo = ds;
          g_lo = g_ds;
          if (side == +1) {
            g_hi *= 0.5;
          }
          side = +1;
        }
      }
      return best;
    }

    template <typename Stepper>
    [[nodiscard]] auto
    folds(const Stepper& stepper, const std::vector<CurvePoint>& curve, const Residual& R, double floor)
        -> std::vector<CurvePoint> {
      auto dlambda = [](const CurvePoint& p) {
        return p.dlambda_ds;
      };
      std::vector<CurvePoint> out;
      for (std::size_t k : sign_changes(curve, dlambda, floor)) {
        out.push_back(locate(stepper, curve[k], curve[k + 1], R, dlambda, 1e-13, 80));
      }
      return out;
    }

    template <typename Stepper>
    [[nodiscard]] auto
    crossings(const Stepper& stepper, const std::vector<CurvePoint>& curve, const Residual& R, const Spectrum& spectrum)
        -> std::vector<Crossing> {
      std::vector<Crossing> out;
      if (curve.empty()) {
        return out;
      }
      auto negatives = [](const arma::vec& eigenvalues) {
        return static_cast<arma::uword>(arma::accu(eigenvalues < 0.0));
      };
      arma::uword m_prev = negatives(spectrum(curve.front()));
      for (std::size_t k = 0; k + 1 < curve.size(); ++k) {
        const arma::uword m_next = negatives(spectrum(curve[k + 1]));
        for (arma::uword j = std::min(m_prev, m_next); j < std::max(m_prev, m_next); ++j) {
          auto g = [&spectrum, j](const CurvePoint& p) {
            return spectrum(p)(j);
          };
          out.push_back(Crossing{.point = locate(stepper, curve[k], curve[k + 1], R, g, 1e-13, 80), .eigenvalue = j});
        }
        m_prev = m_next;
      }
      return out;
    }

    template <typename Stepper>
    [[nodiscard]] auto switch_branch(
        const Stepper& stepper,
        const CurvePoint& bifurcation,
        const Residual& R,
        const arma::vec& dx,
        double ds,
        double dlambda
    ) -> std::optional<CurvePoint> {
      const double norm = std::sqrt(arma::dot(dx, dx) + dlambda * dlambda);
      CurvePoint start{
          .x = bifurcation.x,
          .lambda = bifurcation.lambda,
          .dx_ds = dx / norm,
          .dlambda_ds = dlambda / norm,
      };
      return stepper.step(start, R, ds);
    }

  } // namespace detail

  struct Continuation {
    double initial_step{0.01};
    double max_step{0.1};
    double min_step{1e-5};
    double growth_factor{1.2};
    double shrink_factor{0.5};
    solvers::Newton newton;

    // One pseudo-arclength continuation step.
    // Returns the next CurvePoint or nullopt if Newton fails to converge.
    [[nodiscard]] auto step(const CurvePoint& current, const Residual& R, double ds) const -> std::optional<CurvePoint>;

    // Trace a curve defined by R(x, lambda) = 0 using pseudo-arclength
    // continuation with adaptive step sizing.
    [[nodiscard]] auto trace(CurvePoint start, const Residual& R, std::function<bool(const CurvePoint&)> stop = {})
        const -> std::vector<CurvePoint>;

    // The point of the curve R(x, lambda) = 0 where g(x, lambda) = 0, by
    // Newton on the bordered system [R; g] from the guess (x, lambda). The
    // constraint replaces the arclength condition of a step, so the curve can
    // be sampled at prescribed values of any observable (an amplitude, a
    // mass) instead of at arclength steps. The tangent is oriented with
    // dlambda/ds >= 0. Returns nullopt if Newton fails.
    [[nodiscard]] auto constrained_point(const arma::vec& x, double lambda, const Residual& R, const Constraint& g)
        const -> std::optional<CurvePoint>;

    // Root of g between a and its successor b, where g changes sign. Each
    // trial point is a fresh step from a of length ds in (0, arclength(a, b)),
    // and ds is updated by the Illinois variant of regula falsi. Returns the
    // last point computed, on the curve to the Newton tolerance.
    [[nodiscard]] auto locate(
        const CurvePoint& a,
        const CurvePoint& b,
        const Residual& R,
        const TestFunction& g,
        double tolerance = 1e-13,
        int max_iterations = 80
    ) const -> CurvePoint {
      return detail::locate(*this, a, b, R, g, tolerance, max_iterations);
    }

    // Folds (turning points in lambda): the zeros of dlambda/ds along the
    // curve. The default floor skips sign flips of a tangent component that
    // is zero to the accuracy of the finite-difference tangent.
    [[nodiscard]] auto folds(const std::vector<CurvePoint>& curve, const Residual& R, double floor = 1e-8) const
        -> std::vector<CurvePoint> {
      return detail::folds(*this, curve, R, floor);
    }

    // Zero crossings of the eigenvalues of a symmetric operator along the
    // curve; spectrum(p) returns them in ascending order. Where the number of
    // negative eigenvalues changes from m_a to m_b between consecutive points,
    // eigenvalue j changes sign for every j from min(m_a, m_b) to
    // max(m_a, m_b) - 1, and each root is located. A crossing is a fold or a
    // bifurcation point; the caller tells them apart, for example with folds()
    // or from the eigenvector.
    [[nodiscard]] auto crossings(const std::vector<CurvePoint>& curve, const Residual& R, const Spectrum& spectrum)
        const -> std::vector<Crossing> {
      return detail::crossings(*this, curve, R, spectrum);
    }

    // Branch switching at a bifurcation point: one predictor-corrector step
    // of length ds along the direction (dx, dlambda), normalised to unit
    // length. For a pitchfork the bifurcating tangent is (v, 0), with v the
    // critical null vector, so dlambda defaults to zero. The corrector
    // hyperplane excludes the branch through the point, so Newton converges
    // onto the new one. Returns nullopt if Newton fails.
    [[nodiscard]] auto switch_branch(
        const CurvePoint& bifurcation,
        const Residual& R,
        const arma::vec& dx,
        double ds,
        double dlambda = 0.0
    ) const -> std::optional<CurvePoint> {
      return detail::switch_branch(*this, bifurcation, R, dx, ds, dlambda);
    }
  };

  namespace detail {

    // Compute the tangent vector (dx/ds, dlambda/ds) at a point on the curve
    // using the null space of the extended Jacobian [dR/dx | dR/dlambda].
    // Orients the tangent to agree with the previous direction.
    inline auto tangent(
        const Residual& R,
        const arma::vec& x,
        double lambda,
        const arma::vec& prev_dx_ds,
        double prev_dlambda_ds,
        double eps = 1e-7
    ) -> std::pair<arma::vec, double> {
      const arma::uword n = x.n_elem;

      // dR/dx via central differences
      auto fx = [&](const arma::vec& xi) -> arma::vec {
        return R(xi, lambda);
      };
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

    // Matrix-free tangent using the bordering technique:
    // The tangent (dx/ds, dlambda/ds) satisfies [dR/dx | dR/dlambda] * [dx/ds; dlambda/ds] = 0.
    // We solve J * z = -dR/dlambda via GMRES, then set dlambda/ds = 1/(1 + z^T z)^{1/2},
    // dx/ds = z * dlambda/ds. Oriented to agree with previous direction.

    inline auto matrix_free_tangent(
        const Residual& R,
        const arma::vec& x,
        double lambda,
        const arma::vec& prev_dx_ds,
        double prev_dlambda_ds,
        const solvers::GMRES& gmres,
        double eps = 1e-7
    ) -> std::pair<arma::vec, double> {
      const arma::uword n = x.n_elem;

      // dR/dlambda via forward differences.
      arma::vec Rx = R(x, lambda);
      arma::vec dRdl = (R(x, lambda + eps) - Rx) / eps;

      // J(x) * v via forward differences.
      auto Jv = [&](const arma::vec& v) -> arma::vec {
        return (R(x + eps * v, lambda) - Rx) / eps;
      };

      // Solve J * z = -dR/dlambda.
      auto result = gmres.solve(Jv, -dRdl);

      arma::vec z = result.solution;

      // Tangent: (dx/ds, dlambda/ds) = (z, 1) / ||(z, 1)||
      double norm_inv = 1.0 / std::sqrt(arma::dot(z, z) + 1.0);
      arma::vec dx_ds = z * norm_inv;
      double dlambda_ds = norm_inv;

      // Orient to agree with previous tangent.
      double dot = arma::dot(dx_ds, prev_dx_ds) + dlambda_ds * prev_dlambda_ds;
      if (dot < 0.0) {
        dx_ds = -dx_ds;
        dlambda_ds = -dlambda_ds;
      }

      return {std::move(dx_ds), dlambda_ds};
    }

  } // namespace detail

  [[nodiscard]] inline auto Continuation::step(const CurvePoint& current, const Residual& R, double ds) const
      -> std::optional<CurvePoint> {
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
    auto result = newton.solve(std::move(y), augmented_f);

    if (!result.converged) {
      return std::nullopt;
    }

    arma::vec x_new = result.solution.head(n);
    double lambda_new = result.solution(n);

    // Compute tangent at the new point
    auto [dx_ds_new, dlambda_ds_new] = detail::tangent(R, x_new, lambda_new, current.dx_ds, current.dlambda_ds);

    return CurvePoint{
        .x = std::move(x_new),
        .lambda = lambda_new,
        .dx_ds = std::move(dx_ds_new),
        .dlambda_ds = dlambda_ds_new,
    };
  }

  [[nodiscard]] inline auto
  Continuation::trace(CurvePoint start, const Residual& R, std::function<bool(const CurvePoint&)> stop) const
      -> std::vector<CurvePoint> {
    std::vector<CurvePoint> curve;
    curve.push_back(start);

    double ds = initial_step;

    while (ds >= min_step) {
      std::optional<CurvePoint> next;
      try {
        next = step(curve.back(), R, ds);
      } catch (...) {
        break;
      }

      if (!next) {
        ds *= shrink_factor;
        continue;
      }

      curve.push_back(std::move(*next));

      if (stop && stop(curve.back())) {
        break;
      }

      ds = std::min(ds * growth_factor, max_step);
    }

    return curve;
  }

  [[nodiscard]] inline auto
  Continuation::constrained_point(const arma::vec& x, double lambda, const Residual& R, const Constraint& g) const
      -> std::optional<CurvePoint> {
    const arma::uword n = x.n_elem;
    auto bordered = [&](const arma::vec& y) -> arma::vec {
      arma::vec phys = R(y.head(n), y(n));
      arma::vec out(phys.n_elem + 1);
      out.head(phys.n_elem) = phys;
      out(phys.n_elem) = g(y.head(n), y(n));
      return out;
    };
    auto result = newton.solve(arma::join_cols(x, arma::vec{lambda}), bordered);
    if (!result.converged) {
      return std::nullopt;
    }
    arma::vec x_new = result.solution.head(n);
    const double lambda_new = result.solution(n);
    auto [dx_ds, dlambda_ds] = detail::tangent(R, x_new, lambda_new, arma::zeros(n), 1.0);
    return CurvePoint{.x = std::move(x_new), .lambda = lambda_new, .dx_ds = std::move(dx_ds), .dlambda_ds = dlambda_ds};
  }

  // Matrix-free pseudo-arclength continuation for large-scale problems.
  // Uses Newton-GMRES (no dense Jacobian) and bordering for the tangent.

  struct MatrixFreeContinuation {
    double initial_step{0.01};
    double max_step{0.1};
    double min_step{1e-5};
    double growth_factor{1.2};
    double shrink_factor{0.5};
    double jvp_epsilon{1e-7};
    solvers::Newton newton;

    // One step: predictor-corrector with matrix-free Newton on the augmented system.
    [[nodiscard]] auto step(const CurvePoint& current, const Residual& R, double ds) const
        -> std::optional<CurvePoint> {
      const arma::uword n = current.x.n_elem;
      const double eps = jvp_epsilon;

      // Predictor.
      arma::vec x_pred = current.x + ds * current.dx_ds;
      double lambda_pred = current.lambda + ds * current.dlambda_ds;

      arma::vec y(n + 1);
      y.head(n) = x_pred;
      y(n) = lambda_pred;

      // Augmented residual.
      auto augmented_f = [&](const arma::vec& y_) -> arma::vec {
        arma::vec xi = y_.head(n);
        double lam = y_(n);
        arma::vec phys = R(xi, lam);
        arma::vec dx = xi - current.x;
        double dlam = lam - current.lambda;
        double arc = arma::dot(dx, current.dx_ds) + dlam * current.dlambda_ds - ds;
        arma::vec result(phys.n_elem + 1);
        result.head(phys.n_elem) = phys;
        result(phys.n_elem) = arc;
        return result;
      };

      // JVP factory for the augmented system.
      auto jvp_factory = [&](const arma::vec& y_) {
        const arma::vec fy = augmented_f(y_);
        return [&augmented_f, y_, fy, eps](const arma::vec& v) -> arma::vec {
          return (augmented_f(y_ + eps * v) - fy) / eps;
        };
      };

      auto result = newton.solve_matrix_free(std::move(y), augmented_f, jvp_factory);

      if (!result.converged) {
        return std::nullopt;
      }

      arma::vec x_new = result.solution.head(n);
      double lambda_new = result.solution(n);

      // Tangent via bordering: solve J * dx_ds = -dR/dlambda, then normalise.
      auto [dx_ds_new, dlambda_ds_new] =
          detail::matrix_free_tangent(R, x_new, lambda_new, current.dx_ds, current.dlambda_ds, newton.gmres, eps);

      return CurvePoint{
          .x = std::move(x_new),
          .lambda = lambda_new,
          .dx_ds = std::move(dx_ds_new),
          .dlambda_ds = dlambda_ds_new,
      };
    }

    // Trace a curve using matrix-free continuation with adaptive step sizing.
    [[nodiscard]] auto trace(CurvePoint start, const Residual& R, std::function<bool(const CurvePoint&)> stop = {})
        const -> std::vector<CurvePoint> {
      std::vector<CurvePoint> curve;
      curve.push_back(start);
      double ds = initial_step;

      while (ds >= min_step) {
        std::optional<CurvePoint> next;
        try {
          next = step(curve.back(), R, ds);
        } catch (...) {
          break;
        }

        if (!next) {
          ds *= shrink_factor;
          continue;
        }

        curve.push_back(std::move(*next));
        if (stop && stop(curve.back()))
          break;
        ds = std::min(ds * growth_factor, max_step);
      }

      return curve;
    }

    // Event location and branch switching, as for Continuation.

    // Root of g between a and its successor b, where g changes sign. Each
    // trial point is a fresh step from a of length ds in (0, arclength(a, b)),
    // and ds is updated by the Illinois variant of regula falsi. Returns the
    // last point computed, on the curve to the Newton tolerance.
    [[nodiscard]] auto locate(
        const CurvePoint& a,
        const CurvePoint& b,
        const Residual& R,
        const TestFunction& g,
        double tolerance = 1e-13,
        int max_iterations = 80
    ) const -> CurvePoint {
      return detail::locate(*this, a, b, R, g, tolerance, max_iterations);
    }

    // Folds (turning points in lambda): the zeros of dlambda/ds along the
    // curve. The default floor skips sign flips of a tangent component that
    // is zero to the accuracy of the finite-difference tangent.
    [[nodiscard]] auto folds(const std::vector<CurvePoint>& curve, const Residual& R, double floor = 1e-8) const
        -> std::vector<CurvePoint> {
      return detail::folds(*this, curve, R, floor);
    }

    // Zero crossings of the eigenvalues of a symmetric operator along the
    // curve; spectrum(p) returns them in ascending order. Where the number of
    // negative eigenvalues changes from m_a to m_b between consecutive points,
    // eigenvalue j changes sign for every j from min(m_a, m_b) to
    // max(m_a, m_b) - 1, and each root is located. A crossing is a fold or a
    // bifurcation point; the caller tells them apart, for example with folds()
    // or from the eigenvector.
    [[nodiscard]] auto crossings(const std::vector<CurvePoint>& curve, const Residual& R, const Spectrum& spectrum)
        const -> std::vector<Crossing> {
      return detail::crossings(*this, curve, R, spectrum);
    }

    // Branch switching at a bifurcation point: one predictor-corrector step
    // of length ds along the direction (dx, dlambda), normalised to unit
    // length. For a pitchfork the bifurcating tangent is (v, 0), with v the
    // critical null vector, so dlambda defaults to zero. The corrector
    // hyperplane excludes the branch through the point, so Newton converges
    // onto the new one. Returns nullopt if Newton fails.
    [[nodiscard]] auto switch_branch(
        const CurvePoint& bifurcation,
        const Residual& R,
        const arma::vec& dx,
        double ds,
        double dlambda = 0.0
    ) const -> std::optional<CurvePoint> {
      return detail::switch_branch(*this, bifurcation, R, dx, ds, dlambda);
    }
  };

} // namespace dft::algorithms::continuation

#endif // DFT_ALGORITHMS_SOLVERS_CONTINUATION_HPP
