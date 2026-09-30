#pragma once

#include "branches.hpp"

#include <algorithm>
#include <armadillo>
#include <cmath>
#include <format>
#include <iostream>
#include <optional>
#include <print>
#include <string>
#include <vector>

namespace utils {

  // Largest mismatch between the two arms of a pitchfork. Each sampled point
  // of the minus arm is mapped by arm_image onto the plus arm, and its foot
  // point on the traced plus arm (the zero of d/ds of the squared distance)
  // is located with the library's locate(); when the arms were traced with
  // the same steps (odd n, where the symmetry permutes the nodes) the image
  // is a traced point itself. The arms are images of each other
  // when the image lies on the plus arm with the same mu, N and Omega; the
  // symmetry is an isometry of the trapezoid norm, so matching points also
  // sit at matching arclength.
  //
  // For even n the shift maps solutions to solutions only among states
  // symmetric about the cell boundaries, and it does not carry the Hessian
  // across: the minus arm (a slab away from the walls) has an extra
  // eigenvalue near zero, the translation of the slab, along which the trace
  // drifts by up to the Newton tolerance divided by that eigenvalue (on the
  // mu < 0 half the roles swap and the plus arm drifts). For even n both the
  // sampled state and its foot point are therefore symmetrised and re-solved
  // at the same mu; inside the symmetric subspace the translation mode is not
  // excited. The drift of the traced minus arm is reported separately.
  struct ArmMismatch {
    double observables;
    double profiles;
    double drift;
  };

  inline auto arm_mismatch(
      const Problem& problem,
      const Continuation& continuation,
      const Branch& plus,
      const Branch& minus,
      std::size_t samples
  ) -> ArmMismatch {
    const Residual R = problem.grand_canonical();
    ArmMismatch m{.observables = 0.0, .profiles = 0.0, .drift = 0.0};
    // Samples from the second to the second-to-last traced point, so that the
    // foot point is bracketed away from the appended end points.
    const std::size_t first = 2;
    const std::size_t last = minus.curve.size() - 3;
    for (std::size_t s = 0; s < samples; ++s) {
      const std::size_t k = first + s * (last - first) / (samples - 1);
      arma::vec y = minus.curve[k].x;
      if (plus.mode % 2 == 0) {
        m.drift = std::max(m.drift, arma::abs(y - arma::reverse(y)).max());
        y = problem.stationary_state(0.5 * (y + arma::reverse(y)), minus.mu[k]);
      }
      const arma::vec target = problem.arm_image(y, plus.mode);
      const double target_mu = minus.mu[k];
      auto g = [&](const CurvePoint& q) {
        return arma::dot(q.x - target, q.dx_ds) + (q.lambda - target_mu) * q.dlambda_ds;
      };
      std::size_t best = 1;
      double best_distance = arma::datum::inf;
      for (std::size_t j = 1; j + 1 < plus.curve.size(); ++j) {
        double d = arma::norm(plus.curve[j].x - target) + std::abs(plus.mu[j] - target_mu);
        if (d < best_distance) {
          best_distance = d;
          best = j;
        }
      }
      // Bracket the foot point in one of the two intervals next to the
      // nearest traced point; a sample without a bracket fails the check.
      std::optional<CurvePoint> foot;
      if (best_distance < 1e-6)
        foot = plus.curve[best];
      for (std::size_t a : {best - 1, best}) {
        if (foot)
          break;
        if (a < 1 || a + 2 > plus.curve.size() - 1)
          continue;
        if ((g(plus.curve[a]) > 0.0) != (g(plus.curve[a + 1]) > 0.0)) {
          foot = continuation.locate(plus.curve[a], plus.curve[a + 1], R, g);
          break;
        }
      }
      if (!foot) {
        m.observables = m.profiles = arma::datum::inf;
        continue;
      }
      if (plus.mode % 2 == 0) {
        foot->x = problem.stationary_state(0.5 * (foot->x + arma::reverse(foot->x)), target_mu);
        foot->lambda = target_mu;
      }
      m.observables = std::max(
          {m.observables,
           std::abs(foot->lambda - target_mu),
           std::abs(problem.mass(foot->x) - problem.mass(y)),
           std::abs(problem.grand_potential(foot->x, foot->lambda) - problem.grand_potential(y, target_mu))}
      );
      m.profiles = std::max(m.profiles, arma::abs(foot->x - target).max());
    }
    return m;
  }

  // Residual of dOmega/dmu = -N along a branch, integrated segment by segment
  // with the trapezoid rule: max_k |dOmega_k + Nbar_k dmu_k|, relative to
  // max_k |dOmega_k|.
  inline auto omega_identity_error(const Branch& b) -> double {
    double worst = 0.0;
    double scale = 0.0;
    for (std::size_t k = 0; k + 1 < b.mu.size(); ++k) {
      double d_omega = b.omega[k + 1] - b.omega[k];
      double d_mu = b.mu[k + 1] - b.mu[k];
      double n_bar = 0.5 * (b.mass[k] + b.mass[k + 1]);
      worst = std::max(worst, std::abs(d_omega + n_bar * d_mu));
      scale = std::max(scale, std::abs(d_omega));
    }
    return worst / scale;
  }

  // Verification rows: measured value next to the exact one.

  struct Row {
    std::string group;
    std::string quantity;
    double measured;
    double exact;
    double tolerance;

    [[nodiscard]] auto error() const -> double { return std::abs(measured - exact); }

    [[nodiscard]] auto passed() const -> bool { return error() <= tolerance; }
  };

  inline auto verification(const Problem& problem, const Continuation& continuation, const Results& results)
      -> std::vector<Row> {
    const Branch& uniform = results.uniform;
    const Branch& kink = results.arm(1, +1);
    std::vector<Row> rows;
    const double rho_f = exact::fold_density();
    const double mu_f = exact::fold_chemical_potential();

    // Uniform branch.
    double worst = 0.0;
    for (const auto& q : uniform.curve) {
      double rho = arma::mean(q.x);
      worst = std::max(worst, std::abs(rho * rho * rho - rho - q.lambda));
      worst = std::max(worst, arma::abs(q.x - rho).max());
    }
    rows.push_back({"uniform", "max |rho^3 - rho - mu|, max |y - rho|", worst, 0.0, 1e-9});
    for (const auto& f : uniform.folds) {
      double sign = f.rho_bar < 0.0 ? -1.0 : 1.0;
      rows.push_back({"uniform", "fold rho", f.rho_bar, sign * rho_f, 1e-8});
      rows.push_back({"uniform", "fold mu", f.mu, -sign * mu_f, 1e-8});
    }
    rows.push_back({"uniform", "dOmega/dmu + N (relative)", omega_identity_error(uniform), 0.0, 1e-3});
    rows.push_back({"n = 1", "dOmega/dmu + N (relative)", omega_identity_error(kink), 0.0, 1e-3});

    // Interface at mu = 0, and the gap to the stable states at two box sizes.
    for (double scale : {1.0, 2.0}) {
      Problem scaled{
          .length = scale * problem.length,
          .kappa = problem.kappa,
          .nodes = static_cast<arma::uword>(scale * (problem.nodes - 1)) + 1
      };
      arma::vec y = scaled.interface_state();
      arma::vec ones(scaled.nodes, arma::fill::ones);
      std::string tag = std::format("L = {:g}", scaled.length);
      if (scale == 1.0) {
        double err = arma::abs(y - exact::interface(scaled.positions(), 0.5 * scaled.length, scaled.kappa)).max();
        rows.push_back({"interface", "max |y - tanh((x - L/2) / sqrt(2 kappa))|", err, 0.0, 1e-3});
      }
      rows.push_back(
          {"interface",
           "sigma = Omega_1 - Omega_pm, " + tag,
           scaled.grand_potential(y, 0.0) - scaled.grand_potential(ones, 0.0),
           exact::surface_tension(scaled.kappa),
           1e-3}
      );
      arma::vec mid(scaled.nodes, arma::fill::zeros);
      rows.push_back(
          {"interface",
           "Omega_mid - Omega_pm, " + tag,
           scaled.grand_potential(mid, 0.0) - scaled.grand_potential(ones, 0.0),
           0.25 * scaled.length,
           1e-12}
      );
      if (scale == 1.0) {
        rows.push_back({"interface", "index at mu = 0 (fixed mu)", static_cast<double>(scaled.index(y)), 1.0, 0.0});
        rows.push_back(
            {"interface", "index at mu = 0 (fixed N)", static_cast<double>(scaled.constrained_index(y)), 0.0, 0.0}
        );
      }
    }

    // Index at fixed N of the lettered states: minima A, B, D and E, the
    // saddle C, and the uniform state F with one unstable mass-conserving
    // mode per soft Neumann mode, floor((L / pi) sqrt((1 - 3 rho^2) / kappa)).
    for (const auto& point : results.fixed_mass_points) {
      double expected = point.letter == "C" ? 1.0 : 0.0;
      if (point.letter == "F")
        expected = exact::bifurcation_count(problem.length, problem.kappa, point.mass / problem.length);
      rows.push_back(
          {"fixed N",
           std::format(
               "index at fixed N of {} (N = {:.0f})",
               point.letter,
               std::abs(point.mass) < 1e-9 ? 0.0 : point.mass
           ),
           static_cast<double>(point.index),
           expected,
           0.0}
      );
    }

    // Finite-size fold of the n = 1 branch in N. A minority layer of width l
    // at a wall has |mu| = A exp(-2 q l), q = sqrt(2 / kappa), and N = L (1 -
    // |mu| / 2) - 2 l, so dN/d|mu| = 0 at |mu| = 2 / (q L): L mu_fold tends to
    // -sqrt(2 kappa), and L - N_fold = (1 + ln(A q L / 2)) / q grows by ln 2 / q
    // per doubling of L. Both carry an O(1/L) correction, removed by
    // Richardson extrapolation from 2L and 4L; the tolerances allow for the
    // O(1/L^2) remainder.
    {
      const auto& folds = results.finite_size_folds;
      const double q = std::sqrt(2.0 / problem.kappa);
      const double l_mu_2 = folds[1].length * folds[1].mu;
      const double l_mu_4 = folds[2].length * folds[2].mu;
      const double step_1 = (folds[1].length - folds[1].mass) - (folds[0].length - folds[0].mass);
      const double step_2 = (folds[2].length - folds[2].mass) - (folds[1].length - folds[1].mass);
      rows.push_back(
          {"fold in N",
           "L mu_fold, Richardson from 2L and 4L",
           2.0 * l_mu_4 - l_mu_2,
           -std::sqrt(2.0 * problem.kappa),
           2e-3}
      );
      rows.push_back(
          {"fold in N", "growth of L - N_fold per doubling, Richardson", 2.0 * step_2 - step_1, std::log(2.0) / q, 3e-3}
      );
    }

    // The two arms of each pitchfork are images of each other.
    for (const auto& b : results.arms) {
      if (b.sign < 0)
        continue;
      auto m = arm_mismatch(problem, continuation, b, results.arm(b.mode, -1), 16);
      std::string tag = std::format("n = {}", b.mode);
      rows.push_back({"arms", "max |d mu|, |d N|, |d Omega| between arms, " + tag, m.observables, 0.0, 1e-6});
      rows.push_back({"arms", "max |S y_- - y_+|, " + tag, m.profiles, 0.0, 1e-6});
      if (b.mode % 2 == 0)
        std::println(
            std::cout,
            "  n = {}: drift of the traced minus arm off the symmetric subspace {:.2e}",
            b.mode,
            m.drift
        );
    }

    // Pitchfork exponent: |a_n| ~ C |mu - mu_n|^beta with beta = 1/2. The
    // next term of the normal form, mu - mu_n = c2 a^2 + c4 a^4, shifts the
    // fitted exponent by O(a^2), so the fit converges to 1/2 as the window
    // shrinks. The fit on 1e-4 <= |a_n| <= 1e-3 must lie closer to 1/2 than
    // the fit on the next decade, which sets its tolerance.
    for (std::size_t j = 0; j < results.pitchfork.size(); ++j) {
      const int n = results.pitchfork_points[j].mode;
      auto narrow = fit_power_law(results.pitchfork[j], 1e-4, 1e-3);
      auto middle = fit_power_law(results.pitchfork[j], 1e-3, 1e-2);
      auto wide = fit_power_law(results.pitchfork[j], 1e-2, 1e-1);
      std::println(
          std::cout,
          "  n = {}: pitchfork exponent {:.8f}, {:.8f}, {:.8f} on |a_n| in [1e-4, 1e-3], [1e-3, 1e-2], [1e-2, 1e-1]",
          n,
          narrow.exponent,
          middle.exponent,
          wide.exponent
      );
      rows.push_back(
          {"pitchfork",
           std::format("exponent beta, |a_n| in [1e-4, 1e-3], n = {}", n),
           narrow.exponent,
           0.5,
           std::abs(middle.exponent - 0.5)}
      );
    }

    // Bifurcation points on the rho < 0 half of the uniform branch.
    int count = 0;
    for (const auto& e : uniform.bifurcations) {
      if (e.rho_bar > 0.0)
        continue;
      ++count;
      double q2 = std::pow(e.mode * std::numbers::pi / problem.length, 2);
      std::string tag = std::format("n = {}", e.mode);
      rows.push_back(
          {"bifurcation",
           "rho_bar (discrete D2), " + tag,
           -e.rho_bar,
           exact::bifurcation_density(problem.kappa * problem.laplacian_eigenvalue(e.mode)),
           1e-9}
      );
      rows.push_back(
          {"bifurcation",
           "rho_bar (continuum), " + tag,
           -e.rho_bar,
           exact::bifurcation_density(problem.kappa * q2),
           1e-3}
      );
    }
    rows.push_back(
        {"bifurcation",
         "n_max",
         static_cast<double>(count),
         static_cast<double>(exact::bifurcation_count(problem.length, problem.kappa, 0.0)),
         0.0}
    );
    arma::vec zero(problem.nodes, arma::fill::zeros);
    rows.push_back(
        {"bifurcation",
         "index of rho = 0",
         static_cast<double>(problem.index(zero)),
         1.0 + exact::bifurcation_count(problem.length, problem.kappa, 0.0),
         0.0}
    );
    return rows;
  }

  inline void print_rows(const std::vector<Row>& rows) {
    std::println(
        std::cout,
        "  {:<12s} {:<44s} {:>16s} {:>16s} {:>10s}  {}",
        "group",
        "quantity",
        "measured",
        "exact",
        "error",
        ""
    );
    std::println(std::cout, "  {}", std::string(106, '-'));
    for (const auto& r : rows) {
      std::println(
          std::cout,
          "  {:<12s} {:<44s} {:>16.10f} {:>16.10f} {:>10.2e}  {}",
          r.group,
          r.quantity,
          r.measured,
          r.exact,
          r.error(),
          r.passed() ? "PASS" : "FAIL"
      );
    }
  }

} // namespace utils
