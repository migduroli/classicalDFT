#pragma once

#include <algorithm>
#include <armadillo>
#include <cmath>
#include <dftlib>
#include <format>
#include <functional>
#include <iostream>
#include <numbers>
#include <print>
#include <string>
#include <vector>

namespace utils {

  using dft::algorithms::continuation::Continuation;
  using dft::algorithms::continuation::CurvePoint;
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
      for (arma::uword i = 0; i < k; ++i) {
        J(i, i) = 2.0 * c + 3.0 * y(i) * y(i) - 1.0;
        if (i > 0)
          J(i, i - 1) = -c;
        if (i + 1 < k)
          J(i, i + 1) = -c;
      }
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
      arma::vec ev = arma::eig_sym(A);
      return static_cast<int>(arma::accu(ev < -1e-10));
    }

    // Eigenvalues of -D2 with mirrored end nodes: (4 / h^2) sin^2(n pi h / 2L).
    [[nodiscard]] auto laplacian_eigenvalue(int n) const -> double {
      const double h = spacing();
      const double s = std::sin(n * std::numbers::pi * h / (2.0 * length));
      return 4.0 * s * s / (h * h);
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
      return static_cast<int>(
          std::floor(length / std::numbers::pi * std::sqrt((1.0 - 3.0 * rho_bar * rho_bar) / kappa))
      );
    }

  } // namespace exact

  // Points where a test function changes sign between consecutive curve points.

  struct Event {
    CurvePoint point;
    double rho_bar;
    double mu;
    int mode;
    arma::vec eigenvector;
  };

  struct Branch {
    std::string name;
    std::vector<CurvePoint> curve;
    std::vector<double> mu;
    std::vector<double> mass;
    std::vector<double> omega;
    std::vector<double> amplitude;
    std::vector<int> index;
    std::vector<Event> folds;
    std::vector<Event> bifurcations;
  };

  // Record the observables along a mu-parametrised curve.
  inline auto measure(const Problem& p, std::string name, std::vector<CurvePoint> curve) -> Branch {
    Branch b{.name = std::move(name), .curve = std::move(curve)};
    for (const auto& pt : b.curve) {
      b.mu.push_back(pt.lambda);
      b.mass.push_back(p.mass(pt.x));
      b.omega.push_back(p.grand_potential(pt.x, pt.lambda));
      b.amplitude.push_back(p.amplitude(pt.x));
      b.index.push_back(p.index(pt.x));
    }
    return b;
  }

  // Number of sign changes of a vector, ignoring entries near zero.
  inline auto nodal_count(const arma::vec& v) -> int {
    const double floor = 1e-6 * arma::abs(v).max();
    int count = 0;
    double last = 0.0;
    for (double e : v) {
      if (std::abs(e) < floor)
        continue;
      if (last != 0.0 && (e > 0.0) != (last > 0.0))
        ++count;
      last = e;
    }
    return count;
  }

  // Critical eigenvector j of the symmetrised Hessian, mapped back to y-space
  // (the right null vector of dF/dy at a crossing) with unit Euclidean norm.
  inline auto critical_vector(const Problem& p, const arma::vec& y, arma::uword j) -> arma::vec {
    arma::vec ev;
    arma::mat U;
    arma::eig_sym(ev, U, p.symmetric_hessian(y));
    arma::vec v = U.col(j) / arma::sqrt(p.weights());
    return v / arma::norm(v);
  }

  // Folds and bifurcation points along a traced branch, from the library's
  // event location. Folds are the zeros of dlambda/ds; the eigenvalue
  // crossings with a uniform eigenvector (n = 0) are those same folds and are
  // dropped, and the others are bifurcation points labelled by the number of
  // sign changes of the critical eigenvector.
  inline void detect_events(const Problem& p, const Continuation& cont, Branch& b) {
    namespace ac = dft::algorithms::continuation;
    const Residual R = p.grand_canonical();
    for (auto& f : ac::folds(cont, b.curve, R)) {
      const double rho_bar = p.mass(f.x) / p.length;
      const double mu = f.lambda;
      b.folds.push_back(Event{.point = std::move(f), .rho_bar = rho_bar, .mu = mu, .mode = 0, .eigenvector = {}});
    }
    auto spectrum = [&p](const CurvePoint& q) {
      return p.spectrum(q.x);
    };
    for (auto& c : ac::crossings(cont, b.curve, R, spectrum)) {
      arma::vec v = critical_vector(p, c.point.x, c.eigenvalue);
      const int n = nodal_count(v);
      if (n == 0)
        continue;
      const double rho_bar = p.mass(c.point.x) / p.length;
      const double mu = c.point.lambda;
      b.bifurcations.push_back(
          Event{.point = std::move(c.point), .rho_bar = rho_bar, .mu = mu, .mode = n, .eigenvector = std::move(v)}
      );
    }
  }

  // Uniform branch rho^3 - rho = mu, from rho = -rho_max to rho = +rho_max.
  inline auto trace_uniform(const Problem& p, const Continuation& cont, double rho_max) -> Branch {
    const Residual R = p.grand_canonical();
    const double rho0 = -rho_max;
    arma::vec y0(p.nodes, arma::fill::value(rho0));
    arma::vec up(p.nodes, arma::fill::ones);
    auto [dx, dl] = dft::algorithms::continuation::detail::tangent(R, y0, rho0 * rho0 * rho0 - rho0, up, 1.0);
    CurvePoint start{.x = y0, .lambda = rho0 * rho0 * rho0 - rho0, .dx_ds = dx, .dlambda_ds = dl};
    auto curve = cont.trace(start, R, [&](const CurvePoint& q) { return arma::mean(q.x) > rho_max; });
    Branch b = measure(p, "uniform", std::move(curve));
    detect_events(p, cont, b);
    return b;
  }

  // Non-uniform branch leaving the uniform one at a bifurcation point: the
  // first step is the library's branch switch along the critical
  // eigenvector, with dmu/ds = 0 (the pitchfork tangent). The
  // trace stops when the amplitude collapses back towards the uniform branch;
  // both bifurcation points are then added as the end points of the curve.
  inline auto trace_bifurcating(
      const Problem& p,
      const Continuation& cont,
      const Event& bif,
      const Event& end,
      double kick,
      double mu_max,
      std::size_t max_points
  ) -> Branch {
    const Residual R = p.grand_canonical();
    auto first = dft::algorithms::continuation::switch_branch(cont, bif.point, R, bif.eigenvector, kick);
    if (!first)
      return Branch{.name = "n = " + std::to_string(bif.mode)};
    const double a0 = p.amplitude(first->x);
    std::size_t count = 0;
    auto curve = cont.trace(*first, R, [&](const CurvePoint& q) {
      ++count;
      return std::abs(q.lambda) > mu_max || (count > 5 && p.amplitude(q.x) < 0.5 * a0) || count >= max_points;
    });
    curve.insert(curve.begin(), bif.point);
    curve.push_back(end.point);
    Branch b = measure(p, "n = " + std::to_string(bif.mode), std::move(curve));
    // At the bifurcation points the critical eigenvalue vanishes, so the
    // count there is decided by rounding: take the index of the neighbour.
    b.index.front() = b.index[1];
    b.index.back() = b.index[b.index.size() - 2];
    detect_events(p, cont, b);
    return b;
  }

  // Stationary state at fixed mu by Newton with the analytic Jacobian.
  inline auto solve_fixed_mu(const Problem& p, arma::vec y, double mu) -> arma::vec {
    dft::algorithms::solvers::Newton newton{.max_iterations = 50, .tolerance = 1e-11};
    auto res = newton.solve(
        std::move(y),
        [&](const arma::vec& v) { return p.residual(v, mu); },
        [&](const arma::vec& v) { return p.jacobian(v); }
    );
    return res.solution;
  }

  // Branch traced with N as the parameter, unknown x = [y; mu].
  struct CanonicalBranch {
    std::vector<double> mass;
    std::vector<double> mu;
    std::vector<int> index;
    std::vector<arma::vec> profiles;
  };

  inline auto trace_canonical(
      const Problem& p,
      const Continuation& cont,
      const arma::vec& y0,
      double mu0,
      double direction,
      double n_max
  ) -> CanonicalBranch {
    const Residual R = p.canonical();
    const double n0 = p.mass(y0);
    arma::vec x0(p.nodes + 1);
    x0.head(p.nodes) = y0;
    x0(p.nodes) = mu0;
    arma::vec prev(p.nodes + 1, arma::fill::zeros);
    auto [dx, dl] = dft::algorithms::continuation::detail::tangent(R, x0, n0, prev, direction);
    CurvePoint start{.x = x0, .lambda = n0, .dx_ds = dx, .dlambda_ds = dl};
    auto curve = cont.trace(start, R, [&](const CurvePoint& q) {
      return std::abs(q.lambda) > n_max || p.amplitude(q.x.head(p.nodes)) < 0.05;
    });
    CanonicalBranch out;
    for (const auto& q : curve) {
      arma::vec y = q.x.head(p.nodes);
      out.mass.push_back(q.lambda);
      out.mu.push_back(q.x(p.nodes));
      out.index.push_back(p.constrained_index(y));
      out.profiles.push_back(y);
    }
    return out;
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

  // Uniform density of the metastable state at mu (the root of rho^3 - rho = mu
  // on the side of the stable arc with the higher Omega), for |mu| below the fold.
  inline auto metastable_density(double mu) -> double {
    double rho = mu > 0.0 ? -1.0 : 1.0;
    for (int it = 0; it < 60; ++it)
      rho -= (rho * rho * rho - rho - mu) / (3.0 * rho * rho - 1.0);
    return rho;
  }

  // Centred interface at mu = 0 by Newton from the tanh guess.
  inline auto interface_state(const Problem& p) -> arma::vec {
    arma::vec x = p.positions();
    return solve_fixed_mu(p, exact::interface(x, 0.5 * p.length, p.kappa), 0.0);
  }

  inline auto verification(const Problem& p, const Branch& uniform, const Branch& kink) -> std::vector<Row> {
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
      Problem
          q{.length = scale * p.length, .kappa = p.kappa, .nodes = static_cast<arma::uword>(scale * (p.nodes - 1)) + 1};
      arma::vec y = interface_state(q);
      arma::vec ones(q.nodes, arma::fill::ones);
      std::string tag = std::format("L = {:g}", q.length);
      if (scale == 1.0) {
        double err = arma::abs(y - exact::interface(q.positions(), 0.5 * q.length, q.kappa)).max();
        rows.push_back({"interface", "max |y - tanh((x - L/2) / sqrt(2 kappa))|", err, 0.0, 1e-3});
      }
      rows.push_back(
          {"interface",
           "sigma = Omega_1 - Omega_pm, " + tag,
           q.grand_potential(y, 0.0) - q.grand_potential(ones, 0.0),
           exact::surface_tension(q.kappa),
           1e-3}
      );
      arma::vec mid(q.nodes, arma::fill::zeros);
      rows.push_back(
          {"interface",
           "Omega_mid - Omega_pm, " + tag,
           q.grand_potential(mid, 0.0) - q.grand_potential(ones, 0.0),
           0.25 * q.length,
           1e-12}
      );
      if (scale == 1.0) {
        rows.push_back({"interface", "index at mu = 0 (fixed mu)", static_cast<double>(q.index(y)), 1.0, 0.0});
        rows.push_back(
            {"interface", "index at mu = 0 (fixed N)", static_cast<double>(q.constrained_index(y)), 0.0, 0.0}
        );
      }
    }

    // Bifurcation points on the rho < 0 half of the uniform branch.
    int count = 0;
    for (const auto& e : uniform.bifurcations) {
      if (e.rho_bar > 0.0)
        continue;
      ++count;
      double q2 = std::pow(e.mode * std::numbers::pi / p.length, 2);
      std::string tag = std::format("n = {}", e.mode);
      rows.push_back(
          {"bifurcation",
           "rho_bar (discrete D2), " + tag,
           -e.rho_bar,
           exact::bifurcation_density(p.kappa * p.laplacian_eigenvalue(e.mode)),
           1e-9}
      );
      rows.push_back(
          {"bifurcation", "rho_bar (continuum), " + tag, -e.rho_bar, exact::bifurcation_density(p.kappa * q2), 1e-3}
      );
    }
    rows.push_back(
        {"bifurcation",
         "n_max",
         static_cast<double>(count),
         static_cast<double>(exact::bifurcation_count(p.length, p.kappa, 0.0)),
         0.0}
    );
    arma::vec zero(p.nodes, arma::fill::zeros);
    rows.push_back(
        {"bifurcation",
         "index of rho = 0",
         static_cast<double>(p.index(zero)),
         1.0 + exact::bifurcation_count(p.length, p.kappa, 0.0),
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
