#pragma once

#include <algorithm>
#include <armadillo>
#include <cmath>
#include <dftlib>
#include <format>
#include <functional>
#include <iostream>
#include <numbers>
#include <optional>
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
    int mode{0};
    int sign{0};
    std::vector<CurvePoint> curve;
    std::vector<double> mu;
    std::vector<double> mass;
    std::vector<double> omega;
    std::vector<double> amplitude;
    std::vector<double> modal;
    std::vector<int> index;
    std::vector<Event> folds;
    std::vector<Event> bifurcations;
  };

  // Record the observables along a mu-parametrised curve.
  inline auto measure(const Problem& p, std::string name, std::vector<CurvePoint> curve, int mode = 0, int sign = 0)
      -> Branch {
    Branch b{.name = std::move(name), .mode = mode, .sign = sign, .curve = std::move(curve)};
    for (const auto& pt : b.curve) {
      b.mu.push_back(pt.lambda);
      b.mass.push_back(p.mass(pt.x));
      b.omega.push_back(p.grand_potential(pt.x, pt.lambda));
      b.amplitude.push_back(p.amplitude(pt.x));
      b.modal.push_back(mode > 0 ? p.modal_amplitude(pt.x, mode) : 0.0);
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
    const Residual R = p.grand_canonical();
    for (auto& f : cont.folds(b.curve, R)) {
      const double rho_bar = p.mass(f.x) / p.length;
      const double mu = f.lambda;
      b.folds.push_back(Event{.point = std::move(f), .rho_bar = rho_bar, .mu = mu, .mode = 0, .eigenvector = {}});
    }
    auto spectrum = [&p](const CurvePoint& q) {
      return p.spectrum(q.x);
    };
    for (auto& c : cont.crossings(b.curve, R, spectrum)) {
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
  // eigenvector, with dmu/ds = 0 (the pitchfork tangent). The eigenvector is
  // oriented so that sign = +1 gives the arm with a_n > 0 and sign = -1 the
  // arm with a_n < 0. The trace stops when the amplitude collapses back
  // towards the uniform branch; both bifurcation points are then added as the
  // end points of the curve.
  inline auto trace_bifurcating(
      const Problem& p,
      const Continuation& cont,
      const Event& bif,
      const Event& end,
      int sign,
      double kick,
      double mu_max,
      std::size_t max_points
  ) -> Branch {
    const Residual R = p.grand_canonical();
    const std::string name = std::format("n = {}, {}", bif.mode, sign > 0 ? "+" : "-");
    arma::vec v = bif.eigenvector;
    if ((p.modal_amplitude(bif.point.x + v, bif.mode) > 0.0) != (sign > 0))
      v = -v;
    auto first = cont.switch_branch(bif.point, R, v, kick);
    if (!first)
      return Branch{.name = name, .mode = bif.mode, .sign = sign};
    const double a0 = p.amplitude(first->x);
    std::size_t count = 0;
    auto curve = cont.trace(*first, R, [&](const CurvePoint& q) {
      ++count;
      return std::abs(q.lambda) > mu_max || (count > 5 && p.amplitude(q.x) < 0.5 * a0) || count >= max_points;
    });
    curve.insert(curve.begin(), bif.point);
    curve.push_back(end.point);
    Branch b = measure(p, name, std::move(curve), bif.mode, sign);
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

  // States on the pitchfork of mode n at prescribed amplitude a_n = a,
  // from Newton on the bordered system F(y, mu) = 0, a_n(y) = a, started from
  // the bifurcation point plus the critical eigenvector scaled to amplitude a.
  // This parametrises the branch by its own amplitude, so small amplitudes
  // are reached without resolving them by arclength steps.
  struct PitchforkSample {
    double amplitude;
    double dmu; // mu - mu_n
  };

  inline auto pitchfork_samples(const Problem& p, const Event& bifurcation, const std::vector<double>& amplitudes)
      -> std::vector<PitchforkSample> {
    const arma::uword k = p.nodes;
    // a_n is linear in y, with gradient (2 / L) h w_i (c_i - c_bar), where
    // c_i = cos(n pi x_i / L) and c_bar is its trapezoid mean.
    const arma::vec w = p.weights();
    const arma::vec c = arma::cos(bifurcation.mode * std::numbers::pi * p.positions() / p.length);
    const double c_bar = p.spacing() * arma::dot(w, c) / p.length;
    const arma::vec grad = 2.0 / p.length * p.spacing() * (w % (c - c_bar));
    const double a_v = arma::dot(grad, bifurcation.eigenvector);
    dft::algorithms::solvers::Newton newton{.max_iterations = 50, .tolerance = 1e-12};
    auto jacobian = [&](const arma::vec& v) {
      arma::mat J(k + 1, k + 1, arma::fill::zeros);
      J.submat(0, 0, k - 1, k - 1) = p.jacobian(v.head(k));
      J.col(k).head(k).fill(-1.0);
      J.row(k).head(k) = grad.t();
      return J;
    };
    // Each sign is swept in increasing |a|, every solve starting from the
    // previous one (the first from the linear guess).
    std::vector<PitchforkSample> out;
    for (double sign : {+1.0, -1.0}) {
      std::vector<double> sweep;
      for (double a : amplitudes) {
        if (a * sign > 0.0)
          sweep.push_back(a);
      }
      std::ranges::sort(sweep, {}, [](double a) { return std::abs(a); });
      arma::vec previous;
      double a_previous = 0.0;
      for (double a : sweep) {
        arma::vec z(k + 1);
        if (previous.is_empty()) {
          z.head(k) = bifurcation.point.x + (a / a_v) * bifurcation.eigenvector;
          z(k) = bifurcation.mu;
        } else {
          z = previous;
          z.head(k) += ((a - a_previous) / a_v) * bifurcation.eigenvector;
        }
        auto residual = [&](const arma::vec& v) {
          arma::vec r(k + 1);
          r.head(k) = p.residual(v.head(k), v(k));
          r(k) = arma::dot(grad, v.head(k)) - a;
          return r;
        };
        auto result = newton.solve(std::move(z), residual, jacobian);
        if (!result.converged)
          break;
        out.push_back({.amplitude = a, .dmu = result.solution(k) - bifurcation.mu});
        previous = result.solution;
        a_previous = a;
      }
    }
    return out;
  }

  // Exponent beta and prefactor C of |a_n| = C |mu - mu_n|^beta, by least
  // squares on log |a_n| against log |mu - mu_n|, over the samples with
  // |a_n| in [a_lo, a_hi].
  struct PowerLaw {
    double exponent;
    double prefactor;
  };

  inline auto fit_power_law(const std::vector<PitchforkSample>& samples, double a_lo, double a_hi) -> PowerLaw {
    std::vector<double> x, y;
    for (const auto& s : samples) {
      if (std::abs(s.amplitude) >= a_lo && std::abs(s.amplitude) <= a_hi) {
        x.push_back(std::log(std::abs(s.dmu)));
        y.push_back(std::log(std::abs(s.amplitude)));
      }
    }
    arma::vec lx(x), ly(y);
    const double mx = arma::mean(lx);
    const double my = arma::mean(ly);
    const double slope = arma::dot(lx - mx, ly - my) / arma::dot(lx - mx, lx - mx);
    return {.exponent = slope, .prefactor = std::exp(my - slope * mx)};
  }

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

  inline auto
  arm_mismatch(const Problem& p, const Continuation& cont, const Branch& plus, const Branch& minus, std::size_t samples)
      -> ArmMismatch {
    const Residual R = p.grand_canonical();
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
        y = solve_fixed_mu(p, 0.5 * (y + arma::reverse(y)), minus.mu[k]);
      }
      const arma::vec target = p.arm_image(y, plus.mode);
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
          foot = cont.locate(plus.curve[a], plus.curve[a + 1], R, g);
          break;
        }
      }
      if (!foot) {
        m.observables = m.profiles = arma::datum::inf;
        continue;
      }
      if (plus.mode % 2 == 0)
        foot->x = solve_fixed_mu(p, 0.5 * (foot->x + arma::reverse(foot->x)), foot->lambda = target_mu);
      m.observables = std::max(
          {m.observables,
           std::abs(foot->lambda - target_mu),
           std::abs(p.mass(foot->x) - p.mass(y)),
           std::abs(p.grand_potential(foot->x, foot->lambda) - p.grand_potential(y, target_mu))}
      );
      m.profiles = std::max(m.profiles, arma::abs(foot->x - target).max());
    }
    return m;
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

  // Everything the example traces, shared by main.cpp and check/main.cpp.
  struct Results {
    Branch uniform;
    std::vector<Branch> arms; // n = 1, +; n = 1, -; n = 2, +; ...
    std::vector<CanonicalBranch> canonical;
    std::vector<Event> pitchfork_points;                 // bifurcation points at rho < 0, n = 1, 2
    std::vector<std::vector<PitchforkSample>> pitchfork; // samples at prescribed a_n, both signs

    [[nodiscard]] auto arm(int n, int sign) const -> const Branch& {
      return *std::ranges::find_if(arms, [&](const Branch& b) { return b.mode == n && b.sign == sign; });
    }
  };

  inline auto run(const Problem& p, const Continuation& cont, int n_max) -> Results {
    Results r;

    dft::console::info("Tracing the uniform branch");
    r.uniform = trace_uniform(p, cont, 1.35);
    std::println(
        std::cout,
        "  {} points, {} folds, {} bifurcation points",
        r.uniform.curve.size(),
        r.uniform.folds.size(),
        r.uniform.bifurcations.size()
    );
    for (const auto& f : r.uniform.folds)
      std::println(std::cout, "  fold          rho = {:+.10f}  mu = {:+.10f}", f.rho_bar, f.mu);
    for (const auto& e : r.uniform.bifurcations)
      std::println(std::cout, "  bifurcation   rho = {:+.10f}  mu = {:+.10f}  n = {}", e.rho_bar, e.mu, e.mode);

    // Both arms of each pitchfork, from the bifurcation point at rho < 0 to
    // its mirror at rho > 0.
    for (int n = 1; n <= n_max; ++n) {
      const Event* start = nullptr;
      const Event* end = nullptr;
      for (const auto& e : r.uniform.bifurcations) {
        if (e.mode == n)
          (e.rho_bar < 0.0 ? start : end) = &e;
      }
      for (int sign : {+1, -1}) {
        auto b = trace_bifurcating(p, cont, *start, *end, sign, 0.3, 1.0, 2000);
        dft::console::info(std::format("Traced the branch {}", b.name));
        auto [lo, hi] = std::ranges::minmax_element(b.index);
        auto [a_lo, a_hi] = std::ranges::minmax_element(b.modal);
        std::println(
            std::cout,
            "  {} points, n_minus in [{}, {}], a_n in [{:+.4f}, {:+.4f}]",
            b.curve.size(),
            *lo,
            *hi,
            *a_lo,
            *a_hi
        );
        r.arms.push_back(std::move(b));
      }
    }

    // Small-amplitude samples on the n = 1 and n = 2 pitchforks.
    dft::console::info("Sampling the n = 1 and n = 2 pitchforks at prescribed a_n");
    std::vector<double> amplitudes;
    for (double e : arma::linspace(-4.0, -1.0, 61)) {
      amplitudes.push_back(std::pow(10.0, e));
      amplitudes.push_back(-std::pow(10.0, e));
    }
    for (const auto& e : r.uniform.bifurcations) {
      if (e.rho_bar < 0.0 && e.mode <= 2) {
        r.pitchfork_points.push_back(e);
        r.pitchfork.push_back(pitchfork_samples(p, e, amplitudes));
      }
    }

    // The n = 1 branch traced again with N as the parameter, from the centred
    // interface at mu = 0 towards both walls.
    dft::console::info("Tracing the n = 1 branch at fixed N");
    arma::vec kink = interface_state(p);
    for (double direction : {+1.0, -1.0})
      r.canonical.push_back(trace_canonical(p, cont, kink, 0.0, direction, 1.5 * p.length));
    for (const auto& c : r.canonical) {
      auto [lo, hi] = std::ranges::minmax_element(c.index);
      std::println(
          std::cout,
          "  {} points, N from {:+.4f} to {:+.4f}, mu at the end {:+.6f}, n_minus at fixed N in [{}, {}]",
          c.mass.size(),
          c.mass.front(),
          c.mass.back(),
          c.mu.back(),
          *lo,
          *hi
      );
    }
    return r;
  }

  inline auto verification(const Problem& p, const Continuation& cont, const Results& res) -> std::vector<Row> {
    const Branch& uniform = res.uniform;
    const Branch& kink = res.arm(1, +1);
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

    // The two arms of each pitchfork are images of each other.
    for (const auto& b : res.arms) {
      if (b.sign < 0)
        continue;
      auto m = arm_mismatch(p, cont, b, res.arm(b.mode, -1), 16);
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
    for (std::size_t j = 0; j < res.pitchfork.size(); ++j) {
      const int n = res.pitchfork_points[j].mode;
      auto narrow = fit_power_law(res.pitchfork[j], 1e-4, 1e-3);
      auto middle = fit_power_law(res.pitchfork[j], 1e-3, 1e-2);
      auto wide = fit_power_law(res.pitchfork[j], 1e-2, 1e-1);
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
