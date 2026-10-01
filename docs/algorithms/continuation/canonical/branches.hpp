#pragma once

#include "model.hpp"

#include <algorithm>
#include <armadillo>
#include <array>
#include <cmath>
#include <dftlib>
#include <format>
#include <iostream>
#include <print>
#include <string>
#include <vector>

namespace utils {

  using dft::algorithms::continuation::Continuation;
  using dft::algorithms::continuation::CurvePoint;

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
  inline auto
  measure(const Problem& problem, std::string name, std::vector<CurvePoint> curve, int mode = 0, int sign = 0)
      -> Branch {
    Branch b{.name = std::move(name), .mode = mode, .sign = sign, .curve = std::move(curve)};
    for (const auto& pt : b.curve) {
      b.mu.push_back(pt.lambda);
      b.mass.push_back(problem.mass(pt.x));
      b.omega.push_back(problem.grand_potential(pt.x, pt.lambda));
      b.amplitude.push_back(problem.amplitude(pt.x));
      b.modal.push_back(mode > 0 ? problem.modal_amplitude(pt.x, mode) : 0.0);
      b.index.push_back(problem.index(pt.x));
    }
    return b;
  }

  // Number of sign changes of a vector, ignoring entries near zero.
  inline auto nodal_count(const arma::vec& v) -> int {
    const arma::vec resolved = v.elem(arma::find(arma::abs(v) >= 1e-6 * arma::abs(v).max()));
    if (resolved.n_elem < 2)
      return 0;
    return static_cast<int>(arma::accu(arma::diff(arma::sign(resolved)) != 0.0));
  }

  // Critical eigenvector j of the symmetrised Hessian, mapped back to y-space
  // (the right null vector of dF/dy at a crossing) with unit Euclidean norm.
  inline auto critical_vector(const Problem& problem, const arma::vec& y, arma::uword j) -> arma::vec {
    arma::vec eigenvalues;
    arma::mat U;
    arma::eig_sym(eigenvalues, U, problem.symmetric_hessian(y));
    arma::vec v = U.col(j) / arma::sqrt(problem.weights());
    return v / arma::norm(v);
  }

  // Folds and bifurcation points along a traced branch, from the library's
  // event location. Folds are the zeros of dlambda/ds; the eigenvalue
  // crossings with a uniform eigenvector (n = 0) are those same folds and are
  // dropped, and the others are bifurcation points labelled by the number of
  // sign changes of the critical eigenvector.
  struct Events {
    std::vector<Event> folds;
    std::vector<Event> bifurcations;
  };

  inline auto
  detect_events(const Problem& problem, const Continuation& continuation, const std::vector<CurvePoint>& curve)
      -> Events {
    Events events;
    const Residual R = problem.grand_canonical();
    for (auto& f : continuation.folds(curve, R)) {
      const double rho_bar = problem.mass(f.x) / problem.length;
      const double mu = f.lambda;
      events.folds.push_back(Event{.point = std::move(f), .rho_bar = rho_bar, .mu = mu, .mode = 0, .eigenvector = {}});
    }
    auto spectrum = [&problem](const CurvePoint& q) {
      return problem.spectrum(q.x);
    };
    for (auto& c : continuation.crossings(curve, R, spectrum)) {
      arma::vec v = critical_vector(problem, c.point.x, c.eigenvalue);
      const int n = nodal_count(v);
      if (n == 0)
        continue;
      const double rho_bar = problem.mass(c.point.x) / problem.length;
      const double mu = c.point.lambda;
      events.bifurcations.push_back(
          Event{.point = std::move(c.point), .rho_bar = rho_bar, .mu = mu, .mode = n, .eigenvector = std::move(v)}
      );
    }
    return events;
  }

  // Uniform branch rho^3 - rho = mu, from rho = -rho_max to rho = +rho_max.
  inline auto trace_uniform(const Problem& problem, const Continuation& continuation, double rho_max) -> Branch {
    const Residual R = problem.grand_canonical();
    const double rho0 = -rho_max;
    arma::vec y0(problem.nodes, arma::fill::value(rho0));
    arma::vec up(problem.nodes, arma::fill::ones);
    auto [dx, dl] = dft::algorithms::continuation::detail::tangent(R, y0, rho0 * rho0 * rho0 - rho0, up, 1.0);
    CurvePoint start{.x = y0, .lambda = rho0 * rho0 * rho0 - rho0, .dx_ds = dx, .dlambda_ds = dl};
    auto curve = continuation.trace(start, R, [&](const CurvePoint& q) { return arma::mean(q.x) > rho_max; });
    Branch b = measure(problem, "uniform", std::move(curve));
    auto events = detect_events(problem, continuation, b.curve);
    b.folds = std::move(events.folds);
    b.bifurcations = std::move(events.bifurcations);
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
      const Problem& problem,
      const Continuation& continuation,
      const Event& bifurcation,
      const Event& end,
      int sign,
      double kick,
      double mu_max,
      std::size_t max_points
  ) -> Branch {
    const Residual R = problem.grand_canonical();
    const std::string name = std::format("n = {}, {}", bifurcation.mode, sign > 0 ? "+" : "-");
    arma::vec v = bifurcation.eigenvector;
    if ((problem.modal_amplitude(bifurcation.point.x + v, bifurcation.mode) > 0.0) != (sign > 0))
      v = -v;
    auto first = continuation.switch_branch(bifurcation.point, R, v, kick);
    if (!first)
      return Branch{.name = name, .mode = bifurcation.mode, .sign = sign};
    const double a0 = problem.amplitude(first->x);
    std::size_t count = 0;
    auto curve = continuation.trace(*first, R, [&](const CurvePoint& q) {
      ++count;
      return std::abs(q.lambda) > mu_max || (count > 5 && problem.amplitude(q.x) < 0.5 * a0) || count >= max_points;
    });
    curve.insert(curve.begin(), bifurcation.point);
    curve.push_back(end.point);
    Branch b = measure(problem, name, std::move(curve), bifurcation.mode, sign);
    // At the bifurcation points the critical eigenvalue vanishes, so the
    // count there is decided by rounding: take the index of the neighbour.
    b.index.front() = b.index[1];
    b.index.back() = b.index[b.index.size() - 2];
    auto events = detect_events(problem, continuation, b.curve);
    b.folds = std::move(events.folds);
    b.bifurcations = std::move(events.bifurcations);
    return b;
  }

  // States on the pitchfork of mode n at prescribed amplitude a_n = a, from
  // the library's constrained_point with the condition a_n(y) = a.
  // This parametrises the branch by its own amplitude, so small amplitudes
  // are reached without resolving them by arclength steps.
  struct PitchforkSample {
    double amplitude;
    double dmu; // mu - mu_n
  };

  inline auto pitchfork_samples(
      const Problem& problem,
      const Continuation& continuation,
      const Event& bifurcation,
      const std::vector<double>& amplitudes
  ) -> std::vector<PitchforkSample> {
    const Residual R = problem.grand_canonical();
    const int n = bifurcation.mode;
    // a_n is linear and vanishes on the uniform state: a_n(y_b + t v) = t a_n(v).
    const double a_v = problem.modal_amplitude(bifurcation.point.x + bifurcation.eigenvector, n);
    // At |a_n| = 1e-4 the shift mu - mu_n is about 1e-8, so the samples need a
    // Newton tolerance well below the one used for tracing.
    Continuation sampler = continuation;
    sampler.newton.tolerance = 1e-12;
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
      arma::vec y = bifurcation.point.x;
      double mu = bifurcation.mu;
      double a_previous = 0.0;
      for (double a : sweep) {
        y += ((a - a_previous) / a_v) * bifurcation.eigenvector;
        auto point = sampler.constrained_point(y, mu, R, [&](const arma::vec& v, double) {
          return problem.modal_amplitude(v, n) - a;
        });
        if (!point)
          break;
        out.push_back({.amplitude = a, .dmu = point->lambda - bifurcation.mu});
        y = point->x;
        mu = point->lambda;
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

  // Two-term normal form mu - mu_n = c2 a^2 + c4 a^4, by linear least squares
  // on the samples with |a_n| <= a_max.
  struct NormalForm {
    double c2;
    double c4;
  };

  inline auto fit_normal_form(const std::vector<PitchforkSample>& samples, double a_max) -> NormalForm {
    std::vector<double> a2, dmu;
    for (const auto& s : samples) {
      if (std::abs(s.amplitude) <= a_max) {
        a2.push_back(s.amplitude * s.amplitude);
        dmu.push_back(s.dmu);
      }
    }
    const arma::vec x(a2);
    const arma::mat A = arma::join_rows(x, arma::square(x));
    const arma::vec c = arma::solve(A, arma::vec(dmu));
    return {.c2 = c(0), .c4 = c(1)};
  }

  // Branch traced with N as the parameter, unknown x = [y; mu].
  struct CanonicalBranch {
    std::vector<double> mass;
    std::vector<double> mu;
    std::vector<int> index;
    std::vector<arma::vec> profiles;
  };

  inline auto trace_canonical(
      const Problem& problem,
      const Continuation& continuation,
      const arma::vec& y0,
      double mu0,
      double direction,
      double n_max
  ) -> CanonicalBranch {
    const Residual R = problem.canonical();
    const double n0 = problem.mass(y0);
    arma::vec x0(problem.nodes + 1);
    x0.head(problem.nodes) = y0;
    x0(problem.nodes) = mu0;
    arma::vec prev(problem.nodes + 1, arma::fill::zeros);
    auto [dx, dl] = dft::algorithms::continuation::detail::tangent(R, x0, n0, prev, direction);
    CurvePoint start{.x = x0, .lambda = n0, .dx_ds = dx, .dlambda_ds = dl};
    auto curve = continuation.trace(start, R, [&](const CurvePoint& q) {
      return std::abs(q.lambda) > n_max || problem.amplitude(q.x.head(problem.nodes)) < 0.05;
    });
    CanonicalBranch out;
    for (const auto& q : curve) {
      arma::vec y = q.x.head(problem.nodes);
      out.mass.push_back(q.lambda);
      out.mu.push_back(q.x(problem.nodes));
      out.index.push_back(problem.constrained_index(y));
      out.profiles.push_back(y);
    }
    return out;
  }

  // Fold of the n = 1 branch in N: the largest N with a minority layer at a
  // wall. Starts from the fixed-N state with a layer of width 3.5 sqrt(2 kappa)
  // against x = 0, traces in increasing N with N as the parameter, and locates
  // the zero of dN/ds with the library's folds().
  struct FiniteSizeFold {
    double length;
    double mass;
    double mu;
  };

  inline auto finite_size_fold(const Problem& problem, const Continuation& continuation) -> FiniteSizeFold {
    const double width = std::sqrt(2.0 * problem.kappa);
    const double layer = 3.5 * width;
    const arma::vec guess = arma::tanh((problem.positions() - layer) / width);
    const double n0 = problem.mass(guess);
    auto start = continuation.constrained_point(guess, 0.0, problem.grand_canonical(), [&](const arma::vec& y, double) {
      return problem.mass(y) - n0;
    });
    const Residual R = problem.canonical();
    arma::vec x0 = arma::join_cols(start->x, arma::vec{start->lambda});
    auto [dx, dl] = dft::algorithms::continuation::detail::tangent(R, x0, n0, arma::zeros(problem.nodes + 1), 1.0);
    CurvePoint first{.x = x0, .lambda = n0, .dx_ds = dx, .dlambda_ds = dl};
    int after_fold = 0;
    auto curve = continuation.trace(first, R, [&](const CurvePoint& q) {
      if (q.dlambda_ds < 0.0)
        ++after_fold;
      return after_fold >= 2;
    });
    auto folds = continuation.folds(curve, R);
    return {.length = problem.length, .mass = folds.front().lambda, .mu = folds.front().x(problem.nodes)};
  }

  // Stationary states at mu = 0 with lambda = L / sqrt(kappa) as the
  // parameter, so kappa = (L / lambda)^2. Continuing in lambda at fixed L is
  // continuing in kappa; lambda is the axis on which the Neumann mode n goes
  // soft at n pi (continuum) or at L sqrt(d_n) (discrete).
  struct LambdaBranch {
    int mode;
    int sign;
    std::vector<double> lambda;
    std::vector<double> modal;
    std::vector<int> index;
    std::vector<double> resolution; // smallest |eigenvalue|: the index is decided only where this is resolved
    std::vector<arma::vec> profiles;
  };

  struct NestedPitchforks {
    std::vector<double> lambda;                       // along the uniform state rho = 0
    std::vector<int> index;                           // its index
    std::vector<std::pair<int, double>> bifurcations; // (n, lambda_n)
    std::vector<LambdaBranch> arms;
  };

  inline auto at_lambda(const Problem& problem, double lambda) -> Problem {
    return {.length = problem.length, .kappa = std::pow(problem.length / lambda, 2), .nodes = problem.nodes};
  }

  inline auto
  nested_pitchforks(const Problem& problem, const Continuation& continuation, double lambda_min, double lambda_max)
      -> NestedPitchforks {
    const Residual R = [&problem](const arma::vec& y, double lambda) {
      return at_lambda(problem, lambda).residual(y, 0.0);
    };
    auto spectrum = [&problem](const CurvePoint& q) {
      return at_lambda(problem, q.lambda).spectrum(q.x);
    };
    NestedPitchforks out;
    const CurvePoint start{
        .x = arma::zeros(problem.nodes),
        .lambda = lambda_min,
        .dx_ds = arma::zeros(problem.nodes),
        .dlambda_ds = 1.0,
    };
    auto uniform = continuation.trace(start, R, [&](const CurvePoint& q) { return q.lambda > lambda_max; });
    for (const auto& q : uniform) {
      out.lambda.push_back(q.lambda);
      out.index.push_back(at_lambda(problem, q.lambda).index(q.x));
    }
    for (const auto& crossing : continuation.crossings(uniform, R, spectrum)) {
      const double lambda_n = crossing.point.lambda;
      arma::vec v = critical_vector(at_lambda(problem, lambda_n), crossing.point.x, crossing.eigenvalue);
      const int n = nodal_count(v);
      out.bifurcations.emplace_back(n, lambda_n);
      if (problem.modal_amplitude(v, n) < 0.0)
        v = -v;
      for (int sign : {+1, -1}) {
        LambdaBranch arm{.mode = n, .sign = sign};
        auto first = continuation.switch_branch(crossing.point, R, sign * v, 0.3);
        if (!first) {
          out.arms.push_back(std::move(arm));
          continue;
        }
        auto curve = continuation.trace(*first, R, [&](const CurvePoint& q) { return q.lambda > lambda_max; });
        for (const auto& q : curve) {
          arm.lambda.push_back(q.lambda);
          arm.modal.push_back(problem.modal_amplitude(q.x, n));
          const arma::vec eigenvalues = at_lambda(problem, q.lambda).spectrum(q.x);
          arm.index.push_back(static_cast<int>(arma::accu(eigenvalues < 0.0)));
          arm.resolution.push_back(arma::abs(eigenvalues).min());
          arm.profiles.push_back(q.x);
        }
        out.arms.push_back(std::move(arm));
      }
    }
    return out;
  }

  // Landau pitchfork of a uniform order parameter at mu = 0:
  // f0 = rho^4 / 4 + a rho^2 / 2, so rho^3 + a rho = 0, continued in a from
  // positive to negative. rho = 0 loses stability at a = 0 and splits into
  // rho = +-sqrt(-a); with a proportional to T - T_c this is the pitchfork at
  // the top of the (rho, T) coexistence dome.
  struct LandauPitchfork {
    std::vector<double> a_uniform;
    double a_critical;
    std::array<std::vector<double>, 2> a_arm;
    std::array<std::vector<double>, 2> rho_arm;
  };

  inline auto landau_pitchfork(const Continuation& continuation, double a_max) -> LandauPitchfork {
    const Residual R = [](const arma::vec& rho, double a) {
      return arma::vec{rho(0) * rho(0) * rho(0) + a * rho(0)};
    };
    auto spectrum = [](const CurvePoint& q) {
      return arma::vec{3.0 * q.x(0) * q.x(0) + q.lambda};
    };
    const CurvePoint start{.x = arma::vec{0.0}, .lambda = a_max, .dx_ds = arma::vec{0.0}, .dlambda_ds = -1.0};
    auto uniform = continuation.trace(start, R, [&](const CurvePoint& q) { return q.lambda < -a_max; });
    LandauPitchfork out;
    for (const auto& q : uniform)
      out.a_uniform.push_back(q.lambda);
    const auto crossing = continuation.crossings(uniform, R, spectrum).front();
    out.a_critical = crossing.point.lambda;
    for (std::size_t k = 0; k < 2; ++k) {
      auto first = continuation.switch_branch(crossing.point, R, arma::vec{k == 0 ? 1.0 : -1.0}, 0.05);
      auto curve = continuation.trace(*first, R, [&](const CurvePoint& q) { return q.lambda < -a_max; });
      for (const auto& q : curve) {
        out.a_arm[k].push_back(q.lambda);
        out.rho_arm[k].push_back(q.x(0));
      }
    }
    return out;
  }

  // A state marked with a letter on a figure, with its profile.
  struct Labelled {
    std::string letter;
    arma::vec y;
    double mu;
    double mass;
    double value; // Helmholtz F at fixed N, or Omega - Omega_meta at fixed mu
    int index;    // negative eigenvalues in the ensemble of the figure
  };

  // Everything the example traces, shared by main.cpp and check/main.cpp.
  struct Results {
    Branch uniform;
    std::vector<Branch> arms; // n = 1, +; n = 1, -; n = 2, +; ...
    std::vector<CanonicalBranch> canonical;
    std::vector<Event> pitchfork_points;                 // bifurcation points at rho < 0, n = 1, 2, 3
    std::vector<std::vector<PitchforkSample>> pitchfork; // samples at prescribed a_n, both signs
    std::vector<Labelled> fixed_mass_points;             // A to F on the fixed-N figure
    std::vector<Labelled> fixed_mu_points;               // A to F along the n = 1 plus arm
    std::vector<FiniteSizeFold> finite_size_folds;       // at L, 2L and 4L, same spacing
    NestedPitchforks nested;                             // at mu = 0, continued in L / sqrt(kappa)
    double nested_lambda_max{0.0};
    LandauPitchfork landau; // uniform, continued in a

    [[nodiscard]] auto arm(int n, int sign) const -> const Branch& {
      return *std::ranges::find_if(arms, [&](const Branch& b) { return b.mode == n && b.sign == sign; });
    }
  };

  inline auto run(const Problem& problem, const Continuation& continuation, int n_max) -> Results {
    Results r;

    dft::console::info("Tracing the uniform branch");
    r.uniform = trace_uniform(problem, continuation, 1.35);
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
        auto b = trace_bifurcating(problem, continuation, *start, *end, sign, 0.3, 1.0, 2000);
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

    // Small-amplitude samples on the n = 1, 2 and 3 pitchforks.
    dft::console::info("Sampling the n = 1, 2 and 3 pitchforks at prescribed a_n");
    std::vector<double> amplitudes;
    for (double e : arma::linspace(-4.0, -1.0, 61)) {
      amplitudes.push_back(std::pow(10.0, e));
      amplitudes.push_back(-std::pow(10.0, e));
    }
    for (const auto& e : r.uniform.bifurcations) {
      if (e.rho_bar < 0.0 && e.mode <= 3) {
        r.pitchfork_points.push_back(e);
        r.pitchfork.push_back(pitchfork_samples(problem, continuation, e, amplitudes));
      }
    }

    // The n = 1 branch traced again with N as the parameter, from the centred
    // interface at mu = 0 towards both walls.
    dft::console::info("Tracing the n = 1 branch at fixed N");
    const arma::vec kink = problem.interface_state();
    for (double direction : {+1.0, -1.0})
      r.canonical.push_back(trace_canonical(problem, continuation, kink, 0.0, direction, 1.5 * problem.length));
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
    // Lettered states for the fixed-N figure: uniform states at N = 18 (A),
    // 14 (B) and 5 (F), the fixed-N saddle (C) and the phase-separated state
    // (D) at the same N = 14, and the centred interface (E). C and D are solved
    // at N = 14 exactly, from the nearest traced point with the wanted index.
    const Residual R = problem.grand_canonical();
    auto uniform_state = [&](const std::string& letter, double n) {
      const double rho = n / problem.length;
      const arma::vec y(problem.nodes, arma::fill::value(rho));
      const double mu = rho * rho * rho - rho;
      return Labelled{letter, y, mu, n, problem.grand_potential(y, mu) + mu * n, problem.constrained_index(y)};
    };
    auto fixed_mass = [&](const std::string& letter, double n, int wanted_index) {
      const auto& traced = r.canonical.front();
      std::size_t pick = 0;
      double distance = arma::datum::inf;
      for (std::size_t k = 0; k < traced.mass.size(); ++k) {
        if (traced.index[k] == wanted_index && std::abs(traced.mass[k] - n) < distance) {
          distance = std::abs(traced.mass[k] - n);
          pick = k;
        }
      }
      auto point =
          continuation.constrained_point(traced.profiles[pick], traced.mu[pick], R, [&](const arma::vec& y, double) {
            return problem.mass(y) - n;
          });
      return Labelled{
          letter,
          point->x,
          point->lambda,
          n,
          problem.grand_potential(point->x, point->lambda) + point->lambda * n,
          problem.constrained_index(point->x)
      };
    };
    r.fixed_mass_points = {
        uniform_state("A", 18.0),
        uniform_state("B", 14.0),
        fixed_mass("C", 14.0, 1),
        fixed_mass("D", 14.0, 0),
        Labelled{
            "E",
            kink,
            0.0,
            problem.mass(kink),
            problem.grand_potential(kink, 0.0),
            problem.constrained_index(kink)
        },
        uniform_state("F", 5.0),
    };

    // Lettered states along the n = 1 plus arm at fixed mu: near the
    // bifurcation, at mu = 0.2 and 0.05, the centred interface at mu = 0, and
    // the mirror half at mu = -0.05 and -0.2.
    const auto& arm = r.arm(1, +1);
    auto fixed_mu = [&](const std::string& letter, double mu) {
      arma::vec y;
      if (mu == 0.0) {
        y = arm.modal[arm.modal.size() / 2] > 0.0 ? arma::vec(-kink) : kink;
      } else {
        std::size_t k = 1;
        while (k + 2 < arm.mu.size() && (arm.mu[k] - mu) * (arm.mu[k + 1] - mu) > 0.0)
          ++k;
        const double t = (mu - arm.mu[k]) / (arm.mu[k + 1] - arm.mu[k]);
        y = problem.stationary_state((1.0 - t) * arm.curve[k].x + t * arm.curve[k + 1].x, mu);
      }
      const double rho = exact::metastable_density(mu);
      const double omega_meta = problem.length * (0.25 * std::pow(rho * rho - 1.0, 2) - mu * rho);
      return Labelled{letter, y, mu, problem.mass(y), problem.grand_potential(y, mu) - omega_meta, problem.index(y)};
    };
    r.fixed_mu_points = {
        fixed_mu("A", 0.38),
        fixed_mu("B", 0.2),
        fixed_mu("C", 0.05),
        fixed_mu("D", 0.0),
        fixed_mu("E", -0.05),
        fixed_mu("F", -0.2),
    };

    // Fold of the n = 1 branch in N at L, 2L and 4L with the same spacing.
    dft::console::info("Locating the finite-size fold at L, 2L and 4L");
    for (double scale : {1.0, 2.0, 4.0}) {
      const Problem box{
          .length = scale * problem.length,
          .kappa = problem.kappa,
          .nodes = static_cast<arma::uword>(scale * static_cast<double>(problem.nodes - 1)) + 1,
      };
      const auto fold = finite_size_fold(box, continuation);
      std::println(
          std::cout,
          "  L = {:g}: N_fold = {:.6f}, L - N_fold = {:.6f}, L mu_fold = {:.6f}, l / sqrt(2 kappa) = {:.4f}",
          fold.length,
          fold.mass,
          fold.length - fold.mass,
          fold.length * fold.mu,
          0.5 * (fold.length - fold.mass) / std::sqrt(2.0 * box.kappa)
      );
      r.finite_size_folds.push_back(fold);
    }

    // Nested pitchforks at mu = 0, continued in lambda = L / sqrt(kappa) up to
    // 21, between 6 pi and 7 pi.
    dft::console::info("Continuing the uniform state at mu = 0 in L / sqrt(kappa)");
    r.nested_lambda_max = 21.0;
    r.nested = nested_pitchforks(problem, continuation, 2.5, r.nested_lambda_max);
    for (const auto& [n, lambda_n] : r.nested.bifurcations)
      std::println(
          std::cout,
          "  n = {}: lambda_n = {:.10f}, kappa_n = {:.10f}",
          n,
          lambda_n,
          std::pow(problem.length / lambda_n, 2)
      );

    dft::console::info("Landau pitchfork in a at mu = 0");
    r.landau = landau_pitchfork(continuation, 1.0);
    std::println(std::cout, "  a_c = {:.3e}", r.landau.a_critical);

    return r;
  }

} // namespace utils
