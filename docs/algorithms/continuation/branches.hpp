#pragma once

#include "model.hpp"

#include <algorithm>
#include <armadillo>
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
    std::vector<Event> pitchfork_points;                 // bifurcation points at rho < 0, n = 1, 2
    std::vector<std::vector<PitchforkSample>> pitchfork; // samples at prescribed a_n, both signs
    std::vector<Labelled> fixed_mass_points;             // A to F on the fixed-N figure
    std::vector<Labelled> fixed_mu_points;               // A to F along the n = 1 plus arm

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

    return r;
  }

} // namespace utils
