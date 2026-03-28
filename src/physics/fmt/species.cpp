#include "classicaldft_bits/physics/fmt/species.h"

#include <cmath>
#include <numbers>

namespace dft_core::physics::fmt {

  Species::Species(density::Density density, double diameter, double chemical_potential)
      : species::Species(std::move(density), chemical_potential), diameter_(diameter) {
    auto& d = this->density();
    auto shape = d.shape();

    // Construct all convolution channels
    weights_.for_each([&](ConvolutionField& ch) { ch = ConvolutionField(shape); });

    // Populate the weight functions in Fourier space
    Weights::generate(diameter_, d.dx(), shape, weights_);

    // Bounded alias: rho = rho_min + density_range * y^2 / (1 + y^2)
    // Maximum density corresponds to eta = 0.9999 (safety margin below close packing).
    double rho_max = 0.9999 * 6.0 / (std::numbers::pi * diameter_ * diameter_ * diameter_);
    density_range_ = rho_max - RHO_MIN;
  }

  // ── Forward convolution ───────────────────────────────────────────────────

  void Species::convolve_density(bool tensor) {
    density().forward_fft();
    auto rho_fk = density().fft().fourier();

    weights_.eta.convolve(rho_fk);
    weights_.scalar.convolve(rho_fk);
    for (int a = 0; a < 3; ++a) {
      weights_.vector[a].convolve(rho_fk);
    }

    if (tensor) {
      for (int i = 0; i < 3; ++i)
        for (int j = i; j < 3; ++j)
          weights_.tensor(i, j).convolve(rho_fk);
    }
  }

  // ── Measure extraction ────────────────────────────────────────────────────

  Measures Species::measures_at(arma::uword pos) const {
    Measures m;
    double r = 0.5 * diameter_;
    double inv_4pi_r = 1.0 / (4.0 * std::numbers::pi * r);
    double inv_4pi_r2 = inv_4pi_r / r;

    m.eta = weights_.eta.field()(pos);
    m.n2 = weights_.scalar.field()(pos);
    m.n1 = m.n2 * inv_4pi_r;
    m.n0 = m.n2 * inv_4pi_r2;

    for (int a = 0; a < 3; ++a) {
      m.v2(a) = weights_.vector[a].field()(pos);
    }
    m.v1 = m.v2 * inv_4pi_r;

    for (int i = 0; i < 3; ++i)
      for (int j = i; j < 3; ++j) {
        m.T(i, j) = weights_.tensor(i, j).field()(pos);
        m.T(j, i) = m.T(i, j);
      }

    m.compute_derived();
    return m;
  }

  // ── Derivative collapse ───────────────────────────────────────────────────

  void Species::set_derivatives(const Measures& dm, arma::uword pos, bool tensor) {
    double r = 0.5 * diameter_;
    double inv_4pi_r = 1.0 / (4.0 * std::numbers::pi * r);
    double inv_4pi_r2 = inv_4pi_r / r;

    weights_.eta.derivative()(pos) = dm.eta;

    // Chain rule: d/d(scalar) = dPhi/dn0 / (4piR^2) + dPhi/dn1 / (4piR) + dPhi/dn2
    weights_.scalar.derivative()(pos) = dm.n0 * inv_4pi_r2 + dm.n1 * inv_4pi_r + dm.n2;

    // Chain rule: d/d(vector_a) = dPhi/dv1_a / (4piR) + dPhi/dv2_a
    arma::rowvec3 dv = dm.v1 * inv_4pi_r + dm.v2;
    for (int a = 0; a < 3; ++a) {
      weights_.vector[a].derivative()(pos) = dv(a);
    }

    if (tensor) {
      for (int i = 0; i < 3; ++i) {
        for (int j = i; j < 3; ++j) {
          double val = (i == j) ? dm.T(i, j) : dm.T(i, j) + dm.T(j, i);
          weights_.tensor(i, j).derivative()(pos) = val;
        }
      }
    }
  }

  // ── Back-convolution ──────────────────────────────────────────────────────

  void Species::accumulate_forces(bool tensor) {
    auto shape = density().shape();
    numerics::fourier::FourierTransform force_fft(shape);
    force_fft.zeros();

    auto out = force_fft.fourier();
    weights_.eta.accumulate(out);
    weights_.scalar.accumulate(out);
    for (int a = 0; a < 3; ++a) {
      weights_.vector[a].accumulate(out);
    }
    if (tensor) {
      for (int i = 0; i < 3; ++i)
        for (int j = i; j < 3; ++j)
          weights_.tensor(i, j).accumulate(out);
    }

    force_fft.backward();
    double d_v = density().cell_volume();
    auto real = force_fft.real();
    for (arma::uword i = 0; i < density().size(); ++i) {
      add_to_force(i, real[i] * d_v);
    }
  }

  // ── Single-responsibility pipeline methods ────────────────────────────────

  double Species::compute_free_energy(const Functional& functional) {
    bool tensor = functional.needs_tensor();
    convolve_density(tensor);

    double f_ex = 0.0;
    double d_v = density().cell_volume();
    arma::uword n = density().size();

    for (arma::uword i = 0; i < n; ++i) {
      auto m = measures_at(i);
      f_ex += functional.phi(m) * d_v;
    }

    return f_ex;
  }

  double Species::compute_forces(const Functional& functional) {
    bool tensor = functional.needs_tensor();
    convolve_density(tensor);

    double f_ex = 0.0;
    double d_v = density().cell_volume();
    arma::uword n = density().size();

    for (arma::uword i = 0; i < n; ++i) {
      auto m = measures_at(i);
      f_ex += functional.phi(m) * d_v;
      auto dm = functional.d_phi(m);
      set_derivatives(dm, i, tensor);
    }

    accumulate_forces(tensor);
    return f_ex;
  }

  // ── Bounded alias ─────────────────────────────────────────────────────────

  void Species::set_density_from_alias(const arma::vec& x) {
    arma::vec& rho = density().values();
    for (arma::uword i = 0; i < rho.n_elem; ++i) {
      double y2 = x(i) * x(i);
      rho(i) = RHO_MIN + density_range_ * y2 / (1.0 + y2);
    }
  }

  arma::vec Species::density_alias() const {
    const arma::vec& rho = density().values();
    arma::vec x(rho.n_elem);
    for (arma::uword i = 0; i < rho.n_elem; ++i) {
      double delta = std::max(0.0, rho(i) - RHO_MIN);
      double denom = density_range_ - delta;
      if (denom <= 0.0)
        denom = 1e-30;
      x(i) = std::sqrt(delta / denom);
    }
    return x;
  }

  arma::vec Species::alias_force(const arma::vec& x) const {
    const arma::vec& f = force();
    arma::vec result(x.n_elem);
    for (arma::uword i = 0; i < x.n_elem; ++i) {
      double y = x(i);
      double y2 = y * y;
      double denom = (1.0 + y2) * (1.0 + y2);
      double drho_dy = density_range_ * 2.0 * y / denom;
      result(i) = f(i) * drho_dy;
    }
    return result;
  }

}  // namespace dft_core::physics::fmt
