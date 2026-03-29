#include "cdft/functional/species.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>

namespace cdft::functional {

  // ── WeightedDensity ───────────────────────────────────────────────────────

  WeightedDensity::WeightedDensity(const std::vector<long>& shape)
      : weight_(shape), field_(shape) {}

  void WeightedDensity::convolve(std::span<const std::complex<double>> rho_fourier) {
    auto wk = weight_.fourier();
    auto out = field_.fourier();
    for (long i = 0; i < weight_.fourier_total(); ++i) {
      out[i] = rho_fourier[i] * wk[i];
    }
    field_.backward();
    field_.scale(1.0 / static_cast<double>(field_.total()));
  }

  void WeightedDensity::back_convolve(std::span<const std::complex<double>> deriv_fourier) {
    // This method is not needed for the basic pipeline; the accumulation
    // happens via the field_ buffer which stores derivatives during back-pass.
    (void)deriv_fourier;
  }

  void WeightedDensity::set_weight_from_real(std::span<const double> real_space_weight) {
    auto real = weight_.real();
    std::copy_n(real_space_weight.data(), real_space_weight.size(), real.data());
    weight_.forward();
    double inv_n = 1.0 / static_cast<double>(weight_.total());
    for (auto& c : weight_.fourier()) {
      c *= inv_n;
    }
  }

  // ── WeightedDensitySet ────────────────────────────────────────────────────

  WeightedDensity& WeightedDensitySet::tensor_component(int i, int j) {
    // Map (i,j) to upper-triangle index: (0,0)=0, (0,1)=1, (0,2)=2, (1,1)=3, (1,2)=4, (2,2)=5
    if (i > j) std::swap(i, j);
    static constexpr int INDEX[3][3] = {{0, 1, 2}, {1, 3, 4}, {2, 4, 5}};
    return tensor[INDEX[i][j]];
  }

  const WeightedDensity& WeightedDensitySet::tensor_component(int i, int j) const {
    if (i > j) std::swap(i, j);
    static constexpr int INDEX[3][3] = {{0, 1, 2}, {1, 3, 4}, {2, 4, 5}};
    return tensor[INDEX[i][j]];
  }

  // ── WeightGenerator ───────────────────────────────────────────────────────

  namespace {
    double volume_hat(double k, double r) {
      if (k < 1e-12) return (4.0 * std::numbers::pi / 3.0) * r * r * r;
      double kr = k * r;
      return 4.0 * std::numbers::pi * (std::sin(kr) - kr * std::cos(kr)) / (k * k * k);
    }

    double surface_hat(double k, double r) {
      if (k < 1e-12) return 4.0 * std::numbers::pi * r * r;
      return 4.0 * std::numbers::pi * r * std::sin(k * r) / k;
    }

    double vector_prefactor(double k, double r) {
      if (k < 1e-12) return 0.0;
      double kr = k * r;
      return -4.0 * std::numbers::pi * (r * std::cos(kr) - std::sin(kr) / k) / (k * k);
    }

    std::pair<double, double> tensor_coefficients(double k, double r) {
      if (k < 1e-12) return {0.0, (4.0 * std::numbers::pi / 3.0) * r * r};

      double kr = k * r;
      double sin_kr = std::sin(kr);
      double cos_kr = std::cos(kr);
      double k2 = k * k, k3 = k2 * k;
      double r2 = r * r;

      double gp_over_k = 4.0 * std::numbers::pi * r * (r * cos_kr / k2 - sin_kr / k3);
      double gpp = 4.0 * std::numbers::pi * r * (-r2 * sin_kr / k - 2.0 * r * cos_kr / k2 + 2.0 * sin_kr / k3);

      return {-(gpp - gp_over_k) / r2, -gp_over_k / r2};
    }
  }  // namespace

  void WeightGenerator::generate(double diameter, double spacing,
                                 const std::vector<long>& shape, WeightedDensitySet& w) {
    double r = 0.5 * diameter;
    long nx = shape[0], ny = shape[1], nz = shape[2];
    long nz_half = nz / 2 + 1;
    long n = nx * ny * nz;
    double inv_n = 1.0 / static_cast<double>(n);

    double dk_x = 2.0 * std::numbers::pi / (static_cast<double>(nx) * spacing);
    double dk_y = 2.0 * std::numbers::pi / (static_cast<double>(ny) * spacing);
    double dk_z = 2.0 * std::numbers::pi / (static_cast<double>(nz) * spacing);

    auto fk_eta = w.eta.weight().fourier();
    auto fk_scalar = w.scalar.weight().fourier();
    std::array<std::span<std::complex<double>>, 3> fk_vec;
    for (int a = 0; a < 3; ++a) fk_vec[a] = w.vector[a].weight().fourier();

    std::array<std::array<std::span<std::complex<double>>, 3>, 3> fk_t;
    for (int i = 0; i < 3; ++i)
      for (int j = i; j < 3; ++j)
        fk_t[i][j] = w.tensor_component(i, j).weight().fourier();

    for (long ix = 0; ix < nx; ++ix) {
      double kx = dk_x * static_cast<double>(ix <= nx / 2 ? ix : ix - nx);
      for (long iy = 0; iy < ny; ++iy) {
        double ky = dk_y * static_cast<double>(iy <= ny / 2 ? iy : iy - ny);
        for (long iz = 0; iz < nz_half; ++iz) {
          double kz = dk_z * static_cast<double>(iz);
          double k2 = kx * kx + ky * ky + kz * kz;
          double k = std::sqrt(k2);
          long idx = iz + nz_half * (iy + ny * ix);

          fk_eta[idx] = volume_hat(k, r) * inv_n;
          fk_scalar[idx] = surface_hat(k, r) * inv_n;

          double fv = vector_prefactor(k, r);
          std::complex<double> imag(0.0, 1.0);
          std::array<double, 3> kv = {kx, ky, kz};
          for (int a = 0; a < 3; ++a) {
            fk_vec[a][idx] = imag * kv[a] * fv * inv_n;
          }

          auto [a_coeff, b_coeff] = tensor_coefficients(k, r);
          double inv_k2 = (k > 1e-12) ? 1.0 / k2 : 0.0;

          for (int i = 0; i < 3; ++i) {
            for (int j = i; j < 3; ++j) {
              double delta_ij = (i == j) ? 1.0 : 0.0;
              fk_t[i][j][idx] = (a_coeff * kv[i] * kv[j] * inv_k2 + b_coeff * delta_ij) * inv_n;
            }
          }
        }
      }
    }
  }

  // ── FMTSpecies ────────────────────────────────────────────────────────────

  FMTSpecies::FMTSpecies(DensityField density, double diameter, double chemical_potential)
      : Species(std::move(density), chemical_potential), diameter_(diameter) {
    auto shape = this->density().shape();

    weights_.for_each([&](WeightedDensity& ch) { ch = WeightedDensity(shape); });
    WeightGenerator::generate(diameter_, this->density().spacing(), shape, weights_);

    double rho_max = 0.9999 * 6.0 / (std::numbers::pi * diameter_ * diameter_ * diameter_);
    density_range_ = rho_max - RHO_MIN;
  }

  void FMTSpecies::convolve_density(bool tensor) {
    density().forward_fft();
    auto rho_fk = density().fft().fourier();

    weights_.eta.convolve(rho_fk);
    weights_.scalar.convolve(rho_fk);
    for (int a = 0; a < 3; ++a) weights_.vector[a].convolve(rho_fk);

    if (tensor) {
      for (int i = 0; i < 3; ++i)
        for (int j = i; j < 3; ++j)
          weights_.tensor_component(i, j).convolve(rho_fk);
    }
  }

  Measures<> FMTSpecies::measures_at(arma::uword pos) const {
    Measures<> m;
    double r = 0.5 * diameter_;
    double inv_4pi_r = 1.0 / (4.0 * std::numbers::pi * r);
    double inv_4pi_r2 = inv_4pi_r / r;

    m.eta = weights_.eta.field().real()[pos];
    m.n2 = weights_.scalar.field().real()[pos];
    m.n1 = m.n2 * inv_4pi_r;
    m.n0 = m.n2 * inv_4pi_r2;

    for (int a = 0; a < 3; ++a) m.v2(a) = weights_.vector[a].field().real()[pos];
    m.v1 = m.v2 * inv_4pi_r;

    for (int i = 0; i < 3; ++i)
      for (int j = i; j < 3; ++j) {
        m.T(i, j) = weights_.tensor_component(i, j).field().real()[pos];
        m.T(j, i) = m.T(i, j);
      }

    m.compute_contractions();
    return m;
  }

  void FMTSpecies::set_derivatives(const Measures<>& dm, arma::uword pos, bool tensor) {
    double r = 0.5 * diameter_;
    double inv_4pi_r = 1.0 / (4.0 * std::numbers::pi * r);
    double inv_4pi_r2 = inv_4pi_r / r;

    weights_.eta.field().real()[pos] = dm.eta;
    weights_.scalar.field().real()[pos] = dm.n0 * inv_4pi_r2 + dm.n1 * inv_4pi_r + dm.n2;

    arma::rowvec3 dv = dm.v1 * inv_4pi_r + dm.v2;
    for (int a = 0; a < 3; ++a) weights_.vector[a].field().real()[pos] = dv(a);

    if (tensor) {
      for (int i = 0; i < 3; ++i) {
        for (int j = i; j < 3; ++j) {
          double val = (i == j) ? dm.T(i, j) : dm.T(i, j) + dm.T(j, i);
          weights_.tensor_component(i, j).field().real()[pos] = val;
        }
      }
    }
  }

  void FMTSpecies::accumulate_forces(bool tensor) {
    auto shape = density().shape();
    numerics::FourierTransform force_fft(shape);
    force_fft.zeros();

    auto accumulate_channel = [&force_fft](WeightedDensity& wd) {
      // FFT the derivative field
      wd.field().forward();
      auto wk = wd.weight().fourier();
      auto dk = wd.field().fourier();
      auto out = force_fft.fourier();
      for (long i = 0; i < wd.weight().fourier_total(); ++i) {
        out[i] += std::conj(wk[i]) * dk[i];
      }
    };

    accumulate_channel(weights_.eta);
    accumulate_channel(weights_.scalar);
    for (int a = 0; a < 3; ++a) accumulate_channel(weights_.vector[a]);
    if (tensor) {
      for (int i = 0; i < 3; ++i)
        for (int j = i; j < 3; ++j)
          accumulate_channel(weights_.tensor_component(i, j));
    }

    force_fft.backward();
    double d_v = density().cell_volume();
    auto real = force_fft.real();
    for (arma::uword i = 0; i < density().size(); ++i) {
      force()(i) += real[i] * d_v;
    }
  }

  double FMTSpecies::compute_free_energy(const FMTModel& model) {
    bool tensor = fmt_needs_tensor(model);
    convolve_density(tensor);

    double f_ex = 0.0;
    double d_v = density().cell_volume();
    arma::uword n = density().size();

    for (arma::uword i = 0; i < n; ++i) {
      auto m = measures_at(i);
      f_ex += fmt_phi(model, m) * d_v;
    }
    return f_ex;
  }

  double FMTSpecies::compute_forces(const FMTModel& model) {
    bool tensor = fmt_needs_tensor(model);
    convolve_density(tensor);

    double f_ex = 0.0;
    double d_v = density().cell_volume();
    arma::uword n = density().size();

    for (arma::uword i = 0; i < n; ++i) {
      auto m = measures_at(i);
      f_ex += fmt_phi(model, m) * d_v;
      auto dm = fmt_d_phi(model, m);
      set_derivatives(dm, i, tensor);
    }

    accumulate_forces(tensor);
    return f_ex;
  }

  void FMTSpecies::set_density_from_alias(const arma::vec& x) {
    arma::vec& rho = density().values();
    for (arma::uword i = 0; i < rho.n_elem; ++i) {
      double y2 = x(i) * x(i);
      rho(i) = RHO_MIN + density_range_ * y2 / (1.0 + y2);
    }
  }

  arma::vec FMTSpecies::density_alias() const {
    const arma::vec& rho = density().values();
    arma::vec x(rho.n_elem);
    for (arma::uword i = 0; i < rho.n_elem; ++i) {
      double delta = std::max(0.0, rho(i) - RHO_MIN);
      double denom = density_range_ - delta;
      if (denom <= 0.0) denom = 1e-30;
      x(i) = std::sqrt(delta / denom);
    }
    return x;
  }

  arma::vec FMTSpecies::alias_force(const arma::vec& x) const {
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

}  // namespace cdft::functional
