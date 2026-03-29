#include "cdft/functional/density.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace cdft::functional {

  DensityField::DensityField(double spacing, const arma::rowvec3& box_size)
      : spacing_(spacing), box_size_(box_size) {
    if (spacing <= 0.0) {
      throw std::invalid_argument("DensityField: spacing must be positive");
    }

    shape_.resize(3);
    for (int d = 0; d < 3; ++d) {
      long n = static_cast<long>(std::round(box_size(d) / spacing));
      if (n <= 0 || std::abs(box_size(d) - static_cast<double>(n) * spacing) > 1e-10 * spacing) {
        throw std::invalid_argument(
            "DensityField: box dimension " + std::to_string(box_size(d)) +
            " not commensurate with spacing = " + std::to_string(spacing));
      }
      shape_[d] = n;
    }

    long ntot = shape_[0] * shape_[1] * shape_[2];
    rho_ = arma::vec(static_cast<arma::uword>(ntot), arma::fill::zeros);
    external_field_ = arma::vec(static_cast<arma::uword>(ntot), arma::fill::zeros);
    fft_ = numerics::FourierTransform(shape_);
  }

  void DensityField::set(const arma::vec& rho) {
    if (rho.n_elem != rho_.n_elem) {
      throw std::invalid_argument(
          "DensityField::set: size mismatch (got " + std::to_string(rho.n_elem) +
          ", expected " + std::to_string(rho_.n_elem) + ")");
    }
    rho_ = rho;
  }

  void DensityField::set(arma::uword index, double value) {
    if (index >= rho_.n_elem) {
      throw std::out_of_range(
          "DensityField::set: index " + std::to_string(index) +
          " out of range [0, " + std::to_string(rho_.n_elem) + ")");
    }
    rho_(index) = value;
  }

  void DensityField::forward_fft() {
    auto real_buf = fft_.real();
    std::copy_n(rho_.memptr(), rho_.n_elem, real_buf.data());
    fft_.forward();
  }

  double DensityField::number_of_atoms() const {
    double sum = 0.0;
    double compensation = 0.0;
    for (arma::uword i = 0; i < rho_.n_elem; ++i) {
      double t = sum + rho_(i);
      if (std::abs(sum) >= std::abs(rho_(i))) {
        compensation += (sum - t) + rho_(i);
      } else {
        compensation += (rho_(i) - t) + sum;
      }
      sum = t;
    }
    return (sum + compensation) * cell_volume();
  }

  double DensityField::external_field_energy() const {
    return arma::dot(rho_, external_field_) * cell_volume();
  }

  arma::rowvec3 DensityField::center_of_mass() const {
    long nx = shape_[0], ny = shape_[1], nz = shape_[2];
    arma::rowvec3 com = {0.0, 0.0, 0.0};
    double total = 0.0;

    for (long ix = 0; ix < nx; ++ix) {
      for (long iy = 0; iy < ny; ++iy) {
        for (long iz = 0; iz < nz; ++iz) {
          double rho_i = rho_(flat_index(ix, iy, iz));
          com(0) += rho_i * (spacing_ * static_cast<double>(ix));
          com(1) += rho_i * (spacing_ * static_cast<double>(iy));
          com(2) += rho_i * (spacing_ * static_cast<double>(iz));
          total += rho_i;
        }
      }
    }
    if (total > 0.0) com /= total;
    return com;
  }

  void DensityField::save(const std::string& filename) const {
    rho_.save(filename, arma::raw_binary);
  }

  void DensityField::load(const std::string& filename) {
    arma::vec loaded;
    if (!loaded.load(filename, arma::raw_binary)) {
      throw std::runtime_error("DensityField::load: failed to read " + filename);
    }
    if (loaded.n_elem != rho_.n_elem) {
      throw std::invalid_argument(
          "DensityField::load: size mismatch (file has " + std::to_string(loaded.n_elem) +
          ", expected " + std::to_string(rho_.n_elem) + ")");
    }
    rho_ = loaded;
  }

  // ── Species ───────────────────────────────────────────────────────────────

  Species::Species(DensityField density, double chemical_potential)
      : density_(std::move(density)),
        mu_(chemical_potential),
        force_(arma::vec(density_.size(), arma::fill::zeros)) {}

  double Species::ideal_free_energy() const {
    double d_v = density_.cell_volume();
    double f = 0.0;
    const arma::vec& rho = density_.values();
    for (arma::uword i = 0; i < rho.n_elem; ++i) {
      if (rho(i) > RHO_MIN) {
        f += rho(i) * (std::log(rho(i)) - 1.0) * d_v;
      }
    }
    return f;
  }

  void Species::set_density_from_alias(const arma::vec& x) {
    arma::vec rho = RHO_MIN + arma::square(x);
    density_.set(rho);
  }

  arma::vec Species::density_alias() const {
    return arma::sqrt(arma::clamp(density_.values() - RHO_MIN, 0.0, arma::datum::inf));
  }

  arma::vec Species::alias_force(const arma::vec& x) const {
    return 2.0 * x % force_;
  }

}  // namespace cdft::functional
