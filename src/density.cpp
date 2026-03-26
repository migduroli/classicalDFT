#include "dft/density.h"

#include "dft/math/arithmetic.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace dft::density {

  Density::Density(double dx, const arma::rowvec3& box_size) : dx_(dx), box_size_(box_size) {
    if (dx <= 0.0) {
      throw std::invalid_argument("Density: dx must be positive");
    }

    shape_.resize(3);
    for (int d = 0; d < 3; ++d) {
      long n = static_cast<long>(std::round(box_size(d) / dx));
      if (n <= 0 || std::abs(box_size(d) - static_cast<double>(n) * dx) > 1e-10 * dx) {
        throw std::invalid_argument(
            "Density: box dimension " + std::to_string(box_size(d)) +
            " not commensurate with dx = " + std::to_string(dx)
        );
      }
      shape_[d] = n;
    }

    long ntot = shape_[0] * shape_[1] * shape_[2];
    rho_ = arma::vec(static_cast<arma::uword>(ntot), arma::fill::zeros);
    external_field_ = arma::vec(static_cast<arma::uword>(ntot), arma::fill::zeros);
    fft_ = math::fourier::FourierTransform(shape_);
  }

  // ── Density set ─────────────────────────────────────────────────────────────

  void Density::set(const arma::vec& rho) {
    if (rho.n_elem != rho_.n_elem) {
      throw std::invalid_argument(
          "Density::set: size mismatch (got " + std::to_string(rho.n_elem) + ", expected " +
          std::to_string(rho_.n_elem) + ")"
      );
    }
    rho_ = rho;
  }

  void Density::set(arma::uword index, double value) {
    if (index >= rho_.n_elem) {
      throw std::out_of_range(
          "Density::set: index " + std::to_string(index) + " out of range [0, " + std::to_string(rho_.n_elem) + ")"
      );
    }
    rho_(index) = value;
  }

  // ── FFT ─────────────────────────────────────────────────────────────────────

  void Density::forward_fft() {
    auto real_buf = fft_.real();
    std::copy_n(rho_.memptr(), rho_.n_elem, real_buf.data());
    fft_.forward();
  }

  // ── Derived quantities ──────────────────────────────────────────────────────

  double Density::number_of_atoms() const {
    math::arithmetic::CompensatedSum sum;
    for (arma::uword i = 0; i < rho_.n_elem; ++i) {
      sum += rho_(i);
    }
    return sum.sum() * cell_volume();
  }

  double Density::external_field_energy() const {
    double d_v = cell_volume();
    return arma::dot(rho_, external_field_) * d_v;
  }

  arma::rowvec3 Density::center_of_mass() const {
    long nx = shape_[0];
    long ny = shape_[1];
    long nz = shape_[2];

    arma::rowvec3 com = {0.0, 0.0, 0.0};
    double total = 0.0;

    for (long ix = 0; ix < nx; ++ix) {
      for (long iy = 0; iy < ny; ++iy) {
        for (long iz = 0; iz < nz; ++iz) {
          double rho_i = rho_(flat_index(ix, iy, iz));
          com(0) += rho_i * (dx_ * static_cast<double>(ix));
          com(1) += rho_i * (dx_ * static_cast<double>(iy));
          com(2) += rho_i * (dx_ * static_cast<double>(iz));
          total += rho_i;
        }
      }
    }

    if (total > 0.0) {
      com /= total;
    }
    return com;
  }

  // ── I/O ─────────────────────────────────────────────────────────────────────

  void Density::save(const std::string& filename) const {
    rho_.save(filename, arma::raw_binary);
  }

  void Density::load(const std::string& filename) {
    arma::vec loaded;
    if (!loaded.load(filename, arma::raw_binary)) {
      throw std::runtime_error("Density::load: failed to read " + filename);
    }
    if (loaded.n_elem != rho_.n_elem) {
      throw std::invalid_argument(
          "Density::load: size mismatch (file has " + std::to_string(loaded.n_elem) + ", expected " +
          std::to_string(rho_.n_elem) + ")"
      );
    }
    rho_ = loaded;
  }

}  // namespace dft::density
