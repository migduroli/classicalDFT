#pragma once

#include <armadillo>
#include <vector>

namespace utils {

  // Extract the z-averaged 1D profile from a 3D density field.

  [[nodiscard]] inline auto extract_z_profile(const arma::vec& rho_3d, long nx, long ny, long nz)
      -> std::vector<double> {
    arma::mat rho_mat = arma::reshape(rho_3d, nz, nx * ny);
    arma::vec avg = arma::mean(rho_mat, 1);
    return arma::conv_to<std::vector<double>>::from(avg);
  }

} // namespace utils
