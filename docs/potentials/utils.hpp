#pragma once

#include <armadillo>
#include <vector>

namespace utils {

  // Convert an Armadillo vector to a std::vector<double> for matplotlib.

  [[nodiscard]] inline auto to_vec(const arma::vec& v) -> std::vector<double> {
    return arma::conv_to<std::vector<double>>::from(v);
  }

} // namespace utils
