#pragma once

#include <armadillo>
#include <string>
#include <variant>

namespace cdft {

  // ── Common type aliases ───────────────────────────────────────────────────

  using Vector3 = arma::rowvec3;
  using Matrix33 = arma::mat33;
  using RealVector = arma::vec;

  // ── Export / IO enums ─────────────────────────────────────────────────────

  enum class FileFormat { CSV, XYZ };

}  // namespace cdft
