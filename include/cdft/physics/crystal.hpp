#pragma once

#include "cdft/core/types.hpp"

#include <armadillo>
#include <string>
#include <vector>

namespace cdft::physics {

  enum class CrystalStructure { BCC, FCC, HCP };

  enum class Orientation { _001, _010, _100, _110, _101, _011, _111 };

  class Lattice {
   public:
    Lattice(CrystalStructure structure, Orientation orientation, std::vector<long> shape = {1, 1, 1});

    [[nodiscard]] const arma::mat& positions() const noexcept { return positions_; }
    [[nodiscard]] arma::mat positions(double nearest_neighbor_distance) const;
    [[nodiscard]] arma::mat positions(const Vector3& box) const;

    [[nodiscard]] arma::uword size() const noexcept { return positions_.n_rows; }
    [[nodiscard]] const Vector3& dimensions() const noexcept { return dimensions_; }
    [[nodiscard]] const std::vector<long>& shape() const noexcept { return shape_; }
    [[nodiscard]] CrystalStructure structure() const noexcept { return structure_; }
    [[nodiscard]] Orientation orientation() const noexcept { return orientation_; }

    void export_to(const std::string& filename, FileFormat format = FileFormat::XYZ) const;

   private:
    void build();

    CrystalStructure structure_;
    Orientation orientation_;
    std::vector<long> shape_;
    Vector3 dimensions_ = {0.0, 0.0, 0.0};
    arma::mat positions_;
  };

}  // namespace cdft::physics
