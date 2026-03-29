#ifndef DFT_CRYSTAL_LATTICE_H
#define DFT_CRYSTAL_LATTICE_H

#include "dft/crystal/types.h"

#include <armadillo>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace dft::crystal {

  /**
   * @brief Generates a crystal lattice with a given structure and orientation.
   *
   * The unit cell is constructed with nearest-neighbor distance 1, then
   * replicated over the requested number of unit cells in each direction.
   * Atom positions are stored as an N×3 arma::mat.
   */
  class Lattice {
   public:
    Lattice(Structure structure, Orientation orientation, std::vector<long> shape = {1, 1, 1});

    [[nodiscard]] const arma::mat& positions() const noexcept { return positions_; }
    [[nodiscard]] arma::mat positions(double dnn) const;
    [[nodiscard]] arma::mat positions(const arma::rowvec3& box) const;

    [[nodiscard]] arma::uword size() const noexcept { return positions_.n_rows; }
    [[nodiscard]] const arma::rowvec3& dimensions() const noexcept { return dimensions_; }
    [[nodiscard]] const std::vector<long>& shape() const noexcept { return shape_; }
    [[nodiscard]] Structure structure() const noexcept { return structure_; }
    [[nodiscard]] Orientation orientation() const noexcept { return orientation_; }

    void export_to(const std::string& filename, ExportFormat format = ExportFormat::XYZ) const;

   private:
    void build();

    Structure structure_;
    Orientation orientation_;
    std::vector<long> shape_;
    arma::rowvec3 dimensions_ = {0.0, 0.0, 0.0};
    arma::mat positions_;
  };

}  // namespace dft::crystal

#endif  // DFT_CRYSTAL_LATTICE_H
