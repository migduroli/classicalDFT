#ifndef CLASSICALDFT_3D_MESH_H
#define CLASSICALDFT_3D_MESH_H

#include "classicaldft_bits/geometry/3D/element.h"
#include "classicaldft_bits/geometry/base/mesh.h"

namespace dft_core::geometry::three_dimensional {

  using sqbox_refwrap = std::reference_wrapper<SquareBox>;
  using sqbox_vec = std::vector<SquareBox>;
  using sqbox_map = std::unordered_map<int, sqbox_refwrap>;

  class SUQMesh : public dft_core::geometry::SUQMesh {
   private:
    // region Attributes

    /**
     * Vector of elements which constitutes the mesh
     */
    sqbox_vec elements_raw_ = {};

    /**
     * Dictionary which maps a global index with every mesh element
     */
    sqbox_map elements_ = {};

    // endregion

    void initialise(double dx);

   public:
    // region Cttors:

    explicit SUQMesh(double dx, std::vector<double>& dimensions, std::vector<double>& origin);

    // endregion

    // region Methods:

    void plot(const std::string& path = "", const bool interactive = true) const override;

    const std::vector<SquareBox>& elements() const;

    double element_volume() const final;

    // endregion
  };

}  // namespace dft_core::geometry::three_dimensional
#endif