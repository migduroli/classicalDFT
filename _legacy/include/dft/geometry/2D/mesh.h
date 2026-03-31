#ifndef DFT_GEOMETRY_2D_MESH_H
#define DFT_GEOMETRY_2D_MESH_H

#include "dft/geometry/2D/element.h"
#include "dft/geometry/base/mesh.h"

namespace dft::geometry::two_dimensional {

  using sqbox_refwrap = std::reference_wrapper<SquareBox>;
  using sqbox_vec = std::vector<SquareBox>;
  using sqbox_map = std::unordered_map<int, sqbox_refwrap>;

  class SUQMesh : public dft::geometry::SUQMesh {
   private:
    // region Attributes:

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

    // region Overloads:

    /**
     * Gets the graphical representation of the SUQMesh
     */
    void plot(const std::string& path = "", const bool interactive = true) const override;

    /**
     * Gets a reference to the vector which contains the mesh elements
     * @return
     */
    const std::vector<SquareBox>& elements() const;

    /**
     * Gets the volume of the fundamental element of the mesh
     * @return
     */
    double element_volume() const final;

    // endregion
  };

}  // namespace dft::geometry::two_dimensional

#endif
