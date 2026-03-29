#ifndef DFT_GEOMETRY_2D_UNIFORM_MESH_H
#define DFT_GEOMETRY_2D_UNIFORM_MESH_H

#include "dft/geometry/2D/mesh.h"

namespace dft::geometry::two_dimensional {

  /**
   * @brief Structured uniform mesh with periodic boundary conditions (2D).
   *
   * Extends SUQMesh with periodic boundary conditions via the wrap method.
   * The grid spacing is stored and exposed via the spacing inspector.
   * The plot method is inherited from SUQMesh.
   */
  class UniformMesh : public SUQMesh {
   private:
    // region Attributes:

    /**
     * Grid spacing in all directions
     */
    double dx_ = 0.0;

    // endregion

   public:
    // region Cttors:

    explicit UniformMesh(double dx, std::vector<double>& dimensions, std::vector<double>& origin);

    // endregion

    // region Inspectors:

    /**
     * Gets the uniform grid spacing
     * @return The spacing value
     */
    double spacing() const;

    // endregion

    // region Methods:

    /**
     * Wraps a position into the periodic box using periodic boundary conditions
     * @param position The position to be wrapped
     * @return A new Vertex with coordinates wrapped into [origin, origin + dimensions)
     */
    Vertex wrap(const Vertex& position) const;

    // endregion
  };

}  // namespace dft::geometry::two_dimensional

#endif
