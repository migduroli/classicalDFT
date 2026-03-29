#ifndef CLASSICALDFT_FUNCTIONAL_FMT_WEIGHTS_H
#define CLASSICALDFT_FUNCTIONAL_FMT_WEIGHTS_H

#include "classicaldft_bits/functional/fmt/weighted_density.h"

#include <array>
#include <vector>

namespace dft::functional::fmt {

  /**
   * @brief The complete set of FMT weight functions for a hard sphere.
   *
   * Stores convolution channels grouped by tensorial rank:
   *  - `eta`: volume ($\Theta$) weight
   *  - `scalar`: surface ($\delta$) weight for $n_2$
   *  - `vector[3]`: vector weights for $\mathbf{v}_2$, indexed by spatial axis
   *  - `tensor(i,j)`: symmetric tensor weights for $T_{ij}$, 6 independent components
   *
   * The 11 stored components exploit Rosenfeld scaling:
   * $n_0 = n_2 / (4\pi R^2)$, $n_1 = n_2 / (4\pi R)$.
   * Only the surface, vector, and tensor weights need separate storage.
   */
  struct WeightedDensitySet {
    WeightedDensity eta;
    WeightedDensity scalar;
    std::array<WeightedDensity, 3> vector;                       // [x=0, y=1, z=2]
    std::array<std::array<WeightedDensity, 3>, 3> tensor_data_;  // symmetric; use tensor(i,j)

    static constexpr int NUM_INDEPENDENT = 11;

    /**
     * @brief Access the symmetric tensor component $(i, j)$.
     *
     * Indices follow Cartesian convention: 0=x, 1=y, 2=z.
     * Since $T_{ij} = T_{ji}$, only the upper triangle is stored.
     */
    [[nodiscard]] WeightedDensity& tensor(int i, int j) { return (i <= j) ? tensor_data_[i][j] : tensor_data_[j][i]; }

    [[nodiscard]] const WeightedDensity& tensor(int i, int j) const {
      return (i <= j) ? tensor_data_[i][j] : tensor_data_[j][i];
    }

    /**
     * @brief Applies a function to every convolution channel.
     *
     * The callable receives a `WeightedDensity&`. Iterates in order:
     * eta, scalar, vector[0..2], tensor upper triangle.
     */
    template <typename F>
    void for_each(F&& fn) {
      fn(eta);
      fn(scalar);
      for (auto& v : vector)
        fn(v);
      for (int i = 0; i < 3; ++i)
        for (int j = i; j < 3; ++j)
          fn(tensor_data_[i][j]);
    }

    template <typename F>
    void for_each(F&& fn) const {
      fn(eta);
      fn(scalar);
      for (const auto& v : vector)
        fn(v);
      for (int i = 0; i < 3; ++i)
        for (int j = i; j < 3; ++j)
          fn(tensor_data_[i][j]);
    }
  };

  /**
   * @brief Generates FMT weight functions analytically in Fourier space.
   *
   * Weight functions use the known transforms of spherical kernels:
   *  - Volume: $\hat{w}_3(\mathbf{k}) = 4\pi[\sin(kR) - kR\cos(kR)] / k^3$
   *  - Surface: $\hat{w}_2(\mathbf{k}) = 4\pi R \sin(kR) / k$
   *  - Vector: $\hat{w}_{2i} = i\,k_i\,f(k,R)$ with known prefactor
   *  - Tensor: $\hat{w}_{T,ij} = A(k,R)\,k_i k_j / k^2 + B(k,R)\,\delta_{ij}$
   */
  class WeightGenerator {
   public:
    /**
     * @brief Populates a WeightedDensitySet with the Fourier-space weights for a hard sphere.
     *
     * @param diameter Hard-sphere diameter $d$.
     * @param dx Grid spacing.
     * @param shape Grid dimensions $(N_x, N_y, N_z)$.
     * @param weights Output WeightedDensitySet (channels must already be constructed).
     */
    static void generate(double diameter, double dx, const std::vector<long>& shape, WeightedDensitySet& weights);

   private:
    [[nodiscard]] static double volume_hat(double k, double r);
    [[nodiscard]] static double surface_hat(double k, double r);
    [[nodiscard]] static double vector_prefactor(double k, double r);
    [[nodiscard]] static std::pair<double, double> tensor_coefficients(double k, double r);
  };

}  // namespace dft::functional::fmt

#endif  // CLASSICALDFT_FUNCTIONAL_FMT_WEIGHTS_H
