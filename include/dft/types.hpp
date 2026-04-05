#ifndef DFT_TYPES_HPP
#define DFT_TYPES_HPP

#include <armadillo>
#include <optional>
#include <string>
#include <vector>

namespace dft {

  // Density

  // Pure data: the density profile and optional external field on a grid.
  // No methods, no FFT, no derived quantities. All logic lives in free
  // functions elsewhere.
  struct Density {
    arma::vec values;
    arma::vec external_field;
  };

  // Species

  // Immutable physical identity of a species. Does not change during a
  // simulation. Multiple species in a mixture are distinguished by index.
  struct Species {
    std::string name;
    double hard_sphere_diameter;
  };

  // Mutable simulation state for one species: its density profile, the
  // accumulated functional derivative (force), chemical potential, and
  // an optional fixed-mass constraint.
  struct SpeciesState {
    Density density;
    arma::vec force;
    double chemical_potential{0.0};
    std::optional<double> fixed_mass;
  };

  // State

  // Aggregate of all mutable state in a DFT calculation: one SpeciesState
  // per component plus the thermodynamic temperature.
  struct State {
    std::vector<SpeciesState> species;
    double temperature;
  };

  // Crystal

  enum class Structure { BCC, FCC, HCP };

  enum class Orientation { _001, _010, _100, _110, _101, _011, _111 };

  enum class ExportFormat { XYZ, CSV };

  struct Lattice {
    Structure structure;
    Orientation orientation;
    std::vector<long> shape;
    arma::rowvec3 dimensions;
    arma::mat positions;
  };

  [[nodiscard]] auto build_lattice(
      Structure structure, Orientation orientation, const std::vector<long>& shape = {1, 1, 1}
  ) -> Lattice;

  [[nodiscard]] auto scaled_positions(const Lattice& lattice, double dnn) -> arma::mat;

  [[nodiscard]] auto scaled_positions(const Lattice& lattice, const arma::rowvec3& box) -> arma::mat;

  void export_lattice(
      const Lattice& lattice, const std::string& filename, ExportFormat format = ExportFormat::XYZ
  );

}  // namespace dft

#endif  // DFT_TYPES_HPP
