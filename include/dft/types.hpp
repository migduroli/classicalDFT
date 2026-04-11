#ifndef DFT_TYPES_HPP
#define DFT_TYPES_HPP

#include <armadillo>
#include <cctype>
#include <cstdint>
#include <optional>
#include <stdexcept>
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

  enum class Structure : std::uint8_t {
    BCC,
    FCC,
    HCP
  };

  enum class Orientation : std::uint8_t {
    _001,
    _010,
    _100,
    _110,
    _101,
    _011,
    _111
  };

  enum class ExportFormat : std::uint8_t {
    XYZ,
    CSV
  };

  struct Lattice {
    Structure structure;
    Orientation orientation;
    std::vector<long> shape;
    arma::rowvec3 dimensions;
    arma::mat positions;

    [[nodiscard]] auto scaled_positions(double dnn) const -> arma::mat;
    [[nodiscard]] auto scaled_positions(const arma::rowvec3& box) const -> arma::mat;
    void export_to(const std::string& filename, ExportFormat format = ExportFormat::XYZ) const;
  };

  [[nodiscard]] auto
  build_lattice(Structure structure, Orientation orientation, const std::vector<long>& shape = {1, 1, 1}) -> Lattice;

  // Parse a crystal structure name (case-insensitive): "FCC", "BCC", "HCP".

  [[nodiscard]] inline auto parse_structure(const std::string& name) -> Structure {
    std::string upper;
    upper.reserve(name.size());
    for (unsigned char ch : name) {
      if (std::isalnum(ch))
        upper.push_back(static_cast<char>(std::toupper(ch)));
    }
    if (upper == "FCC")
      return Structure::FCC;
    if (upper == "BCC")
      return Structure::BCC;
    if (upper == "HCP")
      return Structure::HCP;
    throw std::runtime_error("Unknown crystal structure: " + name);
  }

  // Parse a crystal orientation name (case-insensitive): "001", "010", etc.

  [[nodiscard]] inline auto parse_orientation(const std::string& name) -> Orientation {
    std::string upper;
    upper.reserve(name.size());
    for (unsigned char ch : name) {
      if (std::isalnum(ch))
        upper.push_back(static_cast<char>(std::toupper(ch)));
    }
    if (upper == "001")
      return Orientation::_001;
    if (upper == "010")
      return Orientation::_010;
    if (upper == "100")
      return Orientation::_100;
    if (upper == "110")
      return Orientation::_110;
    if (upper == "101")
      return Orientation::_101;
    if (upper == "011")
      return Orientation::_011;
    if (upper == "111")
      return Orientation::_111;
    throw std::runtime_error("Unknown crystal orientation: " + name);
  }

} // namespace dft

#endif // DFT_TYPES_HPP
