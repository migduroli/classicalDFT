#include "dft/crystal/lattice.h"

#include <cmath>
#include <fstream>
#include <stdexcept>
#include <utility>

namespace dft::crystal {

  namespace {

    using Pos3 = std::array<double, 3>;

    // Unit cell atom positions (fractional coordinates, side length = 1)
    std::vector<Pos3> regular_bcc() {
      return {
          {0.0, 0.0, 0.0},
          {0.5, 0.5, 0.5},
      };
    }

    std::vector<Pos3> regular_fcc() {
      return {
          {0.0, 0.0, 0.0},
          {0.5, 0.5, 0.0},
          {0.5, 0.0, 0.5},
          {0.0, 0.5, 0.5},
      };
    }

    std::vector<Pos3> hcp_001() {
      return {
          {0.0, 0.0, 0.0},
          {0.5, 0.5, 0.0},
          {0.0, 2.0 / 6.0, 0.5},
          {0.5, 5.0 / 6.0, 0.5},
      };
    }

    void scale_positions(std::vector<Pos3>& atoms, const Pos3& L) {
      for (auto& atom : atoms) {
        atom[0] *= L[0];
        atom[1] *= L[1];
        atom[2] *= L[2];
      }
    }

    void rotate_y_to_z(std::vector<Pos3>& atoms) {
      for (auto& a : atoms) {
        double temp = a[2];
        a[2] = a[1];
        a[1] = -temp;
      }
    }

    void rotate_x_to_z(std::vector<Pos3>& atoms) {
      for (auto& a : atoms) {
        double temp = a[2];
        a[2] = a[0];
        a[0] = -temp;
      }
    }

    void wrap_to_box(std::vector<Pos3>& atoms, const Pos3& L) {
      for (auto& a : atoms) {
        for (int d = 0; d < 3; ++d) {
          a[d] = std::fmod(a[d], L[d]);
          if (a[d] < 0.0)
            a[d] += L[d];
        }
      }
    }

    struct UnitCell {
      std::vector<Pos3> atoms;
      Pos3 dims{};
    };

    UnitCell make_unit_cell(Structure structure, Orientation orientation) {
      using enum Orientation;
      double a_bcc = 2.0 / std::sqrt(3.0);
      double a_fcc = std::sqrt(2.0);
      UnitCell uc;

      switch (structure) {
        case Structure::BCC: {
          switch (orientation) {
            case _001:
            case _010:
            case _100: {
              uc.atoms = regular_bcc();
              uc.dims = {a_bcc, a_bcc, a_bcc};
              scale_positions(uc.atoms, uc.dims);
              break;
            }
            case _110:
            case _101:
            case _011: {
              uc.atoms = regular_fcc();
              uc.dims = {a_bcc, a_bcc, a_bcc};
              if (orientation == _110) {
                uc.dims[0] *= std::sqrt(2.0);
                uc.dims[1] *= std::sqrt(2.0);
              }
              if (orientation == _101) {
                uc.dims[0] *= std::sqrt(2.0);
                uc.dims[2] *= std::sqrt(2.0);
              }
              if (orientation == _011) {
                uc.dims[1] *= std::sqrt(2.0);
                uc.dims[2] *= std::sqrt(2.0);
              }
              scale_positions(uc.atoms, uc.dims);
              break;
            }
            case _111: {
              uc.atoms = {
                  {1.0 / 6, 0.5, 0.0},
                  {0.0, 0.0, 1.0 / 3},
                  {1.0 / 6, 0.5, 0.5},
                  {0.0, 0.0, 5.0 / 6},
                  {3.0 / 6, 0.5, 1.0 / 3},
                  {1.0 / 3, 0.0, 2.0 / 3},
                  {3.0 / 6, 0.5, 5.0 / 6},
                  {1.0 / 3, 0.0, 1.0 / 6},
                  {5.0 / 6, 0.5, 2.0 / 3},
                  {2.0 / 3, 0.0, 0.0},
                  {5.0 / 6, 0.5, 1.0 / 6},
                  {2.0 / 3, 0.0, 0.5},
              };
              uc.dims = {2.0 * std::sqrt(2.0), 2.0 * std::sqrt(2.0 / 3.0), 2.0};
              scale_positions(uc.atoms, uc.dims);
              break;
            }
          }
          break;
        }
        case Structure::FCC: {
          switch (orientation) {
            case _001:
            case _010:
            case _100: {
              uc.atoms = regular_fcc();
              uc.dims = {a_fcc, a_fcc, a_fcc};
              scale_positions(uc.atoms, uc.dims);
              break;
            }
            case _110:
            case _101:
            case _011: {
              uc.atoms = regular_bcc();
              uc.dims = {1.0, 1.0, 1.0};
              if (orientation == _110) {
                uc.dims[2] *= std::sqrt(2.0);
              }
              if (orientation == _101) {
                uc.dims[1] *= std::sqrt(2.0);
              }
              if (orientation == _011) {
                uc.dims[0] *= std::sqrt(2.0);
              }
              scale_positions(uc.atoms, uc.dims);
              break;
            }
            case _111: {
              uc.atoms = {
                  {0.0, 0.0, 0.0},
                  {0.5, 0.5, 0.0},
                  {0.5, 1.0 / 6, 1.0 / 3},
                  {0.0, 4.0 / 6, 1.0 / 3},
                  {0.0, 2.0 / 6, 2.0 / 3},
                  {0.5, 5.0 / 6, 2.0 / 3},
              };
              uc.dims = {1.0, std::sqrt(3.0), std::sqrt(6.0)};
              scale_positions(uc.atoms, uc.dims);
              break;
            }
          }
          break;
        }
        case Structure::HCP: {
          if (orientation == _110 || orientation == _101 || orientation == _011 || orientation == _111) {
            throw std::invalid_argument("HCP only supports orientations 001, 010, 100");
          }
          uc.atoms = hcp_001();
          uc.dims = {1.0, std::sqrt(3.0), std::sqrt(8.0 / 3.0)};
          scale_positions(uc.atoms, uc.dims);

          if (orientation == _010) {
            rotate_y_to_z(uc.atoms);
            wrap_to_box(uc.atoms, {uc.dims[0], uc.dims[2], uc.dims[1]});
            std::swap(uc.dims[1], uc.dims[2]);
          } else if (orientation == _100) {
            rotate_x_to_z(uc.atoms);
            wrap_to_box(uc.atoms, {uc.dims[2], uc.dims[1], uc.dims[0]});
            std::swap(uc.dims[0], uc.dims[2]);
          }
          break;
        }
      }
      return uc;
    }

  }  // namespace

  Lattice::Lattice(Structure structure, Orientation orientation, std::vector<long> shape)
      : structure_(structure), orientation_(orientation), shape_(std::move(shape)) {
    if (shape_.size() != 3) {
      throw std::invalid_argument("Shape must have exactly 3 elements");
    }
    if (shape_[0] <= 0 || shape_[1] <= 0 || shape_[2] <= 0) {
      throw std::invalid_argument("Number of unit cells must be strictly positive in all directions");
    }
    build();
  }

  void Lattice::build() {
    auto uc = make_unit_cell(structure_, orientation_);
    auto n_per_cell = static_cast<arma::uword>(uc.atoms.size());
    auto total = n_per_cell * static_cast<arma::uword>(shape_[0] * shape_[1] * shape_[2]);

    positions_.set_size(total, 3);
    arma::uword idx = 0;
    for (long iz = 0; iz < shape_[2]; ++iz) {
      for (long iy = 0; iy < shape_[1]; ++iy) {
        for (long ix = 0; ix < shape_[0]; ++ix) {
          for (const auto& atom : uc.atoms) {
            positions_(idx, 0) = atom[0] + static_cast<double>(ix) * uc.dims[0];
            positions_(idx, 1) = atom[1] + static_cast<double>(iy) * uc.dims[1];
            positions_(idx, 2) = atom[2] + static_cast<double>(iz) * uc.dims[2];
            ++idx;
          }
        }
      }
    }

    dimensions_ = {
        uc.dims[0] * static_cast<double>(shape_[0]),
        uc.dims[1] * static_cast<double>(shape_[1]),
        uc.dims[2] * static_cast<double>(shape_[2]),
    };
  }

  arma::mat Lattice::positions(double dnn) const {
    return positions_ * dnn;
  }

  arma::mat Lattice::positions(const arma::rowvec3& box) const {
    arma::mat result = positions_;
    result.each_row() %= box / dimensions_;
    return result;
  }

  void Lattice::export_to(const std::string& filename, ExportFormat format) const {
    std::ofstream out(filename);
    if (!out) {
      throw std::runtime_error("Cannot open file for writing: " + filename);
    }
    switch (format) {
      case ExportFormat::XYZ:
        out << positions_.n_rows << "\n";
        out << "Crystal lattice\n";
        for (arma::uword i = 0; i < positions_.n_rows; ++i) {
          out << "Ar " << positions_(i, 0) << " " << positions_(i, 1) << " " << positions_(i, 2) << "\n";
        }
        break;
      case ExportFormat::CSV:
        out << "x,y,z\n";
        for (arma::uword i = 0; i < positions_.n_rows; ++i) {
          out << positions_(i, 0) << "," << positions_(i, 1) << "," << positions_(i, 2) << "\n";
        }
        break;
    }
  }

}  // namespace dft::crystal
