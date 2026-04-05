#include "dft.hpp"
#include "plot.hpp"

#include <filesystem>
#include <iostream>
#include <print>

using namespace dft;

int main() {
#ifdef DOC_SOURCE_DIR
  std::filesystem::current_path(DOC_SOURCE_DIR);
#endif
  std::filesystem::create_directories("exports");

#ifdef DFT_HAS_MATPLOTLIB
  matplotlibcpp::backend("Agg");
#endif

  // Unit cell properties for all structures and orientations.

  struct Config {
    Structure structure;
    Orientation orientation;
    const char* label;
  };

  Config configs[] = {
    { Structure::BCC, Orientation::_001, "BCC [001]" }, { Structure::BCC, Orientation::_110, "BCC [110]" },
    { Structure::BCC, Orientation::_111, "BCC [111]" }, { Structure::FCC, Orientation::_001, "FCC [001]" },
    { Structure::FCC, Orientation::_110, "FCC [110]" }, { Structure::FCC, Orientation::_111, "FCC [111]" },
    { Structure::HCP, Orientation::_001, "HCP [001]" }, { Structure::HCP, Orientation::_010, "HCP [010]" },
    { Structure::HCP, Orientation::_100, "HCP [100]" },
  };

  std::println(std::cout, "=== Unit cell properties (shape = 1x1x1) ===\n");
  std::println(std::cout, "{:>14s}{:>10s}{:>12s}{:>12s}{:>12s}", "Structure", "Atoms", "Lx", "Ly", "Lz");

  for (const auto& cfg : configs) {
    auto lattice = build_lattice(cfg.structure, cfg.orientation);
    std::println(
        std::cout,
        "{:>14s}{:>10d}{:>12.6f}{:>12.6f}{:>12.6f}",
        cfg.label,
        lattice.positions.n_rows,
        lattice.dimensions(0),
        lattice.dimensions(1),
        lattice.dimensions(2)
    );
  }

  // Replicated lattice.

  std::println(std::cout, "\n=== Replicated FCC [001] (4x4x4) ===\n");
  auto fcc = build_lattice(Structure::FCC, Orientation::_001, { 4, 4, 4 });
  std::println(std::cout, "  Atoms:      {}", fcc.positions.n_rows);
  std::println(std::cout, "  Dimensions: ({}, {}, {})", fcc.dimensions(0), fcc.dimensions(1), fcc.dimensions(2));

  // Scaled positions.

  std::println(std::cout, "\n=== Position scaling (BCC [001], 2x2x2) ===\n");
  auto bcc = build_lattice(Structure::BCC, Orientation::_001, { 2, 2, 2 });
  double dnn = 3.5;
  auto scaled = bcc.scaled_positions(dnn);
  std::println(std::cout, "  Uniform scale (dnn = {:.1f}):", dnn);
  for (arma::uword i = 0; i < std::min<arma::uword>(bcc.positions.n_rows, 4); ++i) {
    std::println(std::cout, "    atom {}: ({:.6f}, {:.6f}, {:.6f})", i, scaled(i, 0), scaled(i, 1), scaled(i, 2));
  }

  arma::rowvec3 box = { 10.0, 10.0, 10.0 };
  auto aniso = bcc.scaled_positions(box);
  std::println(std::cout, "\n  Anisotropic scale (box = 10x10x10):");
  for (arma::uword i = 0; i < std::min<arma::uword>(bcc.positions.n_rows, 4); ++i) {
    std::println(std::cout, "    atom {}: ({:.6f}, {:.6f}, {:.6f})", i, aniso(i, 0), aniso(i, 1), aniso(i, 2));
  }

  // Export.

  fcc.export_to("exports/fcc_4x4x4.xyz", ExportFormat::XYZ);
  std::println(std::cout, "\n  Exported to exports/fcc_4x4x4.xyz");

  fcc.export_to("exports/fcc_4x4x4.csv", ExportFormat::CSV);
  std::println(std::cout, "  Exported to exports/fcc_4x4x4.csv");

  // Plots.

  auto fcc_plot = build_lattice(Structure::FCC, Orientation::_001, { 4, 4, 4 });
  auto bcc_plot = build_lattice(Structure::BCC, Orientation::_110, { 4, 4, 4 });
  auto hcp_plot = build_lattice(Structure::HCP, Orientation::_001, { 4, 4, 4 });

#ifdef DFT_HAS_MATPLOTLIB
  plot::make_plots(fcc_plot, bcc_plot, hcp_plot);
#endif
}
