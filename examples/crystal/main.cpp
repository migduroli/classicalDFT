#include "dft.hpp"

#include <filesystem>
#include <iomanip>
#include <iostream>

#ifdef DFT_HAS_MATPLOTLIB
#include "matplotlibcpp.h"
#endif

using namespace dft;

int main() {
#ifdef EXAMPLE_SOURCE_DIR
  std::filesystem::current_path(EXAMPLE_SOURCE_DIR);
#endif
  std::filesystem::create_directories("exports");
  std::cout << std::fixed << std::setprecision(6);

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
      {Structure::BCC, Orientation::_001, "BCC [001]"},
      {Structure::BCC, Orientation::_110, "BCC [110]"},
      {Structure::BCC, Orientation::_111, "BCC [111]"},
      {Structure::FCC, Orientation::_001, "FCC [001]"},
      {Structure::FCC, Orientation::_110, "FCC [110]"},
      {Structure::FCC, Orientation::_111, "FCC [111]"},
      {Structure::HCP, Orientation::_001, "HCP [001]"},
      {Structure::HCP, Orientation::_010, "HCP [010]"},
      {Structure::HCP, Orientation::_100, "HCP [100]"},
  };

  std::cout << "=== Unit cell properties (shape = 1x1x1) ===\n\n";
  std::cout << std::setw(14) << "Structure"
            << std::setw(10) << "Atoms"
            << std::setw(12) << "Lx"
            << std::setw(12) << "Ly"
            << std::setw(12) << "Lz"
            << "\n";

  for (const auto& cfg : configs) {
    auto lattice = build_lattice(cfg.structure, cfg.orientation);
    std::cout << std::setw(14) << cfg.label
              << std::setw(10) << lattice.positions.n_rows
              << std::setw(12) << lattice.dimensions(0)
              << std::setw(12) << lattice.dimensions(1)
              << std::setw(12) << lattice.dimensions(2)
              << "\n";
  }

  // Replicated lattice.

  std::cout << "\n=== Replicated FCC [001] (4x4x4) ===\n\n";
  auto fcc = build_lattice(Structure::FCC, Orientation::_001, {4, 4, 4});
  std::cout << "  Atoms:      " << fcc.positions.n_rows << "\n";
  std::cout << "  Dimensions: ("
            << fcc.dimensions(0) << ", "
            << fcc.dimensions(1) << ", "
            << fcc.dimensions(2) << ")\n";

  // Scaled positions.

  std::cout << "\n=== Position scaling (BCC [001], 2x2x2) ===\n\n";
  auto bcc = build_lattice(Structure::BCC, Orientation::_001, {2, 2, 2});
  double dnn = 3.5;
  auto scaled = scaled_positions(bcc, dnn);
  std::cout << "  Uniform scale (dnn = " << dnn << "):\n";
  for (arma::uword i = 0; i < std::min<arma::uword>(bcc.positions.n_rows, 4); ++i) {
    std::cout << "    atom " << i << ": ("
              << scaled(i, 0) << ", " << scaled(i, 1) << ", " << scaled(i, 2) << ")\n";
  }

  arma::rowvec3 box = {10.0, 10.0, 10.0};
  auto aniso = scaled_positions(bcc, box);
  std::cout << "\n  Anisotropic scale (box = 10x10x10):\n";
  for (arma::uword i = 0; i < std::min<arma::uword>(bcc.positions.n_rows, 4); ++i) {
    std::cout << "    atom " << i << ": ("
              << aniso(i, 0) << ", " << aniso(i, 1) << ", " << aniso(i, 2) << ")\n";
  }

  // Export.

  export_lattice(fcc, "exports/fcc_4x4x4.xyz", ExportFormat::XYZ);
  std::cout << "\n  Exported to exports/fcc_4x4x4.xyz\n";

  export_lattice(fcc, "exports/fcc_4x4x4.csv", ExportFormat::CSV);
  std::cout << "  Exported to exports/fcc_4x4x4.csv\n";

  // Plots.

#ifdef DFT_HAS_MATPLOTLIB
  namespace plt = matplotlibcpp;

  auto plot_lattice = [](const Lattice& lat, const std::string& title,
                         const std::string& style, const std::string& filename) {
    std::vector<double> xs(lat.positions.n_rows), ys(lat.positions.n_rows);
    for (arma::uword i = 0; i < lat.positions.n_rows; ++i) {
      xs[i] = lat.positions(i, 0);
      ys[i] = lat.positions(i, 1);
    }

    plt::figure_size(700, 700);
    plt::named_plot("Atoms", xs, ys, style);
    double pad = 0.3;
    double range = std::max(lat.dimensions(0), lat.dimensions(1)) + 2 * pad;
    plt::xlim(-pad, -pad + range);
    plt::ylim(-pad, -pad + range);
    plt::xlabel(R"($x / d_\mathrm{nn}$)");
    plt::ylabel(R"($y / d_\mathrm{nn}$)");
    plt::title(title);
    plt::grid(true);
    plt::tight_layout();
    plt::save(filename);
    plt::close();
    std::cout << "Plot saved: " << std::filesystem::absolute(filename) << "\n";
  };

  plot_lattice(
      build_lattice(Structure::FCC, Orientation::_001, {4, 4, 4}),
      R"(FCC [001] ($4^3$ unit cells))", "bo", "exports/fcc_001.png"
  );
  plot_lattice(
      build_lattice(Structure::BCC, Orientation::_110, {4, 4, 4}),
      R"(BCC [110] ($4^3$ unit cells))", "rs", "exports/bcc_110.png"
  );
  plot_lattice(
      build_lattice(Structure::HCP, Orientation::_001, {4, 4, 4}),
      R"(HCP [001] ($4^3$ unit cells))", "g^", "exports/hcp_001.png"
  );
#endif
}
