#include <classicaldft>

#include <filesystem>
#include <iomanip>
#include <iostream>

#ifdef DFT_HAS_MATPLOTLIB
#include "matplotlibcpp.h"
#endif

using namespace dft::crystal;

int main() {
  std::filesystem::create_directories("exports");
  std::cout << std::fixed << std::setprecision(6);

  // ── Build crystals with different structures and orientations ──────────

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
    auto lattice = Lattice(cfg.structure, cfg.orientation);
    std::cout << std::setw(14) << cfg.label
              << std::setw(10) << lattice.size()
              << std::setw(12) << lattice.dimensions()(0)
              << std::setw(12) << lattice.dimensions()(1)
              << std::setw(12) << lattice.dimensions()(2)
              << "\n";
  }

  // ── Replicated lattice ────────────────────────────────────────────────

  std::cout << "\n=== Replicated FCC [001] (4x4x4) ===\n\n";

  auto fcc = Lattice(Structure::FCC, Orientation::_001, {4, 4, 4});
  std::cout << "  Atoms:      " << fcc.size() << "\n";
  std::cout << "  Dimensions: ("
            << fcc.dimensions()(0) << ", "
            << fcc.dimensions()(1) << ", "
            << fcc.dimensions()(2) << ")\n";

  // ── Scaled positions ──────────────────────────────────────────────────

  std::cout << "\n=== Position scaling (BCC [001], 2x2x2) ===\n\n";

  auto bcc = Lattice(Structure::BCC, Orientation::_001, {2, 2, 2});

  double dnn = 3.405;  // Argon nearest-neighbor distance in Angstrom
  arma::mat scaled = bcc.positions(dnn);
  std::cout << "  Uniform scale (dnn = " << dnn << " A):\n";
  for (arma::uword i = 0; i < std::min(bcc.size(), arma::uword(4)); ++i) {
    std::cout << "    atom " << i << ": ("
              << scaled(i, 0) << ", "
              << scaled(i, 1) << ", "
              << scaled(i, 2) << ")\n";
  }

  arma::rowvec3 box = {10.0, 12.0, 14.0};
  arma::mat aniso = bcc.positions(box);
  std::cout << "\n  Anisotropic scale (box = " << box(0) << " x " << box(1) << " x " << box(2) << "):\n";
  for (arma::uword i = 0; i < std::min(bcc.size(), arma::uword(4)); ++i) {
    std::cout << "    atom " << i << ": ("
              << aniso(i, 0) << ", "
              << aniso(i, 1) << ", "
              << aniso(i, 2) << ")\n";
  }

  // ── Export ─────────────────────────────────────────────────────────────

  std::string xyz_file = "exports/fcc_4x4x4.xyz";
  fcc.export_to(xyz_file, ExportFormat::XYZ);
  std::cout << "\n  Exported " << fcc.size() << " atoms to " << xyz_file << " (XYZ)\n";

  std::string csv_file = "exports/fcc_4x4x4.csv";
  fcc.export_to(csv_file, ExportFormat::CSV);
  std::cout << "  Exported " << fcc.size() << " atoms to " << csv_file << " (CSV)\n";

  // ── Plots ────────────────────────────────────────────────────────────

#ifdef DFT_HAS_MATPLOTLIB
  namespace plt = matplotlibcpp;
  plt::backend("Agg");

  auto plot_lattice = [](const Lattice& lat, const std::string& title,
                         const std::string& marker_style, const std::string& filename) {
    const auto& pos = lat.positions();
    std::vector<double> xs(pos.n_rows), ys(pos.n_rows);
    for (arma::uword i = 0; i < pos.n_rows; ++i) {
      xs[i] = pos(i, 0);
      ys[i] = pos(i, 1);
    }

    plt::figure_size(700, 700);
    plt::named_plot("Atoms", xs, ys, marker_style);
    double pad = 0.3;
    double xmax = lat.dimensions()(0);
    double ymax = lat.dimensions()(1);
    double range = std::max(xmax, ymax) + 2 * pad;
    plt::xlim(-pad, -pad + range);
    plt::ylim(-pad, -pad + range);
    plt::xlabel(R"($x / d_\mathrm{nn}$)");
    plt::ylabel(R"($y / d_\mathrm{nn}$)");
    plt::title(title);
    plt::grid(true);
    plt::tight_layout();
    plt::save(filename);
    plt::close();
    std::cout << "Plot saved: " << std::filesystem::absolute(filename) << std::endl;
  };

  {
    auto lat = Lattice(Structure::FCC, Orientation::_001, {4, 4, 4});
    plot_lattice(lat, R"(FCC [001] ($4^3$ unit cells))", "bo", "exports/fcc_001.png");
  }

  {
    auto lat = Lattice(Structure::BCC, Orientation::_001, {4, 4, 4});
    plot_lattice(lat, R"(BCC [001] ($4^3$ unit cells))", "rs", "exports/bcc_001.png");
  }

  {
    auto lat = Lattice(Structure::HCP, Orientation::_001, {4, 4, 4});
    plot_lattice(lat, R"(HCP [001] ($4^3$ unit cells))", "gD", "exports/hcp_001.png");
  }

  {
    auto lat = Lattice(Structure::BCC, Orientation::_110, {4, 4, 4});
    plot_lattice(lat, R"(BCC [110] ($4^3$ unit cells))", "m^", "exports/bcc_110.png");
  }

  {
    auto lat = Lattice(Structure::FCC, Orientation::_110, {4, 4, 4});
    plot_lattice(lat, R"(FCC [110] ($4^3$ unit cells))", "bo", "exports/fcc_110.png");
  }

  {
    auto lat = Lattice(Structure::FCC, Orientation::_111, {4, 4, 4});
    plot_lattice(lat, R"(FCC [111] ($4^3$ unit cells))", "k*", "exports/fcc_111.png");
  }

  {
    auto lat = Lattice(Structure::BCC, Orientation::_111, {4, 4, 4});
    plot_lattice(lat, R"(BCC [111] ($4^3$ unit cells))", "r^", "exports/bcc_111.png");
  }

  {
    auto lat = Lattice(Structure::HCP, Orientation::_010, {4, 4, 4});
    plot_lattice(lat, R"(HCP [010] ($4^3$ unit cells))", "gD", "exports/hcp_010.png");
  }

  {
    auto lat = Lattice(Structure::HCP, Orientation::_100, {4, 4, 4});
    plot_lattice(lat, R"(HCP [100] ($4^3$ unit cells))", "gs", "exports/hcp_100.png");
  }
#endif
}
