#include <classicaldft>

#include <iomanip>
#include <iostream>

using namespace dft_core::physics::crystal;

int main() {
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

  // ── Grace plots ───────────────────────────────────────────────────────

#ifdef DFT_HAS_GRACE
  namespace gp = dft_core::grace_plot;

  auto plot_lattice = [](const Lattice& lat, const std::string& title, gp::Color color, gp::Symbol symbol,
                         const std::string& filename) {
    auto g = gp::Grace();
    g.set_title(title);
    g.set_label("x / d\\snn\\N", gp::Axis::X);
    g.set_label("y / d\\snn\\N", gp::Axis::Y);

    const auto& pos = lat.positions();
    std::vector<double> xs(pos.n_rows), ys(pos.n_rows);
    for (arma::uword i = 0; i < pos.n_rows; ++i) {
      xs[i] = pos(i, 0);
      ys[i] = pos(i, 1);
    }

    auto ds = g.add_dataset(xs, ys);
    g.set_line_type(gp::LineStyle::NO_LINE, ds);
    g.set_symbol(symbol, ds);
    g.set_symbol_color(color, ds);
    g.set_symbol_fill(color, ds);
    g.set_symbol_size(0.4, ds);
    g.set_color(color, ds);

    double pad = 0.3;
    g.set_x_limits(-pad, lat.dimensions()(0) + pad);
    g.set_y_limits(-pad, lat.dimensions()(1) + pad);
    g.set_ticks(1.0, 1.0);
    g.print_to_file(filename, gp::ExportFormat::PNG);
    g.redraw_and_wait(false, false);
  };

  {
    auto lat = Lattice(Structure::FCC, Orientation::_001, {4, 4, 4});
    plot_lattice(lat, "FCC [001] (4\\S3\\N unit cells)", gp::Color::BLUE, gp::Symbol::CIRCLE,
                "exports/fcc_001.png");
  }

  {
    auto lat = Lattice(Structure::BCC, Orientation::_001, {4, 4, 4});
    plot_lattice(lat, "BCC [001] (4\\S3\\N unit cells)", gp::Color::RED, gp::Symbol::SQUARE,
                "exports/bcc_001.png");
  }

  {
    auto lat = Lattice(Structure::HCP, Orientation::_001, {4, 4, 4});
    plot_lattice(lat, "HCP [001] (4\\S3\\N unit cells)", gp::Color::DARKGREEN, gp::Symbol::DIAMOND,
                "exports/hcp_001.png");
  }

  {
    auto lat = Lattice(Structure::BCC, Orientation::_110, {4, 4, 4});
    plot_lattice(lat, "BCC [110] (4\\S3\\N unit cells)", gp::Color::MAROON, gp::Symbol::TRIANGLE_UP,
                "exports/bcc_110.png");
  }

  {
    auto lat = Lattice(Structure::FCC, Orientation::_111, {4, 4, 4});
    plot_lattice(lat, "FCC [111] (4\\S3\\N unit cells)", gp::Color::VIOLET, gp::Symbol::STAR,
                "exports/fcc_111.png");
  }
#endif
}
