#pragma once

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

// Requires dft.hpp to be included before this header (for dft::Lattice).
#ifdef DFT_HAS_MATPLOTLIB
#include "matplotlibcpp.h"

namespace plot {

inline void lattice(
    const dft::Lattice& lat,
    const std::string& title,
    const std::string& style,
    const std::string& filename
) {
  namespace plt = matplotlibcpp;
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
  std::cout << "Plot saved: " << filename << "\n";
}

}  // namespace plot

#endif
