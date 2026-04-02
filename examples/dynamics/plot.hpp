#pragma once

#include <string>
#include <vector>

#ifdef DFT_HAS_MATPLOTLIB
#include "matplotlibcpp.h"

namespace plot {

inline void fire2_energy(const std::vector<double>& steps, const std::vector<double>& energies) {
  namespace plt = matplotlibcpp;
  plt::figure_size(700, 500);
  plt::named_plot("Energy", steps, energies, "b-");
  plt::xlabel("Step");
  plt::ylabel("Energy");
  plt::title("FIRE2: energy convergence");
  plt::legend();
  plt::tight_layout();
  plt::save("exports/fire2_energy.png");
  plt::close();
  std::cout << "  Plot saved: exports/fire2_energy.png\n";
}

inline void ddft_variance(const std::vector<double>& times, const std::vector<double>& variances) {
  namespace plt = matplotlibcpp;
  plt::figure_size(700, 500);
  plt::named_plot("Variance", times, variances, "b-o");
  plt::xlabel("Time");
  plt::ylabel("Density variance");
  plt::title("DDFT: ideal gas relaxation");
  plt::legend();
  plt::tight_layout();
  plt::save("exports/ddft_variance.png");
  plt::close();
  std::cout << "  Plot saved: exports/ddft_variance.png\n";
}

}  // namespace plot

#endif
