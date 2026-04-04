#pragma once

#include <string>
#include <vector>

#ifdef DFT_HAS_MATPLOTLIB
#include "matplotlibcpp.h"

namespace plot {

  namespace detail {

    inline void gaussian_convolution(
        const std::vector<double>& x,
        const std::vector<double>& numerical,
        const std::vector<double>& analytical
    ) {
      namespace plt = matplotlibcpp;
      plt::figure_size(800, 500);

      plt::plot(x, analytical,
                {{"color", "#656775"}, {"linewidth", "2.0"}, {"label", R"(Analytical $g * g$)"}});
      plt::plot(x, numerical,
                {{"color", "#008080"}, {"linewidth", "1.5"}, {"linestyle", "--"}, {"label", "FFT convolution"}});

      plt::xlabel(R"($x$)");
      plt::ylabel(R"($(g * g)(x)$)");
      plt::title(R"(Self-convolution of Gaussian ($\sigma = 1.5$))");
      plt::legend();
      plt::grid(true);
      plt::tight_layout();
      plt::save("exports/gaussian_convolution.png");
      plt::close();
      std::cout << "  exports/gaussian_convolution.png\n";
    }

  }  // namespace detail

  inline void make_plots(
      const std::vector<double>& x,
      const std::vector<double>& numerical,
      const std::vector<double>& analytical
  ) {
    detail::gaussian_convolution(x, numerical, analytical);
  }

}  // namespace plot

#endif
