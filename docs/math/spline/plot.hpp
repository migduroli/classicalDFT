#pragma once

#include <string>
#include <vector>

#ifdef DFT_HAS_MATPLOTLIB
#include "matplotlibcpp.h"

namespace plot {

  inline void spline_interpolation(
      const std::vector<double>& x_knots, const std::vector<double>& y_knots,
      const std::vector<double>& x_eval, const std::vector<double>& y_spline,
      const std::vector<double>& y_exact
  ) {
    namespace plt = matplotlibcpp;
    plt::figure_size(800, 500);

    plt::plot(x_eval, y_exact,
              {{"color", "#656775"}, {"linewidth", "2.0"}, {"label", R"($\sin(x)$ exact)"}});
    plt::plot(x_eval, y_spline,
              {{"color", "#008080"}, {"linewidth", "1.5"}, {"linestyle", "--"}, {"label", "Cubic spline"}});
    plt::plot(x_knots, y_knots,
              {{"color", "#E25822"}, {"marker", "o"}, {"markersize", "7"}, {"linestyle", "None"}, {"label", "Knot points"}});

    plt::xlabel(R"($x$)");
    plt::ylabel(R"($f(x)$)");
    plt::title(R"(Cubic spline interpolation of $\sin(x)$)");
    plt::legend();
    plt::grid(true);
    plt::tight_layout();
    plt::save("exports/spline_interpolation.png");
    plt::close();
    std::cout << "Plot saved: exports/spline_interpolation.png\n";
  }

  inline void spline_derivatives(
      const std::vector<double>& x_eval,
      const std::vector<double>& y_func,
      const std::vector<double>& y_deriv1,
      const std::vector<double>& y_deriv2,
      const std::vector<double>& y_exact_func,
      const std::vector<double>& y_exact_d1,
      const std::vector<double>& y_exact_d2
  ) {
    namespace plt = matplotlibcpp;
    plt::figure_size(900, 600);

    plt::plot(x_eval, y_exact_func,
              {{"color", "#656775"}, {"linewidth", "2.0"}, {"label", R"($\sin(x)$)"}});
    plt::plot(x_eval, y_func,
              {{"color", "#008080"}, {"linewidth", "1.5"}, {"linestyle", "--"}, {"label", R"(spline $f$)"}});

    plt::plot(x_eval, y_exact_d1,
              {{"color", "#00BBD5"}, {"linewidth", "2.0"}, {"label", R"($\cos(x)$)"}});
    plt::plot(x_eval, y_deriv1,
              {{"color", "#00BBD5"}, {"linewidth", "1.5"}, {"linestyle", "--"}, {"label", R"(spline $f'$)"}});

    plt::plot(x_eval, y_exact_d2,
              {{"color", "#E25822"}, {"linewidth", "2.0"}, {"label", R"($-\sin(x)$)"}});
    plt::plot(x_eval, y_deriv2,
              {{"color", "#E25822"}, {"linewidth", "1.5"}, {"linestyle", "--"}, {"label", R"(spline $f''$)"}});

    plt::xlabel(R"($x$)");
    plt::ylabel(R"($f(x)$)");
    plt::title(R"(Spline function and derivatives)");
    plt::legend();
    plt::grid(true);
    plt::tight_layout();
    plt::save("exports/spline_derivatives.png");
    plt::close();
    std::cout << "Plot saved: exports/spline_derivatives.png\n";
  }

}  // namespace plot

#endif
