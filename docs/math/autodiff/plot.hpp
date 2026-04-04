#pragma once

#include <cmath>
#include <string>
#include <vector>

#ifdef DFT_HAS_MATPLOTLIB
#include "matplotlibcpp.h"

namespace plot {

  inline void function_and_derivatives(
      const std::vector<double>& x,
      const std::vector<double>& f,
      const std::vector<double>& df,
      const std::vector<double>& d2f,
      const std::string& title,
      const std::string& f_label,
      const std::string& df_label,
      const std::string& d2f_label,
      const std::string& filename
  ) {
    namespace plt = matplotlibcpp;
    plt::figure_size(900, 550);

    plt::plot(x, f,
              {{"color", "#008080"}, {"linewidth", "2.0"}, {"label", f_label}});
    plt::plot(x, df,
              {{"color", "#00BBD5"}, {"linewidth", "2.0"}, {"label", df_label}});
    plt::plot(x, d2f,
              {{"color", "#E25822"}, {"linewidth", "2.0"}, {"label", d2f_label}});

    plt::xlabel(R"($x$)");
    plt::ylabel(R"($f(x)$)");
    plt::title(title);
    plt::legend();
    plt::grid(true);
    plt::tight_layout();
    plt::save(filename);
    plt::close();
    std::cout << "Plot saved: " << filename << "\n";
  }

  inline void autodiff_vs_finite_diff(
      const std::vector<double>& x,
      const std::vector<double>& ad_err1,
      const std::vector<double>& fd_err1,
      const std::vector<double>& ad_err2,
      const std::vector<double>& fd_err2
  ) {
    namespace plt = matplotlibcpp;
    plt::figure_size(900, 550);

    plt::semilogy(x, ad_err1,
                  {{"color", "#008080"}, {"linewidth", "2.0"}, {"label", R"(autodiff $|f' - f'_{\mathrm{exact}}|$)"}});
    plt::semilogy(x, fd_err1,
                  {{"color", "#008080"}, {"linewidth", "1.5"}, {"linestyle", "--"}, {"label", R"(finite diff $|f' - f'_{\mathrm{exact}}|$)"}});
    plt::semilogy(x, ad_err2,
                  {{"color", "#E25822"}, {"linewidth", "2.0"}, {"label", R"(autodiff $|f'' - f''_{\mathrm{exact}}|$)"}});
    plt::semilogy(x, fd_err2,
                  {{"color", "#E25822"}, {"linewidth", "1.5"}, {"linestyle", "--"}, {"label", R"(finite diff $|f'' - f''_{\mathrm{exact}}|$)"}});

    plt::xlabel(R"($x$)");
    plt::ylabel("Absolute error");
    plt::title(R"(Autodiff vs finite differences for $\log(1 + x^2)$)");
    plt::legend();
    plt::grid(true);
    plt::tight_layout();
    plt::save("exports/autodiff_accuracy.png");
    plt::close();
    std::cout << "Plot saved: exports/autodiff_accuracy.png\n";
  }

}  // namespace plot

#endif
