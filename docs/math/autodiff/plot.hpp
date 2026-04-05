#pragma once

#include <cmath>
#include <string>
#include <vector>

#ifdef DFT_HAS_MATPLOTLIB
#include "matplotlibcpp.h"

namespace plot {

  namespace detail {

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

      plt::plot(x, f, { { "color", "#008080" }, { "linewidth", "2.0" }, { "label", f_label } });
      plt::plot(x, df, { { "color", "#00BBD5" }, { "linewidth", "2.0" }, { "label", df_label } });
      plt::plot(x, d2f, { { "color", "#E25822" }, { "linewidth", "2.0" }, { "label", d2f_label } });

      plt::xlabel(R"($x$)");
      plt::ylabel(R"($f(x)$)");
      plt::title(title);
      plt::legend();
      plt::grid(true);
      plt::tight_layout();
      plt::save(filename);
      plt::clf();
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

      plt::named_semilogy(R"(autodiff $|f' - f'_{\mathrm{exact}}|$)", x, ad_err1, "b-");
      plt::named_semilogy(R"(finite diff $|f' - f'_{\mathrm{exact}}|$)", x, fd_err1, "b--");
      plt::named_semilogy(R"(autodiff $|f'' - f''_{\mathrm{exact}}|$)", x, ad_err2, "r-");
      plt::named_semilogy(R"(finite diff $|f'' - f''_{\mathrm{exact}}|$)", x, fd_err2, "r--");

      plt::xlabel(R"($x$)");
      plt::ylabel("Absolute error");
      plt::title(R"(Autodiff vs finite differences for $\log(1 + x^2)$)");
      plt::legend();
      plt::grid(true);
      plt::tight_layout();
      plt::save("exports/autodiff_accuracy.png");
      plt::clf();
      plt::close();
      std::cout << "Plot saved: exports/autodiff_accuracy.png\n";
    }

  }  // namespace detail

  inline void make_plots(
      const std::vector<double>& x_sin,
      const std::vector<double>& f_sin,
      const std::vector<double>& d1_sin,
      const std::vector<double>& d2_sin,
      const std::vector<double>& x_err,
      const std::vector<double>& ad_err1,
      const std::vector<double>& fd_err1,
      const std::vector<double>& ad_err2,
      const std::vector<double>& fd_err2
  ) {
    detail::function_and_derivatives(
        x_sin,
        f_sin,
        d1_sin,
        d2_sin,
        R"(Autodiff derivatives of $\sin(x)$)",
        R"($\sin(x)$)",
        R"($\cos(x)$)",
        R"($-\sin(x)$)",
        "exports/autodiff_sin.png"
    );
    detail::autodiff_vs_finite_diff(x_err, ad_err1, fd_err1, ad_err2, fd_err2);
  }

}  // namespace plot

#endif
