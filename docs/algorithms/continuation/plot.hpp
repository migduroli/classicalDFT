#pragma once

#include "utils.hpp"

#include <array>
#include <cmath>
#include <dft/plotting/matplotlib.hpp>
#include <format>
#include <limits>
#include <map>
#include <string>
#include <vector>

#ifdef DFT_HAS_MATPLOTLIB
#include "matplotlibcpp.h"

namespace plot {

  namespace plt = matplotlibcpp;

  inline const std::string ink = "#3d3d3a";
  inline const std::string muted = "#8a8980";
  inline const std::vector<std::string> branch_colors = {"#2a78d6", "#eb6834", "#1baf7a"};

  // Blend a #rrggbb colour towards white: t = 0 keeps it, t = 1 gives white.
  inline auto tint(const std::string& hex, double t) -> std::string {
    auto channel = [&](int k) {
      int c = std::stoi(hex.substr(1 + 2 * static_cast<std::size_t>(k), 2), nullptr, 16);
      return static_cast<int>(std::lround(c + t * (255 - c)));
    };
    return std::format("#{:02x}{:02x}{:02x}", channel(0), channel(1), channel(2));
  }

  // Line style for a given number of negative eigenvalues.
  inline auto index_style(int index) -> std::string {
    switch (index) {
      case 0:
        return "-";
      case 1:
        return "--";
      case 2:
        return "-.";
      default:
        return ":";
    }
  }

  // Split a curve into runs of constant index; consecutive runs share their
  // end point so the plotted line stays connected.
  struct Run {
    std::vector<double> x;
    std::vector<double> y;
    int index;
  };

  inline auto runs(const std::vector<double>& x, const std::vector<double>& y, const std::vector<int>& index)
      -> std::vector<Run> {
    std::vector<Run> out;
    for (std::size_t k = 0; k < x.size(); ++k) {
      if (out.empty() || index[k] != out.back().index) {
        Run r{.index = index[k]};
        if (!out.empty()) {
          r.x.push_back(out.back().x.back());
          r.y.push_back(out.back().y.back());
        }
        out.push_back(std::move(r));
      }
      out.back().x.push_back(x[k]);
      out.back().y.push_back(y[k]);
    }
    return out;
  }

  // Stable arcs solid, unstable arcs dashed.
  inline void stability_line(
      const std::vector<double>& x,
      const std::vector<double>& y,
      const std::vector<int>& index,
      const std::string& color,
      bool label
  ) {
    std::vector<int> unstable(index.size());
    for (std::size_t k = 0; k < index.size(); ++k)
      unstable[k] = index[k] > 0 ? 1 : 0;
    bool stable_labelled = false;
    bool unstable_labelled = false;
    for (const auto& r : runs(x, y, unstable)) {
      std::map<std::string, std::string>
          keywords{{"color", color}, {"linewidth", "2"}, {"linestyle", r.index == 0 ? "-" : "--"}};
      if (label && r.index == 0 && !stable_labelled) {
        keywords["label"] = R"(stable, $n_- = 0$)";
        stable_labelled = true;
      }
      if (label && r.index == 1 && !unstable_labelled) {
        keywords["label"] = R"(unstable, $n_- \geq 1$)";
        unstable_labelled = true;
      }
      plt::plot(r.x, r.y, keywords);
    }
  }

  // plt::subplot passes floats, which recent matplotlib rejects; select the
  // axes through the library wrapper instead.
  inline void subplot(int rows, int columns, int index) {
    PyObject* pyplot = dft::plotting::detail::import_module("matplotlib.pyplot");
    Py_DECREF(dft::plotting::detail::select_subplot(pyplot, rows, columns, index));
    Py_DECREF(pyplot);
  }

  // Axes operations that matplotlibcpp does not wrap, through the library's
  // Python helpers.

  // Inset at (x0, y0, width, height) in the axes coordinates of the panel
  // selected by subplot(rows, columns, index), added as a figure axes (which
  // becomes current, unlike an Axes.inset_axes child) so that the plt:: calls
  // that follow draw into it. Call after the layout of the panels is final.
  inline void inset(int rows, int columns, int index, double x0, double y0, double width, double height) {
    namespace py = dft::plotting::detail;
    PyObject* pyplot = py::import_module("matplotlib.pyplot");
    PyObject* parent = py::select_subplot(pyplot, rows, columns, index);
    PyObject* get_position = py::get_attr(parent, "get_position");
    PyObject* box = PyObject_CallObject(get_position, nullptr);
    auto field = [box](const char* name) {
      PyObject* value = py::get_attr(box, name);
      const double out = PyFloat_AsDouble(value);
      Py_DECREF(value);
      return out;
    };
    const double left = field("x0");
    const double bottom = field("y0");
    const double w = field("width");
    const double h = field("height");
    Py_DECREF(py::add_axes_rect(pyplot, left + x0 * w, bottom + y0 * h, width * w, height * h));
    Py_DECREF(box);
    Py_DECREF(get_position);
    Py_DECREF(parent);
    Py_DECREF(pyplot);
  }

  // Logarithmic scale on both axes of the current axes.
  inline void log_axes() {
    namespace py = dft::plotting::detail;
    PyObject* pyplot = py::import_module("matplotlib.pyplot");
    PyObject* ax = py::current_axes(pyplot);
    py::call_axes_string_method(ax, "set_xscale", "log");
    py::call_axes_string_method(ax, "set_yscale", "log");
    Py_DECREF(ax);
    Py_DECREF(pyplot);
  }

  // Legend of the current axes centred below it, in the given number of columns.
  inline void legend_below(long columns) {
    namespace py = dft::plotting::detail;
    PyObject* pyplot = py::import_module("matplotlib.pyplot");
    PyObject* ax = py::current_axes(pyplot);
    PyObject* legend = py::get_attr(ax, "legend");
    PyObject* kwargs = PyDict_New();
    py::set_dict_string(kwargs, "loc", "upper center");
    PyObject* anchor = py::to_pylist_1d({0.5, -0.13});
    PyDict_SetItemString(kwargs, "bbox_to_anchor", anchor);
    py::set_dict_long(kwargs, "ncol", columns);
    py::set_dict_string(kwargs, "fontsize", "small");
    PyObject* args = PyTuple_New(0);
    Py_XDECREF(PyObject_Call(legend, args, kwargs));
    Py_DECREF(args);
    Py_DECREF(anchor);
    Py_DECREF(kwargs);
    Py_DECREF(legend);
    Py_DECREF(ax);
    Py_DECREF(pyplot);
  }

  // Save as PDF and PNG; layout = false keeps the current layout, for figures
  // with insets placed after an explicit tight_layout().
  inline void save(const std::string& name, bool layout = true) {
    if (layout)
      plt::tight_layout();
    plt::save("exports/" + name + ".pdf");
    plt::save("exports/" + name + ".png");
    plt::clf();
    plt::close();
    std::cout << "Plot saved: exports/" << name << ".png\n";
  }

  inline void fold_markers(const utils::Problem& problem, const utils::Branch& uniform, bool omega) {
    std::vector<double> fx, fy;
    for (const auto& f : uniform.folds) {
      fx.push_back(f.mu);
      fy.push_back(omega ? problem.grand_potential(f.point.x, f.mu) : problem.mass(f.point.x));
    }
    plt::plot(
        fx,
        fy,
        {{"color", ink}, {"marker", "o"}, {"markersize", "8"}, {"linestyle", "None"}, {"label", "folds"}}
    );
  }

  // Figure 1: N against mu on the uniform branch.
  inline void s_curve(const utils::Problem& problem, const utils::Branch& uniform) {
    plt::figure_size(800, 560);
    stability_line(uniform.mu, uniform.mass, uniform.index, branch_colors[0], true);
    fold_markers(problem, uniform, false);
    for (const auto& f : uniform.folds) {
      double n = problem.mass(f.point.x);
      plt::annotate(std::format(R"($\mu = {:+.4f}$)", f.mu), f.mu + (f.mu > 0 ? 0.02 : -0.2), n + (n > 0 ? 1.0 : -2.0));
    }
    plt::axhline(0.0, 0.0, 1.0, {{"color", muted}, {"linewidth", "0.6"}});
    plt::axvline(0.0, 0.0, 1.0, {{"color", muted}, {"linewidth", "0.6"}});
    plt::xlim(-0.8, 0.8);
    plt::xlabel(R"($\mu$)");
    plt::ylabel(R"($N = \int_0^L \rho\,dx$)");
    plt::title("Uniform branch: the S-curve");
    plt::legend({{"loc", "upper left"}});
    plt::grid(true);
    save("s_curve");
  }

  // Figure 2: Omega against mu on the uniform branch.
  inline void swallowtail(const utils::Problem& problem, const utils::Branch& uniform) {
    plt::figure_size(800, 560);
    stability_line(uniform.mu, uniform.omega, uniform.index, branch_colors[0], true);
    fold_markers(problem, uniform, true);
    plt::plot(
        {0.0},
        {0.0},
        {{"color", ink},
         {"marker", "D"},
         {"markersize", "7"},
         {"linestyle", "None"},
         {"label", R"(crossing $\mu_0 = 0$)"}}
    );
    arma::vec mid(problem.nodes, arma::fill::zeros);
    double omega_mid = problem.grand_potential(mid, 0.0);
    plt::plot({0.0, 0.0}, {0.0, omega_mid}, {{"color", muted}, {"linewidth", "1"}, {"linestyle", ":"}});
    plt::annotate(std::format(R"($\Omega_{{\rm mid}} - \Omega_\pm = L/4 = {:g}$)", omega_mid), 0.02, 0.5 * omega_mid);
    plt::xlim(-0.6, 0.6);
    plt::ylim(-6.0, 9.0);
    plt::xlabel(R"($\mu$)");
    plt::ylabel(R"($\Omega$)");
    plt::title("Uniform branch: the swallowtail");
    plt::legend({{"loc", "lower left"}});
    plt::grid(true);
    save("swallowtail");
  }

  // Figure 3: bifurcation diagram (signed modal amplitude) and gap to the metastable state.
  inline void branches(const utils::Problem& problem, const utils::Results& results) {
    const auto& uniform = results.uniform;
    plt::figure_size(1900, 560);

    subplot(1, 3, 1);
    std::vector<double> zeros(uniform.mu.size(), 0.0);
    plt::plot(uniform.mu, zeros, {{"color", ink}, {"linewidth", "2"}, {"label", "uniform"}});
    std::vector<double> bx, by;
    for (const auto& e : uniform.bifurcations) {
      bx.push_back(e.mu);
      by.push_back(0.0);
    }
    plt::plot(
        bx,
        by,
        {{"color", ink}, {"marker", "o"}, {"markersize", "6"}, {"linestyle", "None"}, {"label", "bifurcation points"}}
    );
    for (const auto& b : results.arms) {
      const auto& color = branch_colors[static_cast<std::size_t>(b.mode - 1)];
      bool first = b.sign > 0;
      for (const auto& r : runs(b.mu, b.modal, b.index)) {
        std::map<std::string, std::string>
            keywords{{"color", color}, {"linewidth", "2"}, {"linestyle", index_style(r.index)}};
        if (first) {
          keywords["label"] = std::format(R"($n = {}$, $a_{}$, both arms)", b.mode, b.mode);
          first = false;
        }
        plt::plot(r.x, r.y, keywords);
      }
    }
    // Legend proxies for the line styles.
    for (int k = 1; k <= 3; ++k) {
      const double nan = std::numeric_limits<double>::quiet_NaN();
      plt::plot(
          std::vector<double>{nan},
          std::vector<double>{nan},
          {{"color", muted},
           {"linewidth", "2"},
           {"linestyle", index_style(k)},
           {"label", std::format(R"($n_- = {}$)", k)}}
      );
    }
    plt::xlim(-0.45, 0.45);
    plt::ylim(-2.5, 1.4);
    plt::yticks(std::vector<double>{-1.0, -0.5, 0.0, 0.5, 1.0});
    plt::xlabel(R"($\mu$)");
    plt::ylabel(R"($a_n$)");
    plt::title("Branches with n interfaces: both arms");
    plt::legend({{"loc", "lower left"}, {"fontsize", "small"}});
    plt::grid(true);

    // Gap to the metastable uniform state, against mu (panel 2) and N (panel 3).
    const double mu_f = utils::exact::fold_chemical_potential();
    auto gap = [&](const std::vector<double>& axis,
                   const std::vector<double>& mu,
                   const std::vector<double>& omega,
                   const std::vector<int>& index,
                   const std::string& color,
                   const std::string& label) {
      std::vector<double> x, y;
      std::vector<int> indices;
      for (std::size_t k = 0; k < mu.size(); ++k) {
        if (std::abs(mu[k]) >= mu_f)
          continue;
        const double rho = utils::metastable_density(mu[k]);
        const double omega_meta = problem.length * (0.25 * std::pow(rho * rho - 1.0, 2) - mu[k] * rho);
        x.push_back(axis[k]);
        y.push_back(omega[k] - omega_meta);
        indices.push_back(index[k]);
      }
      bool first = true;
      for (const auto& r : runs(x, y, indices)) {
        std::map<std::string, std::string>
            keywords{{"color", color}, {"linewidth", "2"}, {"linestyle", index_style(r.index)}};
        if (first) {
          keywords["label"] = label;
          first = false;
        }
        plt::plot(r.x, r.y, keywords);
      }
    };

    // Middle arc of the uniform branch: between the two folds.
    std::vector<double> middle_mu, middle_mass, middle_omega;
    std::vector<int> middle_index;
    for (std::size_t k = 0; k < uniform.mu.size(); ++k) {
      if (std::abs(uniform.mass[k] / problem.length) < utils::exact::fold_density()) {
        middle_mu.push_back(uniform.mu[k]);
        middle_mass.push_back(uniform.mass[k]);
        middle_omega.push_back(uniform.omega[k]);
        middle_index.push_back(uniform.index[k]);
      }
    }

    const double sigma = utils::exact::surface_tension(problem.kappa);
    for (int panel : {2, 3}) {
      const bool against_mass = panel == 3;
      subplot(1, 3, panel);
      gap(against_mass ? middle_mass : middle_mu, middle_mu, middle_omega, middle_index, ink, "uniform, middle arc");
      // The two arms have the same Omega and N, so one arm per n is drawn.
      for (const auto& b : results.arms) {
        if (b.sign > 0) {
          const auto& color = branch_colors[static_cast<std::size_t>(b.mode - 1)];
          gap(against_mass ? b.mass : b.mu, b.mu, b.omega, b.index, color, std::format("n = {}", b.mode));
        }
      }
      for (int n = 1; n <= 3; ++n)
        plt::axhline(n * sigma, 0.0, 1.0, {{"color", muted}, {"linewidth", "0.8"}, {"linestyle", ":"}});
      plt::yticks(
          std::vector<double>{0.0, sigma, 2.0 * sigma, 3.0 * sigma, 4.0, 5.0, 6.0, 7.0},
          std::vector<std::string>{"0", R"($\sigma$)", R"($2\sigma$)", R"($3\sigma$)", "4", "5", "6", "7"}
      );
      if (against_mass) {
        plt::xlim(-0.85 * problem.length, 0.85 * problem.length);
        plt::xlabel(R"($N$)");
        plt::title(R"(The same gap against $N$: the plateau at $\mu \approx 0$)");
      } else {
        plt::xlim(-0.4, 0.4);
        plt::xlabel(R"($\mu$)");
        plt::title("Gap to the metastable uniform state");
      }
      plt::ylim(0.0, 7.5);
      plt::ylabel(R"($\Omega - \Omega_{\rm meta}$)");
      plt::legend({{"loc", "upper left"}, {"fontsize", "small"}});
      plt::grid(true);
    }
    save("branches");
  }

  // Figure: zoom on the n = 1 and n = 2 pitchforks, a_n against mu - mu_n,
  // with a log-log inset for the exponent and the profiles of both arms.
  inline void pitchfork_zoom(const utils::Problem& problem, const utils::Results& results) {
    plt::figure_size(1400, 640);
    const auto x_nodes = arma::conv_to<std::vector<double>>::from(problem.positions());
    std::vector<std::vector<arma::vec>> profiles_shown;
    for (std::size_t j = 0; j < results.pitchfork.size(); ++j) {
      subplot(1, static_cast<int>(results.pitchfork.size()), static_cast<int>(j + 1));
      const auto& samples = results.pitchfork[j];
      const int n = results.pitchfork_points[j].mode;
      const double mu_n = results.pitchfork_points[j].mu;
      const auto& color = branch_colors[static_cast<std::size_t>(n - 1)];
      const auto fit = utils::fit_power_law(samples, 1e-4, 1e-3);
      const auto fit_all = utils::fit_power_law(samples, 1e-4, 1e-1);

      std::vector<double> dmu, amplitude;
      double span = 0.0;
      double a_max = 0.0;
      for (const auto& s : samples) {
        dmu.push_back(s.dmu);
        amplitude.push_back(s.amplitude);
        span = std::max(span, std::abs(s.dmu));
        a_max = std::max(a_max, std::abs(s.amplitude));
      }
      const double side = samples.front().dmu > 0.0 ? 1.0 : -1.0;

      // Main panel: uniform branch, traced arms, samples and the fitted law.
      plt::plot({-1.2 * span, 1.6 * span}, {0.0, 0.0}, {{"color", ink}, {"linewidth", "2"}, {"label", "uniform"}});
      bool labelled = false;
      for (const auto& b : results.arms) {
        if (b.mode != n)
          continue;
        std::vector<double> bx, by;
        for (std::size_t k = 0; k < b.mu.size() / 2; ++k) {
          if (std::abs(b.modal[k]) <= a_max) {
            bx.push_back(b.mu[k] - mu_n);
            by.push_back(b.modal[k]);
          }
        }
        std::map<std::string, std::string> keywords{{"color", tint(color, 0.5)}, {"linewidth", "4"}};
        if (!labelled) {
          keywords["label"] = "traced arms";
          labelled = true;
        }
        plt::plot(bx, by, keywords);
      }
      plt::plot(
          dmu,
          amplitude,
          {{"color", color},
           {"marker", "o"},
           {"markersize", "4"},
           {"linestyle", "None"},
           {"label", std::format(R"(solved at fixed $a_{}$)", n)}}
      );
      std::vector<double> cx, cy_plus, cy_minus;
      for (double t : arma::linspace(0.0, 1.0, 200)) {
        const double d = side * span * t;
        cx.push_back(d);
        cy_plus.push_back(fit.prefactor * std::sqrt(std::abs(d)));
        cy_minus.push_back(-fit.prefactor * std::sqrt(std::abs(d)));
      }
      plt::plot(
          cx,
          cy_plus,
          {{"color", ink},
           {"linewidth", "1"},
           {"linestyle", "--"},
           {"label",
            std::
                format(R"($\pm {:.3f}\,|\mu - \mu_{}|^{{1/2}}$, fit on $|a_{}| \leq 10^{{-3}}$)", fit.prefactor, n, n)}}
      );
      plt::plot(cx, cy_minus, {{"color", ink}, {"linewidth", "1"}, {"linestyle", "--"}});
      plt::plot({0.0}, {0.0}, {{"color", ink}, {"marker", "o"}, {"markersize", "7"}, {"linestyle", "None"}});

      // The two profiles shown in the insets, marked on their arms.
      std::vector<arma::vec> shown;
      for (int sign : {+1, -1}) {
        const auto& b = results.arm(n, sign);
        std::size_t pick = 1;
        for (std::size_t k = 1; k < b.mu.size() / 2; ++k) {
          if (std::abs(std::abs(b.modal[k]) - 0.8 * a_max) < std::abs(std::abs(b.modal[pick]) - 0.8 * a_max))
            pick = k;
        }
        plt::plot(
            {b.mu[pick] - mu_n},
            {b.modal[pick]},
            {{"color", ink}, {"marker", "s"}, {"markersize", "7"}, {"linestyle", "None"}}
        );
        plt::annotate(sign > 0 ? "A" : "B", b.mu[pick] - mu_n + 0.04 * span, b.modal[pick]);
        shown.push_back(b.curve[pick].x);
      }

      plt::xlim(side > 0 ? -0.2 * span : -1.15 * span, side > 0 ? 1.15 * span : 1.6 * span);
      plt::ylim(-1.3 * a_max, 1.3 * a_max);
      plt::xlabel(std::format(R"($\mu - \mu_{}$)", n));
      plt::ylabel(std::format(R"($a_{}$)", n));
      plt::title(
          std::format(
              R"($n = {}$, $\mu_{} = {:.6f}$: $\beta = {:.4f}$ on $|a| \leq 10^{{-3}}$, ${:.3f}$ on $|a| \leq 10^{{-1}}$)",
              n,
              n,
              mu_n,
              fit.exponent,
              fit_all.exponent
          )
      );
      plt::grid(true);
      legend_below(2);

      profiles_shown.push_back(std::move(shown));
    }
    plt::tight_layout();

    // Insets, placed once the panels have their final positions.
    const int columns = static_cast<int>(results.pitchfork.size());
    for (std::size_t j = 0; j < results.pitchfork.size(); ++j) {
      const int panel = static_cast<int>(j + 1);
      const auto& samples = results.pitchfork[j];
      const int n = results.pitchfork_points[j].mode;
      const auto& color = branch_colors[static_cast<std::size_t>(n - 1)];
      const auto fit = utils::fit_power_law(samples, 1e-4, 1e-3);

      // Profiles of the marked points: mirror images under the arm symmetry.
      const std::array<double, 2> inset_y{0.70, 0.06};
      for (std::size_t s = 0; s < 2; ++s) {
        inset(1, columns, panel, 0.62, inset_y[s], 0.35, 0.24);
        plt::plot(
            x_nodes,
            arma::conv_to<std::vector<double>>::from(profiles_shown[j][s]),
            {{"color", color}, {"linewidth", "1.5"}}
        );
        plt::title(s == 0 ? "A: upper arm" : "B: lower arm", {{"fontsize", "x-small"}});
        plt::tick_params({{"labelsize", "x-small"}});
        plt::grid(true);
      }

      // Log-log inset: slope 1/2 at small amplitude, the fit window shaded.
      inset(1, columns, panel, 0.69, 0.41, 0.28, 0.20);
      std::vector<double> ax_abs, ay_abs;
      for (const auto& sample : samples) {
        ax_abs.push_back(std::abs(sample.dmu));
        ay_abs.push_back(std::abs(sample.amplitude));
      }
      const double x_lo = *std::ranges::min_element(ax_abs);
      const double x_hi = *std::ranges::max_element(ax_abs);
      plt::fill_between(
          std::vector<double>{x_lo, x_hi},
          std::vector<double>{1e-4, 1e-4},
          std::vector<double>{1e-3, 1e-3},
          {{"color", tint(color, 0.8)}}
      );
      plt::plot(ax_abs, ay_abs, {{"color", color}, {"marker", "o"}, {"markersize", "2"}, {"linestyle", "None"}});
      plt::plot(
          std::vector<double>{x_lo, x_hi},
          std::vector<double>{fit.prefactor * std::sqrt(x_lo), fit.prefactor * std::sqrt(x_hi)},
          {{"color", ink}, {"linewidth", "1"}, {"linestyle", "--"}}
      );
      log_axes();
      plt::title("slope 1/2 (dashed), fit window shaded", {{"fontsize", "x-small"}});
      plt::tick_params({{"labelsize", "x-small"}});
    }
    save("pitchfork_zoom", false);
  }

  // Figure 4: profiles along the plus arm of each branch.
  inline void profiles(const utils::Problem& problem, const utils::Results& results) {
    std::vector<utils::Branch> branches;
    for (const auto& b : results.arms) {
      if (b.sign > 0)
        branches.push_back(b);
    }
    plt::figure_size(1400, 460);
    auto x = arma::conv_to<std::vector<double>>::from(problem.positions());
    const double w = std::sqrt(2.0 * problem.kappa);
    for (std::size_t b = 0; b < branches.size(); ++b) {
      subplot(1, static_cast<int>(branches.size()), static_cast<int>(b + 1));
      const auto& branch = branches[b];
      auto peak = static_cast<std::size_t>(std::ranges::max_element(branch.amplitude) - branch.amplitude.begin());
      const double a_max = branch.amplitude[peak];
      // Points on the mu > 0 half at a fraction of the peak amplitude.
      for (double frac : {0.1, 0.4, 0.7, 1.0}) {
        std::size_t pick = peak;
        for (std::size_t k = 0; k <= peak; ++k) {
          if (branch.amplitude[k] >= frac * a_max) {
            pick = k;
            break;
          }
        }
        auto y = arma::conv_to<std::vector<double>>::from(branch.curve[pick].x);
        plt::plot(
            x,
            y,
            {{"color", tint(branch_colors[b], 0.7 * (1.0 - frac))},
             {"linewidth", "2"},
             {"label", std::format(R"($\mu = {:+.3f}$)", std::abs(branch.mu[pick]) < 5e-4 ? 0.0 : branch.mu[pick])}}
        );
      }
      // Tanh chain with interfaces at (2j - 1) L / 2n, signed to match the peak profile.
      const int n = static_cast<int>(b + 1);
      arma::vec chain(problem.nodes, arma::fill::ones);
      arma::vec xs = problem.positions();
      for (int j = 1; j <= n; ++j)
        chain %= arma::tanh((xs - (2.0 * j - 1.0) * problem.length / (2.0 * n)) / w);
      if ((chain(0) > 0.0) != (branch.curve[peak].x(0) > 0.0))
        chain = -chain;
      plt::plot(
          x,
          arma::conv_to<std::vector<double>>::from(chain),
          {{"color", ink}, {"linewidth", "1"}, {"linestyle", "--"}, {"label", "tanh chain"}}
      );
      plt::ylim(-1.2, 2.3);
      plt::yticks(std::vector<double>{-1.0, -0.5, 0.0, 0.5, 1.0});
      plt::xlabel(R"($x$)");
      if (b == 0)
        plt::ylabel(R"($\rho(x)$)");
      plt::title(std::format("Branch n = {}", branch.mode));
      plt::legend({{"loc", "upper center"}, {"fontsize", "x-small"}});
      plt::grid(true);
    }
    save("profiles");
  }

  // Figure 5: the same states parametrised by N (fixed mass).
  inline void canonical(
      const utils::Problem& problem,
      const utils::Branch& uniform,
      const utils::Branch& kink,
      const std::vector<utils::CanonicalBranch>& traced
  ) {
    plt::figure_size(800, 560);
    // Uniform branch: index at fixed N.
    std::vector<int> u_index;
    for (const auto& q : uniform.curve)
      u_index.push_back(problem.constrained_index(q.x));
    std::vector<double> u_n = uniform.mass;
    stability_line(u_n, uniform.mu, u_index, ink, true);
    plt::plot(
        kink.mass,
        kink.mu,
        {{"color", tint(branch_colors[0], 0.7)}, {"linewidth", "7"}, {"label", R"($n = 1$, traced in $\mu$)"}}
    );
    bool first = true;
    for (const auto& c : traced) {
      std::vector<int> unstable(c.index.size());
      for (std::size_t k = 0; k < c.index.size(); ++k)
        unstable[k] = c.index[k] > 0 ? 1 : 0;
      for (const auto& r : runs(c.mass, c.mu, unstable)) {
        std::map<std::string, std::string>
            keywords{{"color", branch_colors[0]}, {"linewidth", "2"}, {"linestyle", r.index == 0 ? "-" : "--"}};
        if (first && r.index == 0) {
          keywords["label"] = R"($n = 1$, traced in $N$)";
          first = false;
        }
        plt::plot(r.x, r.y, keywords);
      }
    }
    plt::axhline(0.0, 0.0, 1.0, {{"color", muted}, {"linewidth", "0.6"}});
    plt::xlim(-1.3 * problem.length, 1.3 * problem.length);
    plt::ylim(-0.7, 0.7);
    plt::xlabel(R"($N$)");
    plt::ylabel(R"($\mu$)");
    plt::title("Fixed mass: index at fixed $N$ by line style");
    plt::legend({{"loc", "upper left"}, {"fontsize", "small"}});
    plt::grid(true);
    save("canonical");
  }

} // namespace plot

#endif
