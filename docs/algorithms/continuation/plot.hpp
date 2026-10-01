#pragma once

#include "branches.hpp"

#include <array>
#include <cmath>
#include <dft/plotting/matplotlib.hpp>
#include <format>
#include <map>
#include <numbers>
#include <string>
#include <vector>

#ifdef DFT_HAS_MATPLOTLIB
#include "matplotlibcpp.h"

namespace plot {

  namespace plt = matplotlibcpp;
  namespace py = dft::plotting::detail;

  // Figure sizes in inches, at journal widths: a single plot fills one
  // column, multipanel figures fill two.
  struct Size {
    double width;
    double height;
  };

  inline constexpr Size ONE_COLUMN{3.4, 3.0};
  inline constexpr Size TWO_COLUMN{7.0, 3.0};
  inline constexpr Size TWO_COLUMN_TALL{7.0, 5.2};

  inline const std::string ink = "#3d3d3a";
  inline const std::string muted = "#8a8980";
  inline const std::vector<std::string> branch_colors =
      {"#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"};

  // Font sizes for print at the figure sizes above, and 300 dpi PNGs.
  inline void style() {
    plt::rcparams({
        {"font.size", "8"},
        {"axes.titlesize", "8"},
        {"axes.labelsize", "8"},
        {"legend.fontsize", "7"},
        {"xtick.labelsize", "7"},
        {"ytick.labelsize", "7"},
        {"lines.linewidth", "1.2"},
        {"lines.markersize", "4"},
        {"savefig.dpi", "300"},
    });
  }

  // matplotlibcpp's figure_size takes pixels at 100 dpi.
  inline void figure(Size size) {
    plt::figure_size(static_cast<std::size_t>(size.width * 100.0), static_cast<std::size_t>(size.height * 100.0));
  }

  // Blend a #rrggbb colour towards white: t = 0 keeps it, t = 1 gives white.
  inline auto tint(const std::string& hex, double t) -> std::string {
    auto channel = [&](int k) {
      int c = std::stoi(hex.substr(1 + 2 * static_cast<std::size_t>(k), 2), nullptr, 16);
      return static_cast<int>(std::lround(c + t * (255 - c)));
    };
    return std::format("#{:02x}{:02x}{:02x}", channel(0), channel(1), channel(2));
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
      const std::string& stable_label,
      const std::string& unstable_label
  ) {
    std::vector<int> unstable(index.size());
    for (std::size_t k = 0; k < index.size(); ++k)
      unstable[k] = index[k] > 0 ? 1 : 0;
    bool stable_labelled = stable_label.empty();
    bool unstable_labelled = unstable_label.empty();
    for (const auto& r : runs(x, y, unstable)) {
      std::map<std::string, std::string> keywords{{"color", color}, {"linestyle", r.index == 0 ? "-" : "--"}};
      if (r.index == 0 && !stable_labelled) {
        keywords["label"] = stable_label;
        stable_labelled = true;
      }
      if (r.index == 1 && !unstable_labelled) {
        keywords["label"] = unstable_label;
        unstable_labelled = true;
      }
      plt::plot(r.x, r.y, keywords);
    }
  }

  // Axes operations that matplotlibcpp does not wrap, or wraps with a bug
  // (subplot passes floats, which recent matplotlib rejects, and
  // subplot2grid releases its shape and location tuples twice), through the
  // library's Python helpers.

  inline void call(PyObject* callable, PyObject* args, PyObject* kwargs, const char* what) {
    PyObject* result = PyObject_Call(callable, args, kwargs);
    if (!result) {
      PyErr_Print();
      throw std::runtime_error(std::string("matplotlib call failed: ") + what);
    }
    Py_DECREF(result);
  }

  // Panel spanning rows [row, row + rowspan) and columns [column, column + colspan)
  // of a rows x columns grid, made the current axes.
  inline void panel(long rows, long columns, long row, long column, long rowspan = 1, long colspan = 1) {
    PyObject* pyplot = py::import_module("matplotlib.pyplot");
    PyObject* subplot2grid = py::get_attr(pyplot, "subplot2grid");
    PyObject* args = Py_BuildValue("((ll)(ll))", rows, columns, row, column);
    PyObject* kwargs = PyDict_New();
    py::set_dict_long(kwargs, "rowspan", rowspan);
    py::set_dict_long(kwargs, "colspan", colspan);
    call(subplot2grid, args, kwargs, "subplot2grid");
    Py_DECREF(kwargs);
    Py_DECREF(args);
    Py_DECREF(subplot2grid);
    Py_DECREF(pyplot);
  }

  // Position (x0, y0, width, height) of the current axes in figure coordinates.
  using Bounds = std::array<double, 4>;

  inline auto current_bounds() -> Bounds {
    PyObject* pyplot = py::import_module("matplotlib.pyplot");
    PyObject* ax = py::current_axes(pyplot);
    PyObject* get_position = py::get_attr(ax, "get_position");
    PyObject* box = PyObject_CallObject(get_position, nullptr);
    auto field = [box](const char* name) {
      PyObject* value = py::get_attr(box, name);
      const double out = PyFloat_AsDouble(value);
      Py_DECREF(value);
      return out;
    };
    Bounds bounds{field("x0"), field("y0"), field("width"), field("height")};
    Py_DECREF(box);
    Py_DECREF(get_position);
    Py_DECREF(ax);
    Py_DECREF(pyplot);
    return bounds;
  }

  // Inset at (x0, y0, width, height) in the axes coordinates of a panel with
  // the given bounds, added as a figure axes and made current. The panel
  // positions must be final: figures with insets fix their margins with
  // subplots_adjust and are saved without tight_layout.
  inline void inset(const Bounds& parent, double x0, double y0, double width, double height) {
    PyObject* pyplot = py::import_module("matplotlib.pyplot");
    Py_DECREF(
        py::add_axes_rect(
            pyplot,
            parent[0] + x0 * parent[2],
            parent[1] + y0 * parent[3],
            width * parent[2],
            height * parent[3]
        )
    );
    Py_DECREF(pyplot);
  }

  // At most `count` major ticks on the x axis of the current axes.
  inline void x_tick_count(long count) {
    PyObject* pyplot = py::import_module("matplotlib.pyplot");
    PyObject* ticker = py::import_module("matplotlib.ticker");
    PyObject* locator_type = py::get_attr(ticker, "MaxNLocator");
    PyObject* locator = PyObject_CallFunction(locator_type, "l", count);
    PyObject* ax = py::current_axes(pyplot);
    PyObject* xaxis = py::get_attr(ax, "xaxis");
    PyObject* set_locator = py::get_attr(xaxis, "set_major_locator");
    PyObject* args = PyTuple_Pack(1, locator);
    call(set_locator, args, nullptr, "set_major_locator");
    Py_DECREF(args);
    Py_DECREF(set_locator);
    Py_DECREF(xaxis);
    Py_DECREF(ax);
    Py_DECREF(locator);
    Py_DECREF(locator_type);
    Py_DECREF(ticker);
    Py_DECREF(pyplot);
  }

  inline void log_axes() {
    PyObject* pyplot = py::import_module("matplotlib.pyplot");
    PyObject* ax = py::current_axes(pyplot);
    py::call_axes_string_method(ax, "set_xscale", "log");
    py::call_axes_string_method(ax, "set_yscale", "log");
    Py_DECREF(ax);
    Py_DECREF(pyplot);
  }

  // Text at (text_x, text_y) with an arrow to (x, y), in data coordinates.
  inline void callout(const std::string& text, double x, double y, double text_x, double text_y) {
    PyObject* pyplot = py::import_module("matplotlib.pyplot");
    PyObject* ax = py::current_axes(pyplot);
    PyObject* annotate = py::get_attr(ax, "annotate");
    PyObject* args = Py_BuildValue("(s)", text.c_str());
    PyObject* kwargs = PyDict_New();
    PyObject* xy = Py_BuildValue("(dd)", x, y);
    PyObject* xytext = Py_BuildValue("(dd)", text_x, text_y);
    PyObject* arrow = PyDict_New();
    py::set_dict_string(arrow, "arrowstyle", "->");
    py::set_dict_string(arrow, "color", muted);
    py::set_dict_double(arrow, "lw", 0.7);
    PyDict_SetItemString(kwargs, "xy", xy);
    PyDict_SetItemString(kwargs, "xytext", xytext);
    PyDict_SetItemString(kwargs, "arrowprops", arrow);
    py::set_dict_string(kwargs, "fontsize", "7");
    call(annotate, args, kwargs, "annotate");
    Py_DECREF(arrow);
    Py_DECREF(xytext);
    Py_DECREF(xy);
    Py_DECREF(kwargs);
    Py_DECREF(args);
    Py_DECREF(annotate);
    Py_DECREF(ax);
    Py_DECREF(pyplot);
  }

  // Text at (x, y) in data coordinates, at the given font size.
  inline void text(const std::string& content, double x, double y, const std::string& size) {
    PyObject* pyplot = py::import_module("matplotlib.pyplot");
    PyObject* ax = py::current_axes(pyplot);
    PyObject* text_fn = py::get_attr(ax, "text");
    PyObject* args = Py_BuildValue("(dds)", x, y, content.c_str());
    PyObject* kwargs = PyDict_New();
    py::set_dict_string(kwargs, "fontsize", size);
    call(text_fn, args, kwargs, "text");
    Py_DECREF(kwargs);
    Py_DECREF(args);
    Py_DECREF(text_fn);
    Py_DECREF(ax);
    Py_DECREF(pyplot);
  }

  // Legend of the current axes entries, centred below all panels.
  inline void figure_legend(long columns) {
    PyObject* pyplot = py::import_module("matplotlib.pyplot");
    PyObject* ax = py::current_axes(pyplot);
    PyObject* handles_labels_fn = py::get_attr(ax, "get_legend_handles_labels");
    PyObject* handles_labels = PyObject_CallObject(handles_labels_fn, nullptr);
    PyObject* gcf = py::get_attr(pyplot, "gcf");
    PyObject* fig = PyObject_CallObject(gcf, nullptr);
    PyObject* legend = py::get_attr(fig, "legend");
    PyObject* kwargs = PyDict_New();
    py::set_dict_string(kwargs, "loc", "lower center");
    py::set_dict_long(kwargs, "ncol", columns);
    PyDict_SetItemString(kwargs, "frameon", Py_False);
    call(legend, handles_labels, kwargs, "figure legend");
    Py_DECREF(kwargs);
    Py_DECREF(legend);
    Py_DECREF(fig);
    Py_DECREF(gcf);
    Py_DECREF(handles_labels);
    Py_DECREF(handles_labels_fn);
    Py_DECREF(ax);
    Py_DECREF(pyplot);
  }

  // Save as PDF and PNG; layout = false keeps the margins set by subplots_adjust.
  inline void save(const std::string& name, bool layout = true) {
    if (layout)
      plt::tight_layout();
    plt::save("exports/" + name + ".pdf");
    plt::save("exports/" + name + ".png");
    plt::clf();
    plt::close();
    std::cout << "Plot saved: exports/" << name << ".png\n";
  }

  // A profile y(x) in a small panel titled by its label, with the details
  // written inside. By default every such panel has the same x range and the
  // y range (-1.15, 1.9), which leaves room above the profile for the
  // details; fit = true scales the y range to the profile instead, for
  // small-amplitude states.
  inline void profile(
      const utils::Problem& problem,
      const arma::vec& y,
      const std::string& color,
      const std::string& label,
      const std::string& details,
      bool fit = false
  ) {
    plt::plot(
        arma::conv_to<std::vector<double>>::from(problem.positions()),
        arma::conv_to<std::vector<double>>::from(y),
        {{"color", color}, {"linewidth", "1.2"}}
    );
    plt::xlim(0.0, problem.length);
    plt::xticks(std::vector<double>{0.0, 0.5 * problem.length, problem.length});
    if (fit) {
      const double pad = 0.15 * (y.max() - y.min());
      plt::ylim(y.min() - pad, y.max() + pad);
    } else {
      plt::ylim(-1.15, 1.9);
      plt::yticks(std::vector<double>{-1.0, 0.0, 1.0});
    }
    plt::tick_params({{"labelsize", "6"}});
    if (!details.empty())
      plt::annotate(details, 0.04 * problem.length, 1.3);
    plt::title(label, {{"fontsize", "7"}});
    plt::grid(true);
  }

  inline auto omega_meta(const utils::Problem& problem, double mu) -> double {
    const double rho = utils::exact::metastable_density(mu);
    return problem.length * (0.25 * std::pow(rho * rho - 1.0, 2) - mu * rho);
  }

  inline auto widest(const utils::Branch& branch) -> std::size_t {
    return static_cast<std::size_t>(
        std::ranges::max_element(branch.modal, {}, [](double a) { return std::abs(a); }) - branch.modal.begin()
    );
  }

  inline auto color_of(int mode) -> const std::string& {
    return branch_colors[static_cast<std::size_t>(mode - 1)];
  }

  // A marked state: a point on the curve and its letter next to it.
  inline void mark(double x, double y, const std::string& letter, double dx, double dy) {
    plt::plot({x}, {y}, {{"color", ink}, {"marker", "o"}, {"linestyle", "None"}});
    plt::annotate(letter, x + dx, y + dy);
  }

  // The uniform branch on the a_n = 0 axis: every uniform state has a_n = 0,
  // so for |mu| < mu_f three of them share the axis. The stable arcs are drawn
  // solid in grey, the unstable middle arc (where the pitchforks sit) dashed.
  inline void uniform_axis(const utils::Branch& uniform, bool label) {
    std::vector<double> zeros(uniform.mu.size(), 0.0);
    std::vector<int> unstable(uniform.index.size());
    for (std::size_t k = 0; k < uniform.index.size(); ++k)
      unstable[k] = uniform.index[k] > 0 ? 1 : 0;
    bool stable_labelled = !label;
    bool unstable_labelled = !label;
    for (int pass : {0, 1}) {
      for (const auto& r : runs(uniform.mu, zeros, unstable)) {
        if (r.index != pass)
          continue;
        std::map<std::string, std::string>
            keywords{{"color", pass == 0 ? tint(ink, 0.55) : ink}, {"linestyle", pass == 0 ? "-" : "--"}};
        if (pass == 0 && !stable_labelled) {
          keywords["label"] = "uniform, stable";
          stable_labelled = true;
        }
        if (pass == 1 && !unstable_labelled) {
          keywords["label"] = "uniform, unstable";
          unstable_labelled = true;
        }
        plt::plot(r.x, r.y, keywords);
      }
    }
  }

  // N against mu on the uniform branch.
  inline void s_curve(const utils::Problem& problem, const utils::Branch& uniform) {
    figure(ONE_COLUMN);
    stability_line(
        uniform.mu,
        uniform.mass,
        uniform.index,
        branch_colors[0],
        R"(stable, $n_- = 0$)",
        R"(unstable, $n_- \geq 1$)"
    );
    std::vector<double> fx, fy;
    for (const auto& f : uniform.folds) {
      fx.push_back(f.mu);
      fy.push_back(problem.mass(f.point.x));
    }
    plt::plot(fx, fy, {{"color", ink}, {"marker", "o"}, {"linestyle", "None"}, {"label", "folds"}});
    for (const auto& f : uniform.folds) {
      const double n = problem.mass(f.point.x);
      plt::annotate(
          std::format(R"($\mu = {:+.4f}$)", f.mu),
          f.mu > 0 ? f.mu + 0.03 : f.mu - 0.36,
          n + (n > 0 ? 1.5 : -3.0)
      );
    }
    plt::axhline(0.0, 0.0, 1.0, {{"color", muted}, {"linewidth", "0.5"}});
    plt::axvline(0.0, 0.0, 1.0, {{"color", muted}, {"linewidth", "0.5"}});
    plt::xlim(-0.8, 0.8);
    plt::xlabel(R"($\mu$)");
    plt::ylabel(R"($N = \int_0^L \rho\,dx$)");
    plt::legend({{"loc", "center right"}, {"fontsize", "6"}});
    plt::grid(true);
    save("s_curve");
  }

  // Omega against mu: the uniform swallowtail with the interface branches
  // (dashed: unstable at fixed mu), and the profiles of the marked states.
  inline void swallowtail(const utils::Problem& problem, const utils::Results& results) {
    const auto& uniform = results.uniform;
    figure(TWO_COLUMN);
    plt::subplots_adjust(
        {{"left", 0.07}, {"right", 0.98}, {"bottom", 0.14}, {"top", 0.93}, {"wspace", 0.35}, {"hspace", 0.45}}
    );
    panel(2, 4, 0, 0, 2, 2);
    stability_line(uniform.mu, uniform.omega, uniform.index, ink, "uniform, stable", "uniform, unstable (flat)");
    for (const auto& b : results.arms) {
      if (b.sign > 0)
        plt::plot(
            b.mu,
            b.omega,
            {{"color", color_of(b.mode)},
             {"linestyle", "--"},
             {"label", std::format("{} interface{}, unstable", b.mode, b.mode > 1 ? "s" : "")}}
        );
    }
    const arma::vec middle(problem.nodes, arma::fill::zeros);
    mark(0.0, problem.grand_potential(middle, 0.0), "A", 0.02, 0.25);
    const std::array<std::string, 3> letters{"B", "C", "D"};
    for (int n = 1; n <= 3; ++n) {
      const auto& b = results.arm(n, +1);
      const std::size_t k = widest(b);
      // The three points sit at mu = 0 inside the nested curves: letters in
      // the free wedge on the right, with arrows.
      plt::plot({b.mu[k]}, {b.omega[k]}, {{"color", ink}, {"marker", "o"}, {"linestyle", "None"}});
      callout(letters[static_cast<std::size_t>(n - 1)], b.mu[k], b.omega[k], 0.45, 1.5 * n - 2.5);
    }
    plt::xlim(-1.0, 0.6);
    plt::ylim(-6.0, 9.0);
    plt::xlabel(R"($\mu$)");
    plt::ylabel(R"($\Omega$)");
    plt::legend({{"loc", "center left"}, {"fontsize", "6"}});
    plt::grid(true);

    // Profiles of the marked states at mu = 0.
    const std::array<std::array<long, 2>, 4> cells{{{0, 2}, {0, 3}, {1, 2}, {1, 3}}};
    panel(2, 4, cells[0][0], cells[0][1]);
    profile(
        problem,
        middle,
        ink,
        "(A)",
        std::format(R"(uniform, $\Omega = {:.2f}$)", problem.grand_potential(middle, 0.0))
    );
    for (int n = 1; n <= 3; ++n) {
      const auto& b = results.arm(n, +1);
      const std::size_t k = widest(b);
      panel(2, 4, cells[static_cast<std::size_t>(n)][0], cells[static_cast<std::size_t>(n)][1]);
      profile(
          problem,
          b.curve[k].x,
          color_of(n),
          std::format("({})", letters[static_cast<std::size_t>(n - 1)]),
          std::format(R"({} interface{}, $\Omega = {:.2f}$)", n, n > 1 ? "s" : "", b.omega[k])
      );
    }
    save("swallowtail", false);
  }

  // Bifurcation diagram (signed modal amplitude) over the gap to the
  // metastable uniform state, against mu and against N.
  inline void branches(const utils::Problem& problem, const utils::Results& results) {
    const auto& uniform = results.uniform;
    figure(TWO_COLUMN_TALL);
    plt::subplots_adjust(
        {{"left", 0.08}, {"right", 0.98}, {"bottom", 0.16}, {"top", 0.97}, {"wspace", 0.22}, {"hspace", 0.45}}
    );

    // Top: a_n against mu, both arms (dashed: unstable at fixed mu, n_- = n),
    // and the bifurcation points numbered by n.
    panel(3, 2, 0, 0, 2, 2);
    uniform_axis(uniform, true);
    for (const auto& b : results.arms) {
      std::map<std::string, std::string> keywords{{"color", color_of(b.mode)}, {"linestyle", "--"}};
      if (b.sign > 0)
        keywords["label"] = std::format(R"($n = {}$ interface{}, both arms)", b.mode, b.mode > 1 ? "s" : "");
      plt::plot(b.mu, b.modal, keywords);
    }
    std::vector<double> bx, by;
    for (const auto& e : uniform.bifurcations) {
      bx.push_back(e.mu);
      by.push_back(0.0);
    }
    plt::plot(bx, by, {{"color", ink}, {"marker", "o"}, {"linestyle", "None"}, {"label", R"(bifurcation point $n$)"}});
    for (const auto& e : uniform.bifurcations) {
      if (e.mode >= 5)
        plt::annotate(std::to_string(e.mode), e.mu - 0.006, -0.17);
    }
    // The points n = 1 to 4 lie within 0.03 of each fold: boxed here, zoomed in the insets.
    for (double side : {-1.0, 1.0}) {
      const double a = side * 0.353;
      const double b = side * 0.388;
      plt::plot({a, b, b, a, a}, {-0.12, -0.12, 0.12, 0.12, -0.12}, {{"color", muted}, {"linewidth", "0.7"}});
    }
    plt::annotate(R"($n = 1..4$)", 0.345, -0.3);
    plt::annotate(R"($n = 1..4$)", -0.395, -0.3);
    plt::xlim(-0.45, 0.45);
    plt::ylim(-1.35, 1.35);
    plt::xlabel(R"($\mu$)");
    plt::ylabel(R"($a_n$)");
    plt::grid(true);
    figure_legend(3);

    // Zooms on the points n = 1 to 4 on both sides, in the space between the arms.
    const Bounds top = current_bounds();
    for (double side : {-1.0, 1.0}) {
      inset(top, side > 0 ? 0.64 : 0.14, 0.66, 0.22, 0.24);
      uniform_axis(uniform, false);
      for (const auto& b : results.arms)
        plt::plot(b.mu, b.modal, {{"color", color_of(b.mode)}, {"linestyle", "--"}});
      for (const auto& e : uniform.bifurcations) {
        if (side * e.mu > 0.35 && e.mode <= 4) {
          plt::plot({e.mu}, {0.0}, {{"color", ink}, {"marker", "o"}, {"linestyle", "None"}});
          plt::annotate(std::to_string(e.mode), e.mu - 0.0012, e.mode % 2 == 0 ? 0.03 : -0.07);
        }
      }
      plt::xlim(side > 0 ? 0.353 : -0.388, side > 0 ? 0.388 : -0.353);
      plt::ylim(-0.12, 0.12);
      plt::xticks(side > 0 ? std::vector<double>{0.36, 0.37, 0.38} : std::vector<double>{-0.38, -0.37, -0.36});
      plt::yticks(std::vector<double>{-0.1, 0.0, 0.1});
      plt::tick_params({{"labelsize", "6"}});
    }

    // Bottom: gap to the metastable uniform state at the same mu, against mu
    // and against N. All the curves are unstable at fixed mu; the uniform
    // middle arc is dashed as on the axis above.
    const double mu_f = utils::exact::fold_chemical_potential();
    auto gap = [&](const std::vector<double>& axis,
                   const std::vector<double>& mu,
                   const std::vector<double>& omega,
                   const std::string& color,
                   const std::string& linestyle) {
      std::vector<double> x, y;
      for (std::size_t k = 0; k < mu.size(); ++k) {
        if (std::abs(mu[k]) < mu_f) {
          x.push_back(axis[k]);
          y.push_back(omega[k] - omega_meta(problem, mu[k]));
        }
      }
      plt::plot(x, y, {{"color", color}, {"linestyle", linestyle}});
    };
    std::vector<double> middle_mu, middle_mass, middle_omega;
    for (std::size_t k = 0; k < uniform.mu.size(); ++k) {
      if (std::abs(uniform.mass[k] / problem.length) < utils::exact::fold_density()) {
        middle_mu.push_back(uniform.mu[k]);
        middle_mass.push_back(uniform.mass[k]);
        middle_omega.push_back(uniform.omega[k]);
      }
    }
    const double sigma = utils::exact::surface_tension(problem.kappa);
    for (long column : {0L, 1L}) {
      const bool against_mass = column == 1;
      panel(3, 2, 2, column);
      gap(against_mass ? middle_mass : middle_mu, middle_mu, middle_omega, ink, "--");
      for (const auto& b : results.arms) {
        if (b.sign > 0)
          gap(against_mass ? b.mass : b.mu, b.mu, b.omega, color_of(b.mode), "--");
      }
      for (int n = 1; n <= 3; ++n)
        plt::axhline(n * sigma, 0.0, 1.0, {{"color", muted}, {"linewidth", "0.6"}, {"linestyle", ":"}});
      plt::yticks(
          std::vector<double>{0.0, sigma, 2.0 * sigma, 3.0 * sigma, 4.0, 5.0, 6.0},
          std::vector<std::string>{"0", R"($\sigma$)", R"($2\sigma$)", R"($3\sigma$)", "4", "5", "6"}
      );
      plt::ylim(0.0, 6.5);
      plt::ylabel(R"($\Omega - \Omega_{\rm meta}(\mu)$)");
      if (against_mass) {
        plt::xlim(-0.85 * problem.length, 0.85 * problem.length);
        plt::xlabel(R"($N$)");
      } else {
        plt::xlim(-0.4, 0.4);
        plt::xlabel(R"($\mu$)");
      }
      plt::grid(true);
    }
    save("branches", false);
  }

  // Zoom on the n = 1, 2 and 3 pitchforks: a_n against mu - mu_n with the
  // leading-order and two-term normal forms, and three insets in each panel:
  // the profiles on the two arms and |a_n| against |mu - mu_n| on logarithmic
  // axes, with the fit window shaded.
  inline void pitchfork_zoom(const utils::Problem& problem, const utils::Results& results) {
    figure(TWO_COLUMN_TALL);
    plt::subplots_adjust({{"left", 0.09}, {"right", 0.98}, {"bottom", 0.16}, {"top", 0.95}, {"wspace", 0.4}});
    const long columns = static_cast<long>(results.pitchfork.size());
    for (std::size_t j = 0; j < results.pitchfork.size(); ++j) {
      const auto& samples = results.pitchfork[j];
      const int n = results.pitchfork_points[j].mode;
      const double mu_n = results.pitchfork_points[j].mu;
      const auto& color = color_of(n);
      const auto leading = utils::fit_power_law(samples, 1e-4, 1e-3);
      const auto two_term = utils::fit_normal_form(samples, 0.1);

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

      panel(1, columns, 0, static_cast<long>(j));
      plt::plot(
          {-1.2 * span, 2.0 * span},
          {0.0, 0.0},
          {{"color", ink}, {"linestyle", "--"}, {"label", "uniform, unstable"}}
      );
      for (int sign : {+1, -1}) {
        const auto& b = results.arm(n, sign);
        std::vector<double> bx, by;
        for (std::size_t k = 0; k < b.mu.size() / 2; ++k) {
          if (std::abs(b.modal[k]) <= a_max) {
            bx.push_back(b.mu[k] - mu_n);
            by.push_back(b.modal[k]);
          }
        }
        std::map<std::string, std::string>
            keywords{{"color", tint(color, 0.45)}, {"linewidth", "2"}, {"linestyle", "--"}};
        if (sign > 0 && j == 0)
          keywords["label"] = "traced arms, unstable";
        plt::plot(bx, by, keywords);
      }
      std::map<std::string, std::string>
          sample_keywords{{"color", color}, {"marker", "o"}, {"markersize", "2.5"}, {"linestyle", "None"}};
      if (j == 0)
        sample_keywords["label"] = R"(solved at fixed $a_n$)";
      plt::plot(dmu, amplitude, sample_keywords);
      std::vector<double> a_curve, leading_curve, two_term_curve;
      for (double a : arma::linspace(-a_max, a_max, 201)) {
        a_curve.push_back(a);
        leading_curve.push_back(side * std::pow(std::abs(a) / leading.prefactor, 2.0));
        two_term_curve.push_back(two_term.c2 * a * a + two_term.c4 * a * a * a * a);
      }
      plt::plot(
          leading_curve,
          a_curve,
          {{"color", ink},
           {"linewidth", "0.8"},
           {"linestyle", "-."},
           {"label", R"(leading order, $a \propto |\mu - \mu_n|^{1/2}$)"}}
      );
      plt::plot(
          two_term_curve,
          a_curve,
          {{"color", ink}, {"linewidth", "0.8"}, {"linestyle", ":"}, {"label", R"($\mu - \mu_n = c_2 a^2 + c_4 a^4$)"}}
      );
      std::array<arma::vec, 2> shown;
      for (int sign : {+1, -1}) {
        const auto& b = results.arm(n, sign);
        std::size_t pick = 1;
        for (std::size_t k = 1; k < b.mu.size() / 2; ++k) {
          if (std::abs(std::abs(b.modal[k]) - 0.8 * a_max) < std::abs(std::abs(b.modal[pick]) - 0.8 * a_max))
            pick = k;
        }
        mark(b.mu[pick] - mu_n, b.modal[pick], sign > 0 ? "A" : "B", 0.06 * span, 0.0);
        shown[sign > 0 ? 0 : 1] = b.curve[pick].x;
      }
      plt::xlim(side > 0 ? -0.1 * span : -1.1 * span, side > 0 ? 1.1 * span : 1.4 * span);
      plt::ylim(-1.2 * a_max, 1.2 * a_max);
      plt::xlabel(std::format(R"($\mu - \mu_{}$)", n));
      plt::ylabel(std::format(R"($a_{}$)", n));
      plt::title(std::format(R"($n = {}$, $\mu_{} = {:.6f}$)", n, n, mu_n));
      plt::grid(true);
      x_tick_count(3);
      if (j == 0)
        figure_legend(3);

      // Insets on the right half of the panel: A, the log-log fit, B.
      const Bounds main = current_bounds();
      inset(main, 0.56, 0.74, 0.41, 0.2);
      profile(problem, shown[0], color, "(A)", "", true);
      inset(main, 0.62, 0.4, 0.35, 0.22);
      std::vector<double> abs_dmu, abs_a;
      for (const auto& s : samples) {
        abs_dmu.push_back(std::abs(s.dmu));
        abs_a.push_back(std::abs(s.amplitude));
      }
      const double x_lo = *std::ranges::min_element(abs_dmu);
      const double x_hi = *std::ranges::max_element(abs_dmu);
      plt::fill_between(
          std::vector<double>{x_lo, x_hi},
          std::vector<double>{1e-4, 1e-4},
          std::vector<double>{1e-3, 1e-3},
          {{"color", tint(color, 0.8)}}
      );
      plt::plot(abs_dmu, abs_a, {{"color", color}, {"marker", "o"}, {"markersize", "1.5"}, {"linestyle", "None"}});
      plt::plot(
          std::vector<double>{x_lo, x_hi},
          std::vector<double>{leading.prefactor * std::sqrt(x_lo), leading.prefactor * std::sqrt(x_hi)},
          {{"color", ink}, {"linewidth", "0.8"}, {"linestyle", "-."}}
      );
      log_axes();
      plt::tick_params({{"labelsize", "5"}});
      text(std::format(R"($\beta = {:.4f}$)", leading.exponent), x_lo * 1.5, 2e-2, "6");
      inset(main, 0.56, 0.06, 0.41, 0.2);
      profile(problem, shown[1], color, "(B)", "", true);
    }
    save("pitchfork_zoom", false);
  }

  // The n = 1 branch at fixed mu: a_1 against mu with six lettered states,
  // and their profiles in a row below.
  inline void walk(const utils::Problem& problem, const utils::Results& results) {
    figure(TWO_COLUMN_TALL);
    plt::subplots_adjust(
        {{"left", 0.07}, {"right", 0.98}, {"bottom", 0.07}, {"top", 0.97}, {"wspace", 0.35}, {"hspace", 0.5}}
    );
    panel(3, 6, 0, 0, 2, 6);
    uniform_axis(results.uniform, true);
    const auto& color = color_of(1);
    for (int sign : {+1, -1}) {
      const auto& arm = results.arm(1, sign);
      std::map<std::string, std::string> keywords{{"color", color}, {"linestyle", "--"}};
      if (sign > 0)
        keywords["label"] = R"($n = 1$, both arms, unstable ($n_- = 1$))";
      plt::plot(arm.mu, arm.modal, keywords);
    }
    for (const auto& point : results.fixed_mu_points)
      mark(point.mu, problem.modal_amplitude(point.y, 1), point.letter, 0.008, 0.06);
    plt::xlim(-0.42, 0.42);
    plt::ylim(-1.4, 1.4);
    plt::xlabel(R"($\mu$)");
    plt::ylabel(R"($a_1$)");
    plt::legend({{"loc", "lower left"}});
    plt::grid(true);
    for (std::size_t k = 0; k < results.fixed_mu_points.size(); ++k) {
      const auto& point = results.fixed_mu_points[k];
      panel(3, 6, 2, static_cast<long>(k));
      profile(
          problem,
          point.y,
          color,
          std::format("({})", point.letter),
          std::format("$\\mu = {:+.2f}$\n$\\Delta\\Omega = {:.3f}$", point.mu == 0.0 ? 0.0 : point.mu, point.value)
      );
      plt::xlabel(R"($x$)");
    }
    save("walk", false);
  }

  // The same states at fixed N: mu against N, the index at fixed N by line
  // style (solid stable, dashed unstable), and the profiles of six lettered
  // states in insets.
  inline void canonical(const utils::Problem& problem, const utils::Results& results) {
    const auto& uniform = results.uniform;
    figure(TWO_COLUMN_TALL);
    plt::subplots_adjust({{"left", 0.07}, {"right", 0.98}, {"bottom", 0.17}, {"top", 0.98}});
    panel(1, 1, 0, 0);
    auto fixed_mass_index = [&](const std::vector<utils::CurvePoint>& curve) {
      std::vector<int> index;
      for (const auto& q : curve)
        index.push_back(problem.constrained_index(q.x));
      return index;
    };
    stability_line(
        uniform.mass,
        uniform.mu,
        fixed_mass_index(uniform.curve),
        ink,
        "uniform, stable at fixed N",
        "uniform, unstable at fixed N"
    );
    // The n = 1 branch traced in mu (pale band) and in N (thin line); they
    // coincide. The n = 2 and 3 branches traced in mu. All with their index
    // at fixed N.
    const auto& kink = results.arm(1, +1);
    plt::plot(
        kink.mass,
        kink.mu,
        {{"color", tint(color_of(1), 0.7)}, {"linewidth", "5"}, {"label", R"(1 interface, traced in $\mu$)"}}
    );
    bool first = true;
    for (const auto& c : results.canonical) {
      std::vector<int> unstable(c.index.size());
      for (std::size_t k = 0; k < c.index.size(); ++k)
        unstable[k] = c.index[k] > 0 ? 1 : 0;
      for (const auto& r : runs(c.mass, c.mu, unstable)) {
        std::map<std::string, std::string> keywords{{"color", color_of(1)}, {"linestyle", r.index == 0 ? "-" : "--"}};
        if (first && r.index == 0) {
          keywords["label"] = R"(1 interface, traced in $N$)";
          first = false;
        }
        plt::plot(r.x, r.y, keywords);
      }
    }
    for (int n = 2; n <= 3; ++n) {
      const auto& b = results.arm(n, +1);
      std::vector<utils::CurvePoint> inner(b.curve.begin() + 1, b.curve.end() - 1);
      std::vector<double> mass(b.mass.begin() + 1, b.mass.end() - 1);
      std::vector<double> mu(b.mu.begin() + 1, b.mu.end() - 1);
      stability_line(
          mass,
          mu,
          fixed_mass_index(inner),
          color_of(n),
          std::format("{} interface{}, stable at fixed N", n, n > 1 ? "s" : ""),
          std::format("{} interface{}, unstable at fixed N", n, n > 1 ? "s" : "")
      );
    }
    plt::axhline(0.0, 0.0, 1.0, {{"color", muted}, {"linewidth", "0.5"}});
    for (const auto& point : results.fixed_mass_points) {
      double dx = -1.3;
      double dy = 0.02;
      if (point.letter == "E" || point.letter == "F") {
        dx = 0.5;
        dy = 0.03;
      }
      mark(point.mass, point.mu, point.letter, dx, dy);
    }
    const auto& fold = results.finite_size_folds.front();
    const double rho_1 = utils::exact::bifurcation_density(problem.kappa * problem.laplacian_eigenvalue(1));
    callout(R"(Maxwell plateau, $\mu = 0$)", -6.0, 0.0, -10.5, -0.2);
    callout("finite-size fold", fold.mass, fold.mu, 4.5, 0.08);
    callout(R"(fixed-$N$ saddles)", 15.6, -0.15, 5.5, -0.1);
    callout(
        std::format(R"(uniform unstable at fixed $N$ for $|\bar\rho| < \bar\rho_1 = {:.3f}$)", rho_1),
        -3.0,
        std::pow(-3.0 / problem.length, 3) + 3.0 / problem.length,
        0.5,
        0.24
    );
    plt::xlim(-1.3 * problem.length, 1.3 * problem.length);
    plt::ylim(-0.75, 0.75);
    plt::xlabel(R"($N$)");
    plt::ylabel(R"($\mu$)");
    plt::grid(true);
    figure_legend(3);

    // Insets in the bands above mu = 0.42 and below mu = -0.39, clear of the curves.
    const Bounds main = current_bounds();
    const std::array<std::array<double, 2>, 6> corners{
        {{0.04, 0.77}, {0.33, 0.77}, {0.62, 0.77}, {0.1, 0.03}, {0.33, 0.03}, {0.56, 0.03}}
    };
    for (std::size_t k = 0; k < results.fixed_mass_points.size(); ++k) {
      const auto& point = results.fixed_mass_points[k];
      inset(main, corners[k][0], corners[k][1], 0.19, 0.17);
      profile(
          problem,
          point.y,
          point.letter == "C" || point.letter == "D" || point.letter == "E" ? color_of(1) : ink,
          std::format("({})", point.letter),
          std::format(R"($F = {:.2f}$, $n_- = {}$)", point.value, point.index)
      );
    }
    save("canonical", false);
  }

  // Nested pitchforks at mu = 0: a_n against L / sqrt(kappa), all arms, and
  // the profile of each branch at the largest L / sqrt(kappa).
  inline void nested_pitchforks(const utils::Problem& problem, const utils::Results& results) {
    const auto& nested = results.nested;
    figure(TWO_COLUMN_TALL);
    plt::subplots_adjust(
        {{"left", 0.07}, {"right", 0.98}, {"bottom", 0.07}, {"top", 0.98}, {"wspace", 0.35}, {"hspace", 0.5}}
    );
    panel(3, 6, 0, 0, 2, 6);
    std::vector<double> zeros(nested.lambda.size(), 0.0);
    plt::plot(
        nested.lambda,
        zeros,
        {{"color", ink}, {"linestyle", "--"}, {"label", R"(uniform $\rho = 0$, unstable)"}}
    );
    const std::array<std::string, 6> letters{"A", "B", "C", "D", "E", "F"};
    for (const auto& arm : nested.arms) {
      std::map<std::string, std::string> keywords{{"color", color_of(arm.mode)}, {"linestyle", "--"}};
      if (arm.sign > 0)
        keywords["label"] = std::format(R"($n = {}$, unstable, $n_- = {}$)", arm.mode, arm.mode);
      plt::plot(arm.lambda, arm.modal, keywords);
      if (arm.sign > 0)
        mark(
            arm.lambda.back(),
            arm.modal.back(),
            letters[static_cast<std::size_t>(arm.mode - 1)],
            0.3,
            arm.mode == 1 ? 0.07 : (arm.mode == 2 ? 0.0 : -0.04)
        );
    }
    for (const auto& [n, lambda_n] : nested.bifurcations) {
      plt::plot({lambda_n}, {0.0}, {{"color", ink}, {"marker", "o"}, {"linestyle", "None"}});
      plt::annotate(std::to_string(n), lambda_n - 0.25, -0.2);
    }
    plt::xlim(2.5, results.nested_lambda_max + 1.5);
    plt::ylim(-1.5, 1.5);
    plt::xlabel(R"($L / \sqrt{\kappa}$)");
    plt::ylabel(R"($a_n$)");
    plt::legend({{"loc", "lower left"}, {"fontsize", "6"}});
    plt::grid(true);
    for (const auto& arm : nested.arms) {
      if (arm.sign < 0)
        continue;
      panel(3, 6, 2, static_cast<long>(arm.mode - 1));
      profile(
          problem,
          arm.profiles.back(),
          color_of(arm.mode),
          std::format("({})", letters[static_cast<std::size_t>(arm.mode - 1)]),
          std::format(R"($n = {}$)", arm.mode)
      );
      plt::xlabel(R"($x$)");
    }
    save("nested_pitchforks", false);
  }

  // Number of branches found against L / sqrt(kappa), with the staircase
  // floor((L / pi) / sqrt(kappa)).
  inline void branch_count(const utils::Results& results) {
    const auto& nested = results.nested;
    figure(ONE_COLUMN);
    std::vector<double> x, found, formula;
    for (double lambda : arma::linspace(2.5, results.nested_lambda_max, 2000)) {
      int count = 0;
      for (const auto& [n, lambda_n] : nested.bifurcations)
        count += lambda_n <= lambda ? 1 : 0;
      x.push_back(lambda);
      found.push_back(count);
      formula.push_back(std::floor(lambda / std::numbers::pi));
    }
    plt::plot(
        x,
        formula,
        {{"color", muted}, {"linewidth", "3"}, {"label", R"($\lfloor (L/\pi) / \sqrt{\kappa} \rfloor$)"}}
    );
    plt::plot(x, found, {{"color", branch_colors[0]}, {"label", "branches found"}});
    std::vector<double> bx, by;
    for (const auto& [n, lambda_n] : nested.bifurcations) {
      bx.push_back(lambda_n);
      by.push_back(n);
    }
    plt::plot(
        bx,
        by,
        {{"color", ink}, {"marker", "o"}, {"linestyle", "None"}, {"label", R"(detected $L/\sqrt{\kappa_n}$)"}}
    );
    plt::xlabel(R"($L / \sqrt{\kappa}$)");
    plt::ylabel("number of branches");
    plt::legend({{"loc", "upper left"}, {"fontsize", "6"}});
    plt::grid(true);
    save("branch_count");
  }

  // Landau pitchfork of a uniform order parameter at mu = 0, continued in a,
  // with a proportional to T - T_c.
  inline void landau(const utils::Results& results) {
    const auto& landau = results.landau;
    figure(ONE_COLUMN);
    std::vector<double> stable_a, unstable_a;
    for (double a : landau.a_uniform)
      (a > 0.0 ? stable_a : unstable_a).push_back(a);
    stable_a.push_back(landau.a_critical);
    unstable_a.insert(unstable_a.begin(), landau.a_critical);
    plt::plot(
        stable_a,
        std::vector<double>(stable_a.size(), 0.0),
        {{"color", ink}, {"label", R"($\rho = 0$, stable)"}}
    );
    plt::plot(
        unstable_a,
        std::vector<double>(unstable_a.size(), 0.0),
        {{"color", ink}, {"linestyle", "--"}, {"label", R"($\rho = 0$, unstable)"}}
    );
    for (std::size_t k = 0; k < 2; ++k) {
      std::map<std::string, std::string> keywords{{"color", branch_colors[0]}};
      if (k == 0)
        keywords["label"] = R"($\rho = \pm\sqrt{-a}$, stable)";
      plt::plot(landau.a_arm[k], landau.rho_arm[k], keywords);
    }
    plt::plot(
        {landau.a_critical},
        {0.0},
        {{"color", ink}, {"marker", "o"}, {"linestyle", "None"}, {"label", R"(pitchfork, $a_c = 0$)"}}
    );
    plt::xlabel(R"($a \propto T - T_c$)");
    plt::ylabel(R"($\rho$)");
    plt::xlim(-1.0, 1.0);
    plt::legend({{"loc", "upper right"}, {"fontsize", "6"}});
    plt::grid(true);
    save("landau");
  }

} // namespace plot

#endif
