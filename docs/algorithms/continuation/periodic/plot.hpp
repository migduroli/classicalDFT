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

namespace periodic::plot {

  namespace plt = matplotlibcpp;
  namespace py = dft::plotting::detail;

  struct Size {
    double width;
    double height;
  };

  inline constexpr Size ONE_COLUMN{3.4, 3.0};
  inline constexpr Size TWO_COLUMN{7.0, 3.0};
  inline constexpr Size TWO_COLUMN_TALL{7.0, 5.2};

  inline const std::string ink = "#3d3d3a";
  inline const std::string muted = "#8a8980";
  inline const std::vector<std::string> colors =
      {"#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"};

  inline auto tint(const std::string& hex, double t) -> std::string {
    auto channel = [&](int k) {
      const int value = std::stoi(hex.substr(1 + 2 * static_cast<std::size_t>(k), 2), nullptr, 16);
      return static_cast<int>(std::lround(value + t * (255 - value)));
    };
    return std::format("#{:02x}{:02x}{:02x}", channel(0), channel(1), channel(2));
  }

  inline void style() {
    plt::rcparams({
        {"font.size", "8"},
        {"axes.titlesize", "8"},
        {"axes.labelsize", "8"},
        {"legend.fontsize", "7"},
        {"xtick.labelsize", "7"},
        {"ytick.labelsize", "7"},
        {"lines.linewidth", "1.2"},
        {"savefig.dpi", "300"},
    });
  }

  inline void figure(Size size) {
    plt::figure_size(static_cast<std::size_t>(100.0 * size.width), static_cast<std::size_t>(100.0 * size.height));
  }

  inline void call(PyObject* callable, PyObject* args, PyObject* kwargs, const char* what) {
    PyObject* result = PyObject_Call(callable, args, kwargs);
    if (!result) {
      PyErr_Print();
      throw std::runtime_error(std::string("matplotlib call failed: ") + what);
    }
    Py_DECREF(result);
  }

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

  inline void inset(const Bounds& parent, double x0, double y0, double width, double height) {
    PyObject* pyplot = py::import_module("matplotlib.pyplot");
    Py_DECREF(py::add_axes_rect(
        pyplot,
        parent[0] + x0 * parent[2],
        parent[1] + y0 * parent[3],
        width * parent[2],
        height * parent[3]
    ));
    Py_DECREF(pyplot);
  }

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

  inline void log_axes() {
    PyObject* pyplot = py::import_module("matplotlib.pyplot");
    PyObject* ax = py::current_axes(pyplot);
    py::call_axes_string_method(ax, "set_xscale", "log");
    py::call_axes_string_method(ax, "set_yscale", "log");
    Py_DECREF(ax);
    Py_DECREF(pyplot);
  }

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

  inline void save(const std::string& name, bool layout = true) {
    if (layout)
      plt::tight_layout();
    plt::save("exports/" + name + ".pdf");
    plt::save("exports/" + name + ".png");
    plt::clf();
    plt::close();
    std::cout << "Plot saved: exports/" << name << ".png\n";
  }

  struct Run {
    std::vector<double> x;
    std::vector<double> y;
    int unstable;
  };

  inline auto runs(const std::vector<double>& x, const std::vector<double>& y, const std::vector<int>& index)
      -> std::vector<Run> {
    std::vector<Run> out;
    for (std::size_t k = 0; k < x.size(); ++k) {
      const int unstable = index[k] > 0 ? 1 : 0;
      if (out.empty() || unstable != out.back().unstable) {
        Run run{.unstable = unstable};
        if (!out.empty()) {
          run.x.push_back(out.back().x.back());
          run.y.push_back(out.back().y.back());
        }
        out.push_back(std::move(run));
      }
      out.back().x.push_back(x[k]);
      out.back().y.push_back(y[k]);
    }
    return out;
  }

  inline void stability_line(
      const std::vector<double>& x,
      const std::vector<double>& y,
      const std::vector<int>& index,
      const std::string& color,
      const std::string& stable,
      const std::string& unstable
  ) {
    bool stable_labelled = stable.empty();
    bool unstable_labelled = unstable.empty();
    for (const auto& run : runs(x, y, index)) {
      std::map<std::string, std::string> keywords{
          {"color", color},
          {"linestyle", run.unstable == 0 ? "-" : "--"},
      };
      if (run.unstable == 0 && !stable_labelled) {
        keywords["label"] = stable;
        stable_labelled = true;
      }
      if (run.unstable == 1 && !unstable_labelled) {
        keywords["label"] = unstable;
        unstable_labelled = true;
      }
      plt::plot(run.x, run.y, keywords);
    }
  }

  inline void profile(
      const Problem& problem,
      const arma::vec& y,
      const std::string& color,
      const std::string& title,
      const std::string& details,
      bool fit = false
  ) {
    plt::plot(
        arma::conv_to<std::vector<double>>::from(problem.positions()),
        arma::conv_to<std::vector<double>>::from(y),
        {{"color", color}}
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
    plt::title(title, {{"fontsize", "7"}});
    plt::annotate(details, 0.04 * problem.length, 1.3);
    plt::tick_params({{"labelsize", "6"}});
    plt::grid(true);
  }

  inline void mark(double x, double y, const std::string& letter, double dx, double dy) {
    plt::plot({x}, {y}, {{"color", ink}, {"marker", "o"}, {"linestyle", "None"}});
    plt::annotate(letter, x + dx, y + dy);
  }

  inline auto canonical_uniform_index(const UniformBranch& uniform) -> std::vector<int> {
    std::vector<int> index;
    for (std::size_t k = 0; k < uniform.rho.size(); ++k)
      index.push_back(std::max(0, uniform.index[k] - (3.0 * uniform.rho[k] * uniform.rho[k] < 1.0 ? 1 : 0)));
    return index;
  }

  inline auto omega_meta(const Problem& problem, double mu) -> double {
    double rho = mu >= 0.0 ? -1.0 : 1.0;
    for (int iteration = 0; iteration < 20; ++iteration)
      rho -= (rho * rho * rho - rho - mu) / (3.0 * rho * rho - 1.0);
    return problem.length * (0.25 * std::pow(rho * rho - 1.0, 2) - mu * rho);
  }

  inline auto widest(const GrandBranch& branch) -> std::size_t {
    return static_cast<std::size_t>(
        std::ranges::max_element(branch.modal, {}, [](double value) { return std::abs(value); }) - branch.modal.begin()
    );
  }

  struct PowerFit {
    double prefactor;
    double exponent;
  };

  inline auto fit_power_law(const std::vector<PitchforkSample>& samples) -> PowerFit {
    double mean_x = 0.0;
    double mean_y = 0.0;
    for (const auto& sample : samples) {
      mean_x += std::log(std::abs(sample.dmu));
      mean_y += std::log(std::abs(sample.amplitude));
    }
    mean_x /= static_cast<double>(samples.size());
    mean_y /= static_cast<double>(samples.size());
    double covariance = 0.0;
    double variance = 0.0;
    for (const auto& sample : samples) {
      const double x = std::log(std::abs(sample.dmu)) - mean_x;
      covariance += x * (std::log(std::abs(sample.amplitude)) - mean_y);
      variance += x * x;
    }
    const double exponent = covariance / variance;
    return {.prefactor = std::exp(mean_y - exponent * mean_x), .exponent = exponent};
  }

  inline void uniform_axis(const UniformBranch& uniform, bool label) {
    std::vector<double> zeros(uniform.mu.size(), 0.0);
    std::vector<int> unstable(uniform.index.size());
    for (std::size_t k = 0; k < uniform.index.size(); ++k)
      unstable[k] = uniform.index[k] > 0 ? 1 : 0;
    bool stable_labelled = !label;
    bool unstable_labelled = !label;
    for (int pass : {0, 1}) {
      for (const auto& run : runs(uniform.mu, zeros, unstable)) {
        if (run.unstable != pass)
          continue;
        std::map<std::string, std::string> keywords{
            {"color", pass == 0 ? tint(ink, 0.55) : ink},
            {"linestyle", pass == 0 ? "-" : "--"},
        };
        if (pass == 0 && !stable_labelled) {
          keywords["label"] = "uniform, stable";
          stable_labelled = true;
        }
        if (pass == 1 && !unstable_labelled) {
          keywords["label"] = "uniform, unstable";
          unstable_labelled = true;
        }
        plt::plot(run.x, run.y, keywords);
      }
    }
  }

  inline void s_curve(const Problem& problem, const UniformBranch& uniform) {
    figure(ONE_COLUMN);
    stability_line(
        uniform.mu,
        uniform.mass,
        uniform.index,
        colors[0],
        R"(stable, $n_- = 0$)",
        R"(unstable, $n_- \geq 1$)"
    );
    const double rho_fold = 1.0 / std::sqrt(3.0);
    const std::vector<double> fold_mu{-2.0 / (3.0 * std::sqrt(3.0)), 2.0 / (3.0 * std::sqrt(3.0))};
    const std::vector<double> fold_mass{-problem.length * rho_fold, problem.length * rho_fold};
    plt::plot(fold_mu, fold_mass, {{"color", ink}, {"marker", "o"}, {"linestyle", "None"}, {"label", "folds"}});
    for (std::size_t k = 0; k < fold_mu.size(); ++k)
      plt::annotate(
          std::format(R"($\mu = {:+.4f}$)", fold_mu[k]),
          fold_mu[k] > 0.0 ? fold_mu[k] + 0.03 : fold_mu[k] - 0.36,
          fold_mass[k] + (fold_mass[k] > 0.0 ? 1.5 : -3.0)
      );
    plt::axhline(0.0, 0.0, 1.0, {{"color", muted}, {"linewidth", "0.5"}});
    plt::axvline(0.0, 0.0, 1.0, {{"color", muted}, {"linewidth", "0.5"}});
    plt::xlim(-0.8, 0.8);
    plt::xlabel(R"($\mu$)");
    plt::ylabel(R"($N = \int_0^L \rho\,dx$)");
    plt::legend({{"loc", "center right"}, {"fontsize", "6"}});
    plt::grid(true);
    save("s_curve");
  }

  inline void swallowtail(const Problem& problem, const Results& results) {
    figure(TWO_COLUMN);
    plt::subplots_adjust(
        {{"left", 0.07}, {"right", 0.98}, {"bottom", 0.14}, {"top", 0.93}, {"wspace", 0.35}, {"hspace", 0.45}}
    );
    panel(2, 4, 0, 0, 2, 2);
    stability_line(
        results.uniform.mu,
        results.uniform.omega,
        results.uniform.index,
        ink,
        "uniform, stable",
        "uniform, unstable"
    );
    for (const auto& branch : results.grand) {
      stability_line(
          branch.mu,
          branch.omega,
          branch.index,
          colors[static_cast<std::size_t>(branch.mode - 1)],
          "",
          std::format(R"(mode $m = {}$, unstable)", branch.mode)
      );
    }
    const arma::vec uniform(problem.nodes, arma::fill::zeros);
    mark(0.0, problem.grand_potential(uniform, 0.0), "A", 0.02, 0.25);
    const std::array<std::string, 3> letters{"B", "C", "D"};
    for (int mode = 1; mode <= 3; ++mode) {
      const auto& branch = results.branch(mode);
      const std::size_t k = widest(branch);
      plt::plot({branch.mu[k]}, {branch.omega[k]}, {{"color", ink}, {"marker", "o"}, {"linestyle", "None"}});
      callout(
          letters[static_cast<std::size_t>(mode - 1)],
          branch.mu[k],
          branch.omega[k],
          0.45,
          1.5 * mode - 2.5
      );
    }
    plt::xlim(-1.0, 0.6);
    plt::ylim(-6.0, 9.0);
    plt::xlabel(R"($\mu$)");
    plt::ylabel(R"($\Omega$)");
    plt::legend({{"loc", "center left"}, {"fontsize", "6"}});
    plt::grid(true);

    const std::array<std::array<long, 2>, 4> cells{{{0, 2}, {0, 3}, {1, 2}, {1, 3}}};
    panel(2, 4, cells[0][0], cells[0][1]);
    profile(problem, uniform, ink, "(A)", std::format(R"(uniform, $\Omega = {:.2f}$)", problem.grand_potential(uniform, 0.0)));
    for (int mode = 1; mode <= 3; ++mode) {
      const auto& branch = results.branch(mode);
      const std::size_t k = widest(branch);
      panel(2, 4, cells[static_cast<std::size_t>(mode)][0], cells[static_cast<std::size_t>(mode)][1]);
      profile(
          problem,
          branch.curve[k].x.head(problem.nodes),
          colors[static_cast<std::size_t>(mode - 1)],
          std::format("({})", letters[static_cast<std::size_t>(mode - 1)]),
          std::format(R"(mode $m = {}$, $\Omega = {:.2f}$)", mode, branch.omega[k])
      );
    }
    save("swallowtail", false);
  }

  inline void branches(const Problem& problem, const Results& results) {
    figure(TWO_COLUMN_TALL);
    plt::subplots_adjust(
        {{"left", 0.08}, {"right", 0.98}, {"bottom", 0.16}, {"top", 0.97}, {"wspace", 0.22}, {"hspace", 0.45}}
    );

    panel(3, 2, 0, 0, 2, 2);
    uniform_axis(results.uniform, true);
    for (const auto& branch : results.grand) {
      stability_line(
          branch.mu,
          branch.modal,
          branch.index,
          colors[static_cast<std::size_t>(branch.mode - 1)],
          "",
          std::format(R"(mode $m = {}$, translated pair)", branch.mode)
      );
      std::vector<double> translated(branch.modal.size());
      for (std::size_t k = 0; k < branch.modal.size(); ++k)
        translated[k] = -branch.modal[k];
      stability_line(branch.mu, translated, branch.index, colors[static_cast<std::size_t>(branch.mode - 1)], "", "");
    }
    std::vector<double> bx, by;
    for (const auto& point : results.uniform.bifurcations) {
      bx.push_back(point.mu);
      by.push_back(0.0);
    }
    plt::plot(bx, by, {{"color", ink}, {"marker", "o"}, {"linestyle", "None"}, {"label", "bifurcation"}});
    for (const auto& point : results.uniform.bifurcations)
      plt::annotate(std::to_string(point.mode), point.mu - 0.006, point.rho < 0.0 ? 0.08 : -0.16);
    for (double side : {-1.0, 1.0}) {
      const double a = side > 0.0 ? 0.34 : -0.40;
      const double b = side > 0.0 ? 0.40 : -0.34;
      plt::plot({a, b, b, a, a}, {-0.12, -0.12, 0.12, 0.12, -0.12}, {{"color", muted}, {"linewidth", "0.7"}});
    }
    plt::xlim(-0.45, 0.45);
    plt::ylim(-1.35, 1.35);
    plt::xlabel(R"($\mu$)");
    plt::ylabel(R"($a_m$)");
    plt::grid(true);
    figure_legend(3);

    const Bounds top = current_bounds();
    for (double side : {-1.0, 1.0}) {
      inset(top, side < 0.0 ? 0.14 : 0.64, 0.66, 0.22, 0.24);
      uniform_axis(results.uniform, false);
      for (const auto& branch : results.grand) {
        plt::plot(
            branch.mu,
            branch.modal,
            {{"color", colors[static_cast<std::size_t>(branch.mode - 1)]}, {"linestyle", "--"}}
        );
        std::vector<double> translated(branch.modal.size());
        for (std::size_t k = 0; k < branch.modal.size(); ++k)
          translated[k] = -branch.modal[k];
        plt::plot(
            branch.mu,
            translated,
            {{"color", colors[static_cast<std::size_t>(branch.mode - 1)]}, {"linestyle", "--"}}
        );
      }
      for (const auto& point : results.uniform.bifurcations) {
        if (side * point.mu < 0.34)
          continue;
        plt::plot({point.mu}, {0.0}, {{"color", ink}, {"marker", "o"}, {"linestyle", "None"}});
        plt::annotate(std::to_string(point.mode), point.mu - 0.002, point.mode % 2 == 0 ? 0.04 : -0.08);
      }
      plt::xlim(side < 0.0 ? -0.40 : 0.34, side < 0.0 ? -0.34 : 0.40);
      plt::ylim(-0.15, 0.15);
      plt::yticks(std::vector<double>{-0.1, 0.0, 0.1});
      plt::tick_params({{"labelsize", "6"}});
    }

    for (long column : {0L, 1L}) {
      panel(3, 2, 2, column);
      const bool against_mass = column == 1;
      for (const auto& branch : results.grand) {
        std::vector<double> x, gap;
        for (std::size_t k = 0; k < branch.mu.size(); ++k) {
          if (std::abs(branch.mu[k]) < 0.4) {
            x.push_back(against_mass ? branch.mass[k] : branch.mu[k]);
            gap.push_back(branch.omega[k] - omega_meta(problem, branch.mu[k]));
          }
        }
        std::vector<int> index(x.size(), 1);
        stability_line(x, gap, index, colors[static_cast<std::size_t>(branch.mode - 1)], "", "");
      }
      const double sigma = exact::surface_tension(problem.kappa);
      for (int interfaces : {2, 4, 6})
        plt::axhline(
            interfaces * sigma,
            0.0,
            1.0,
            {{"color", muted}, {"linewidth", "0.6"}, {"linestyle", ":"}}
        );
      plt::yticks(
          std::vector<double>{0.0, sigma, 2.0 * sigma, 3.0 * sigma, 4.0 * sigma, 5.0 * sigma, 6.0 * sigma},
          std::vector<std::string>{
              "0", R"($\sigma$)", R"($2\sigma$)", R"($3\sigma$)",
              R"($4\sigma$)", R"($5\sigma$)", R"($6\sigma$)"
          }
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

  inline void pitchfork_zoom(const Problem& problem, const Results& results) {
    figure(TWO_COLUMN_TALL);
    plt::subplots_adjust({{"left", 0.09}, {"right", 0.98}, {"bottom", 0.16}, {"top", 0.95}, {"wspace", 0.4}});
    for (std::size_t j = 0; j < results.pitchfork.size(); ++j) {
      const int mode = static_cast<int>(j + 1);
      const auto& branch = results.branch(mode);
      const auto& samples = results.pitchfork[j];
      const auto bifurcation = *std::ranges::find_if(
          results.uniform.bifurcations,
          [mode](const Bifurcation& point) { return point.mode == mode && point.rho < 0.0; }
      );
      const auto fit = fit_power_law(samples);
      std::vector<double> dmu, amplitude;
      double span = 0.0;
      double a_max = 0.0;
      for (const auto& sample : samples) {
        dmu.push_back(sample.dmu);
        amplitude.push_back(sample.amplitude);
        span = std::max(span, std::abs(sample.dmu));
        a_max = std::max(a_max, std::abs(sample.amplitude));
      }
      const double side = samples.front().dmu > 0.0 ? 1.0 : -1.0;
      panel(1, static_cast<long>(results.pitchfork.size()), 0, static_cast<long>(j));
      plt::plot(
          {-1.2 * span, 2.0 * span},
          {0.0, 0.0},
          {{"color", ink}, {"linestyle", "--"}, {"label", "uniform, unstable"}}
      );
      std::vector<double> traced_x, traced_y, translated;
      for (std::size_t k = 0; k < branch.mu.size(); ++k) {
        if (std::abs(branch.modal[k]) <= 1.2 * a_max) {
          traced_x.push_back(branch.mu[k] - bifurcation.mu);
          traced_y.push_back(branch.modal[k]);
          translated.push_back(-branch.modal[k]);
        }
      }
      plt::plot(
          traced_x,
          traced_y,
          {{"color", tint(colors[j], 0.45)}, {"linewidth", "2"}, {"linestyle", "--"}, {"label", "traced representative"}}
      );
      plt::plot(
          traced_x,
          translated,
          {{"color", tint(colors[j], 0.45)}, {"linewidth", "2"}, {"linestyle", "--"}, {"label", "translated copy"}}
      );
      plt::plot(
          dmu,
          amplitude,
          {{"color", colors[j]}, {"marker", "o"}, {"markersize", "2.5"}, {"linestyle", "None"}, {"label", "solved at fixed $a_m$"}}
      );
      std::vector<double> negative(amplitude.size());
      for (std::size_t k = 0; k < amplitude.size(); ++k)
        negative[k] = -amplitude[k];
      plt::plot(dmu, negative, {{"color", colors[j]}, {"marker", "o"}, {"markersize", "2.5"}, {"linestyle", "None"}});
      std::vector<double> a_curve, leading_curve;
      for (double a : arma::linspace(-a_max, a_max, 201)) {
        a_curve.push_back(a);
        leading_curve.push_back(side * std::pow(std::abs(a) / fit.prefactor, 1.0 / fit.exponent));
      }
      plt::plot(
          leading_curve,
          a_curve,
          {{"color", ink},
           {"linewidth", "0.8"},
           {"linestyle", "-."},
           {"label", R"(fit, $a \propto |\mu - \mu_m|^\beta$)"}}
      );
      std::size_t pick = 0;
      for (std::size_t k = 1; k < samples.size(); ++k) {
        if (std::abs(std::abs(samples[k].amplitude) - 0.8 * a_max)
            < std::abs(std::abs(samples[pick].amplitude) - 0.8 * a_max))
          pick = k;
      }
      mark(samples[pick].dmu, samples[pick].amplitude, "A", 0.06 * span, 0.0);
      mark(samples[pick].dmu, -samples[pick].amplitude, "B", 0.06 * span, 0.0);
      plt::xlim(side > 0.0 ? -0.1 * span : -1.1 * span, side > 0.0 ? 1.1 * span : 1.4 * span);
      plt::ylim(-1.2 * a_max, 1.2 * a_max);
      plt::xlabel(std::format(R"($\mu - \mu_{}$)", mode));
      plt::ylabel(std::format(R"($a_{}$)", mode));
      plt::title(std::format(R"(mode $m = {}$, $\mu_{} = {:.6f}$)", mode, mode, bifurcation.mu));
      plt::grid(true);
      if (j == 0)
        figure_legend(3);
      const Bounds main = current_bounds();
      inset(main, 0.56, 0.74, 0.41, 0.2);
      profile(
          problem,
          samples[pick].y,
          colors[j],
          "(A)",
          "",
          true
      );
      inset(main, 0.56, 0.06, 0.41, 0.2);
      profile(
          problem,
          arma::shift(
              samples[pick].y,
              static_cast<arma::sword>(problem.nodes / (2 * mode))
          ),
          colors[j],
          "(B)",
          "",
          true
      );
      inset(main, 0.62, 0.4, 0.35, 0.22);
      std::vector<double> abs_dmu, abs_a;
      for (const auto& sample : samples) {
        abs_dmu.push_back(std::abs(sample.dmu));
        abs_a.push_back(std::abs(sample.amplitude));
      }
      const double x_lo = *std::ranges::min_element(abs_dmu);
      const double x_hi = *std::ranges::max_element(abs_dmu);
      plt::plot(abs_dmu, abs_a, {{"color", colors[j]}, {"marker", "o"}, {"markersize", "1.5"}, {"linestyle", "None"}});
      plt::plot(
          std::vector<double>{x_lo, x_hi},
          std::vector<double>{fit.prefactor * std::pow(x_lo, fit.exponent), fit.prefactor * std::pow(x_hi, fit.exponent)},
          {{"color", ink}, {"linewidth", "0.8"}, {"linestyle", "-."}}
      );
      log_axes();
      plt::tick_params({{"labelsize", "5"}});
      text(std::format(R"($\beta = {:.4f}$)", fit.exponent), x_lo * 1.5, 0.5 * a_max, "6");
    }
    save("pitchfork_zoom", false);
  }

  inline void walk(const Problem& problem, const Results& results) {
    const auto& branch = results.branch(1);
    const std::array<std::string, 6> letters{"A", "B", "C", "D", "E", "F"};
    const std::array<double, 6> fractions{0.08, 0.24, 0.40, 0.56, 0.72, 0.88};
    figure(TWO_COLUMN_TALL);
    plt::subplots_adjust(
        {{"left", 0.07}, {"right", 0.98}, {"bottom", 0.07}, {"top", 0.97}, {"wspace", 0.35}, {"hspace", 0.5}}
    );
    panel(3, 6, 0, 0, 2, 6);
    uniform_axis(results.uniform, true);
    stability_line(branch.mu, branch.modal, branch.index, colors[0], "", "mode $m = 1$, phase-fixed trace");
    std::vector<double> translated(branch.modal.size());
    for (std::size_t k = 0; k < branch.modal.size(); ++k)
      translated[k] = -branch.modal[k];
    stability_line(branch.mu, translated, branch.index, colors[0], "", "translated copy");
    std::array<std::size_t, 6> selected{};
    for (std::size_t j = 0; j < selected.size(); ++j) {
      selected[j] = static_cast<std::size_t>(fractions[j] * static_cast<double>(branch.mu.size() - 1));
      mark(branch.mu[selected[j]], branch.modal[selected[j]], letters[j], 0.008, 0.06);
    }
    plt::xlim(-0.42, 0.42);
    plt::ylim(-1.4, 1.4);
    plt::xlabel(R"($\mu$)");
    plt::ylabel(R"($a_1$)");
    plt::legend({{"loc", "lower left"}, {"fontsize", "6"}});
    plt::grid(true);
    for (std::size_t j = 0; j < selected.size(); ++j) {
      const std::size_t k = selected[j];
      panel(3, 6, 2, static_cast<long>(j));
      profile(
          problem,
          branch.curve[k].x.head(problem.nodes),
          colors[0],
          std::format("({})", letters[j]),
          std::format("$\\mu = {:+.2f}$\n$\\Delta\\Omega = {:.3f}$", branch.mu[k], branch.omega[k] - omega_meta(problem, branch.mu[k]))
      );
      plt::xlabel(R"($x$)");
    }
    save("walk", false);
  }

  inline void canonical(const Problem& problem, const Results& results) {
    figure(TWO_COLUMN_TALL);
    plt::subplots_adjust({{"left", 0.07}, {"right", 0.98}, {"bottom", 0.17}, {"top", 0.98}});
    panel(1, 1, 0, 0);
    stability_line(
        results.uniform.mass,
        results.uniform.mu,
        canonical_uniform_index(results.uniform),
        ink,
        "uniform, stable at fixed N",
        "uniform, unstable at fixed N"
    );
    const auto& first_mode = results.branch(1);
    plt::plot(
        first_mode.mass,
        first_mode.mu,
        {{"color", tint(colors[0], 0.7)}, {"linewidth", "5"}, {"label", R"(mode $m = 1$, traced in $\mu$)"}}
    );
    bool labelled = false;
    for (const auto& branch : results.canonical) {
      stability_line(
          branch.mass,
          branch.mu,
          branch.index,
          colors[0],
          labelled ? "" : "interface pair, stable",
          labelled ? "" : "interface pair, saddle"
      );
      labelled = true;
    }
    for (int mode : {2, 3}) {
      const auto& branch = results.branch(mode);
      std::vector<int> index;
      for (const auto& point : branch.curve)
        index.push_back(problem.constrained_index(point.x.head(problem.nodes)));
      stability_line(
          branch.mass,
          branch.mu,
          index,
          colors[static_cast<std::size_t>(mode - 1)],
          std::format(R"(mode $m = {}$, stable at fixed $N$)", mode),
          std::format(R"(mode $m = {}$, unstable at fixed $N$)", mode)
      );
    }
    plt::axhline(0.0, 0.0, 1.0, {{"color", muted}, {"linewidth", "0.5"}});
    for (const auto& point : results.points) {
      double dx = -1.3;
      double dy = 0.02;
      if (point.letter == "B") {
        dx = -0.85;
        dy = -0.03;
      }
      if (point.letter == "C") {
        dx = -0.85;
        dy = 0.035;
      }
      if (point.letter == "D") {
        dx = -0.85;
        dy = 0.025;
      }
      if (point.letter == "E" || point.letter == "F") {
        dx = 0.5;
        dy = 0.03;
      }
      mark(point.mass, point.mu, point.letter, dx, dy);
    }
    const auto& positive = results.canonical.front();
    const std::size_t fold = static_cast<std::size_t>(
        std::ranges::max_element(positive.mass) - positive.mass.begin()
    );
    callout(R"(Maxwell plateau, $\mu = 0$)", 0.0, 0.0, -10.5, -0.2);
    callout("finite-size fold", positive.mass[fold], positive.mu[fold], 4.5, 0.08);
    callout(R"(fixed-$N$ saddles)", results.points[2].mass, results.points[2].mu, 4.0, -0.1);
    callout("uniform unstable at fixed N", results.points[5].mass, results.points[5].mu, -4.5, 0.25);
    plt::xlim(-1.1 * problem.length, 1.1 * problem.length);
    plt::ylim(-0.75, 0.75);
    plt::xlabel(R"($N$)");
    plt::ylabel(R"($\mu$)");
    plt::grid(true);
    figure_legend(3);

    const Bounds main = current_bounds();
    const std::array<std::array<double, 2>, 6> corners{
        {{0.04, 0.77}, {0.33, 0.77}, {0.62, 0.77}, {0.10, 0.03}, {0.33, 0.03}, {0.56, 0.03}}
    };
    for (std::size_t k = 0; k < results.points.size(); ++k) {
      const auto& point = results.points[k];
      inset(main, corners[k][0], corners[k][1], 0.19, 0.17);
      profile(
          problem,
          point.y,
          point.letter == "C" || point.letter == "D" || point.letter == "E" ? colors[0] : ink,
          std::format("({})", point.letter),
          std::format(R"($F = {:.2f}$, $n_- = {}$)", point.helmholtz, point.index)
      );
    }
    save("canonical", false);
  }

  inline void translation(const Problem& problem, const Results& results) {
    figure(TWO_COLUMN);
    plt::subplots_adjust({{"left", 0.09}, {"right", 0.98}, {"bottom", 0.17}, {"top", 0.92}, {"wspace", 0.35}});
    panel(1, 2, 0, 0);
    const arma::vec shifted = arma::shift(results.pair, static_cast<arma::sword>(problem.nodes / 4));
    plt::plot(
        arma::conv_to<std::vector<double>>::from(problem.positions()),
        arma::conv_to<std::vector<double>>::from(results.pair),
        {{"color", colors[0]}, {"label", "phase-fixed pair"}}
    );
    plt::plot(
        arma::conv_to<std::vector<double>>::from(problem.positions()),
        arma::conv_to<std::vector<double>>::from(shifted),
        {{"color", colors[1]}, {"linestyle", "--"}, {"label", "translated pair"}}
    );
    plt::xlim(0.0, problem.length);
    plt::ylim(-1.15, 1.15);
    plt::xlabel(R"($x$)");
    plt::ylabel(R"($\rho$)");
    plt::legend({{"loc", "upper right"}});
    plt::grid(true);

    panel(1, 2, 0, 1);
    const arma::vec tangent = problem.derivative(results.pair);
    plt::plot(
        arma::conv_to<std::vector<double>>::from(problem.positions()),
        arma::conv_to<std::vector<double>>::from(tangent),
        {{"color", colors[2]}, {"label", R"($\partial_x\rho_{\rm ref}$)"}}
    );
    plt::axhline(0.0, 0.0, 1.0, {{"color", muted}, {"linewidth", "0.5"}});
    plt::xlim(0.0, problem.length);
    plt::xlabel(R"($x$)");
    plt::ylabel(R"($\partial_x\rho_{\rm ref}$)");
    plt::legend({{"loc", "upper right"}});
    plt::grid(true);
    save("translation_gauge", false);
  }

  inline void nested_pitchforks(const Problem& problem, const Results& results) {
    figure(TWO_COLUMN_TALL);
    plt::subplots_adjust(
        {{"left", 0.07}, {"right", 0.98}, {"bottom", 0.07}, {"top", 0.98}, {"wspace", 0.35}, {"hspace", 0.5}}
    );
    panel(3, 6, 0, 0, 2, 6);
    plt::plot(
        results.nested.lambda,
        std::vector<double>(results.nested.lambda.size(), 0.0),
        {{"color", ink}, {"linestyle", "--"}, {"label", R"(uniform $\rho = 0$, unstable)"}}
    );
    const std::array<std::string, 6> letters{"A", "B", "C", "D", "E", "F"};
    for (const auto& branch : results.nested.branches) {
      std::map<std::string, std::string> keywords{
          {"color", colors[static_cast<std::size_t>(branch.mode - 1)]},
          {"linestyle", "--"},
          {"label", std::format(R"(mode $m = {}$, translated pair)", branch.mode)},
      };
      plt::plot(branch.lambda, branch.modal, keywords);
      std::vector<double> translated(branch.modal.size());
      for (std::size_t k = 0; k < branch.modal.size(); ++k)
        translated[k] = -branch.modal[k];
      plt::plot(branch.lambda, translated, {{"color", colors[static_cast<std::size_t>(branch.mode - 1)]}, {"linestyle", "--"}});
      mark(branch.lambda.back(), branch.modal.back(), letters[static_cast<std::size_t>(branch.mode - 1)], 0.3, 0.0);
    }
    for (const auto& [mode, lambda] : results.nested.bifurcations) {
      plt::plot({lambda}, {0.0}, {{"color", ink}, {"marker", "o"}, {"linestyle", "None"}});
      plt::annotate(std::to_string(mode), lambda - 0.25, -0.2);
    }
    plt::xlim(2.5, results.nested_lambda_max + 1.5);
    plt::ylim(-1.5, 1.5);
    plt::xlabel(R"($L / \sqrt{\kappa}$)");
    plt::ylabel(R"($a_m$)");
    plt::legend({{"loc", "lower left"}, {"fontsize", "6"}});
    plt::grid(true);
    for (const auto& branch : results.nested.branches) {
      panel(3, 6, 2, static_cast<long>(branch.mode - 1));
      profile(
          problem,
          branch.profiles.back(),
          colors[static_cast<std::size_t>(branch.mode - 1)],
          std::format("({})", letters[static_cast<std::size_t>(branch.mode - 1)]),
          std::format(R"(mode $m = {}$)", branch.mode)
      );
      plt::xlabel(R"($x$)");
    }
    save("nested_pitchforks", false);
  }

  inline void branch_count(const Results& results) {
    figure(ONE_COLUMN);
    std::vector<double> found, formula;
    for (double lambda : results.nested.lambda) {
      int count = 0;
      for (const auto& [mode, threshold] : results.nested.bifurcations)
        count += threshold <= lambda ? 1 : 0;
      found.push_back(count);
      formula.push_back(std::floor(lambda / (2.0 * std::numbers::pi)));
    }
    plt::plot(
        results.nested.lambda,
        formula,
        {{"color", muted}, {"linewidth", "3"}, {"label", R"($\lfloor L/(2\pi\sqrt{\kappa}) \rfloor$)"}}
    );
    plt::plot(results.nested.lambda, found, {{"color", colors[0]}, {"label", "branches found"}});
    std::vector<double> x, y;
    for (const auto& [mode, threshold] : results.nested.bifurcations) {
      x.push_back(threshold);
      y.push_back(mode);
    }
    plt::plot(x, y, {{"color", ink}, {"marker", "o"}, {"linestyle", "None"}, {"label", "detected bifurcation"}});
    plt::xlabel(R"($L / \sqrt{\kappa}$)");
    plt::ylabel("number of branches");
    plt::legend({{"loc", "upper left"}, {"fontsize", "6"}});
    plt::grid(true);
    save("branch_count");
  }

} // namespace periodic::plot

#endif
