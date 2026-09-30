#pragma once

#include "utils.hpp"

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
          kw{{"color", color}, {"linewidth", "2"}, {"linestyle", r.index == 0 ? "-" : "--"}};
      if (label && r.index == 0 && !stable_labelled) {
        kw["label"] = R"(stable, $n_- = 0$)";
        stable_labelled = true;
      }
      if (label && r.index == 1 && !unstable_labelled) {
        kw["label"] = R"(unstable, $n_- \geq 1$)";
        unstable_labelled = true;
      }
      plt::plot(r.x, r.y, kw);
    }
  }

  // plt::subplot passes floats, which recent matplotlib rejects; select the
  // axes through the library wrapper instead.
  inline void subplot(int rows, int columns, int index) {
    PyObject* pyplot = dft::plotting::detail::import_module("matplotlib.pyplot");
    Py_DECREF(dft::plotting::detail::select_subplot(pyplot, rows, columns, index));
    Py_DECREF(pyplot);
  }

  inline void save(const std::string& name) {
    plt::tight_layout();
    plt::save("exports/" + name + ".pdf");
    plt::save("exports/" + name + ".png");
    plt::clf();
    plt::close();
    std::cout << "Plot saved: exports/" << name << ".png\n";
  }

  inline void fold_markers(const utils::Problem& p, const utils::Branch& u, bool omega) {
    std::vector<double> fx, fy;
    for (const auto& f : u.folds) {
      fx.push_back(f.mu);
      fy.push_back(omega ? p.grand_potential(f.point.x, f.mu) : p.mass(f.point.x));
    }
    plt::plot(
        fx,
        fy,
        {{"color", ink}, {"marker", "o"}, {"markersize", "8"}, {"linestyle", "None"}, {"label", "folds"}}
    );
  }

  // Figure 1: N against mu on the uniform branch.
  inline void s_curve(const utils::Problem& p, const utils::Branch& u) {
    plt::figure_size(800, 560);
    stability_line(u.mu, u.mass, u.index, branch_colors[0], true);
    fold_markers(p, u, false);
    for (const auto& f : u.folds) {
      double n = p.mass(f.point.x);
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
  inline void swallowtail(const utils::Problem& p, const utils::Branch& u) {
    plt::figure_size(800, 560);
    stability_line(u.mu, u.omega, u.index, branch_colors[0], true);
    fold_markers(p, u, true);
    plt::plot(
        {0.0},
        {0.0},
        {{"color", ink},
         {"marker", "D"},
         {"markersize", "7"},
         {"linestyle", "None"},
         {"label", R"(crossing $\mu_0 = 0$)"}}
    );
    arma::vec mid(p.nodes, arma::fill::zeros);
    double omega_mid = p.grand_potential(mid, 0.0);
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

  // Figure 3: bifurcation diagram (amplitude) and gap to the metastable state.
  inline void branches(const utils::Problem& p, const utils::Branch& u, const std::vector<utils::Branch>& bs) {
    plt::figure_size(1300, 560);

    subplot(1, 2, 1);
    std::vector<double> zeros(u.mu.size(), 0.0);
    plt::plot(u.mu, zeros, {{"color", ink}, {"linewidth", "2"}, {"label", "uniform"}});
    std::vector<double> bx, by;
    for (const auto& e : u.bifurcations) {
      bx.push_back(e.mu);
      by.push_back(0.0);
    }
    plt::plot(
        bx,
        by,
        {{"color", ink}, {"marker", "o"}, {"markersize", "6"}, {"linestyle", "None"}, {"label", "bifurcation points"}}
    );
    for (std::size_t b = 0; b < bs.size(); ++b) {
      for (const auto& r : runs(bs[b].mu, bs[b].amplitude, bs[b].index)) {
        plt::plot(r.x, r.y, {{"color", branch_colors[b]}, {"linewidth", "2"}, {"linestyle", index_style(r.index)}});
      }
      auto peak = std::ranges::max_element(bs[b].amplitude) - bs[b].amplitude.begin();
      plt::annotate(bs[b].name, 0.03, bs[b].amplitude[static_cast<std::size_t>(peak)] + 0.08);
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
    plt::ylim(-0.2, 4.8);
    plt::xlabel(R"($\mu$)");
    plt::ylabel(R"($\| \rho - \bar\rho \|$)");
    plt::title("Branches with n interfaces");
    plt::legend({{"loc", "upper left"}, {"fontsize", "small"}});
    plt::grid(true);

    subplot(1, 2, 2);
    const double mu_f = utils::exact::fold_chemical_potential();
    auto gap = [&](const std::vector<double>& mu,
                   const std::vector<double>& omega,
                   const std::vector<int>& index,
                   const std::string& color,
                   const std::string& label) {
      std::vector<double> x, y;
      std::vector<int> idx;
      for (std::size_t k = 0; k < mu.size(); ++k) {
        if (std::abs(mu[k]) >= mu_f)
          continue;
        double rho = utils::metastable_density(mu[k]);
        double omega_meta = p.length * (0.25 * std::pow(rho * rho - 1.0, 2) - mu[k] * rho);
        x.push_back(mu[k]);
        y.push_back(omega[k] - omega_meta);
        idx.push_back(index[k]);
      }
      bool first = true;
      for (const auto& r : runs(x, y, idx)) {
        std::map<std::string, std::string>
            kw{{"color", color}, {"linewidth", "2"}, {"linestyle", index_style(r.index)}};
        if (first) {
          kw["label"] = label;
          first = false;
        }
        plt::plot(r.x, r.y, kw);
      }
    };
    // Middle arc of the uniform branch: between the two folds.
    {
      std::vector<double> mu, omega;
      std::vector<int> idx;
      for (std::size_t k = 0; k < u.mu.size(); ++k) {
        double rho = p.mass(u.curve[k].x) / p.length;
        if (std::abs(rho) < utils::exact::fold_density()) {
          mu.push_back(u.mu[k]);
          omega.push_back(u.omega[k]);
          idx.push_back(u.index[k]);
        }
      }
      gap(mu, omega, idx, ink, "uniform, middle arc");
    }
    for (std::size_t b = 0; b < bs.size(); ++b)
      gap(bs[b].mu, bs[b].omega, bs[b].index, branch_colors[b], bs[b].name);
    const double sigma = utils::exact::surface_tension(p.kappa);
    for (int n = 1; n <= 3; ++n) {
      plt::axhline(n * sigma, 0.0, 1.0, {{"color", muted}, {"linewidth", "0.8"}, {"linestyle", ":"}});
      plt::annotate(std::format(R"(${}\sigma$)", n == 1 ? std::string{} : std::to_string(n)), 0.33, n * sigma + 0.08);
    }
    plt::xlim(-0.4, 0.4);
    plt::ylim(0.0, 7.5);
    plt::xlabel(R"($\mu$)");
    plt::ylabel(R"($\Omega - \Omega_{\rm meta}$)");
    plt::title("Gap to the metastable uniform state");
    plt::legend({{"loc", "upper left"}, {"fontsize", "small"}});
    plt::grid(true);
    save("branches");
  }

  // Figure 4: profiles along each branch.
  inline void profiles(const utils::Problem& p, const std::vector<utils::Branch>& bs) {
    plt::figure_size(1400, 460);
    auto x = arma::conv_to<std::vector<double>>::from(p.positions());
    const double w = std::sqrt(2.0 * p.kappa);
    for (std::size_t b = 0; b < bs.size(); ++b) {
      subplot(1, static_cast<int>(bs.size()), static_cast<int>(b + 1));
      const auto& br = bs[b];
      auto peak = static_cast<std::size_t>(std::ranges::max_element(br.amplitude) - br.amplitude.begin());
      const double a_max = br.amplitude[peak];
      // Points on the mu > 0 half at a fraction of the peak amplitude.
      for (double frac : {0.1, 0.4, 0.7, 1.0}) {
        std::size_t pick = peak;
        for (std::size_t k = 0; k <= peak; ++k) {
          if (br.amplitude[k] >= frac * a_max) {
            pick = k;
            break;
          }
        }
        auto y = arma::conv_to<std::vector<double>>::from(br.curve[pick].x);
        plt::plot(
            x,
            y,
            {{"color", tint(branch_colors[b], 0.7 * (1.0 - frac))},
             {"linewidth", "2"},
             {"label", std::format(R"($\mu = {:+.3f}$)", std::abs(br.mu[pick]) < 5e-4 ? 0.0 : br.mu[pick])}}
        );
      }
      // Tanh chain with interfaces at (2j - 1) L / 2n, signed to match the peak profile.
      const int n = static_cast<int>(b + 1);
      arma::vec chain(p.nodes, arma::fill::ones);
      arma::vec xs = p.positions();
      for (int j = 1; j <= n; ++j)
        chain %= arma::tanh((xs - (2.0 * j - 1.0) * p.length / (2.0 * n)) / w);
      if ((chain(0) > 0.0) != (br.curve[peak].x(0) > 0.0))
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
      plt::title(std::format("Branch {}", br.name));
      plt::legend({{"loc", "upper center"}, {"fontsize", "x-small"}});
      plt::grid(true);
    }
    save("profiles");
  }

  // Figure 5: the same states parametrised by N (fixed mass).
  inline void canonical(
      const utils::Problem& p,
      const utils::Branch& u,
      const utils::Branch& kink,
      const std::vector<utils::CanonicalBranch>& traced
  ) {
    plt::figure_size(800, 560);
    // Uniform branch: index at fixed N.
    std::vector<int> u_index;
    for (const auto& q : u.curve)
      u_index.push_back(p.constrained_index(q.x));
    std::vector<double> u_n = u.mass;
    stability_line(u_n, u.mu, u_index, ink, true);
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
            kw{{"color", branch_colors[0]}, {"linewidth", "2"}, {"linestyle", r.index == 0 ? "-" : "--"}};
        if (first && r.index == 0) {
          kw["label"] = R"($n = 1$, traced in $N$)";
          first = false;
        }
        plt::plot(r.x, r.y, kw);
      }
    }
    plt::axhline(0.0, 0.0, 1.0, {{"color", muted}, {"linewidth", "0.6"}});
    plt::xlim(-1.3 * p.length, 1.3 * p.length);
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
