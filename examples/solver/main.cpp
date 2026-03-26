#include "dft.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>

#ifdef DFT_HAS_MATPLOTLIB
#include "matplotlibcpp.h"
#endif

using namespace dft;

/// Build a single-component LJ solver with a given FMT model.
/// The LJ potential must outlive the returned Solver.
static Solver make_lj_solver(
    double dx, const arma::rowvec3& box,
    double diameter, const potentials::LennardJones& lj, double kT,
    functional::fmt::FMTModel model) {
  Solver s;
  auto dens = density::Density(dx, box);
  dens.values().fill(0.1);
  auto sp = std::make_unique<functional::fmt::Species>(std::move(dens), diameter);
  auto& sp_ref = *sp;
  s.add_species(std::move(sp));
  s.add_interaction(std::make_unique<functional::interaction::Interaction>(
      sp_ref, sp_ref, lj, kT));
  s.set_fmt(std::make_unique<functional::fmt::FMT>(std::move(model)));
  return s;
}

/// Coexistence data for one FMT model.
struct CoexData {
  std::string name;
  std::vector<double> T, rho_v, rho_l;
  std::vector<double> T_sp, rho_s1, rho_s2;
  double Tc = 0.0;
  double rho_c = 0.0;
};

/// Sweep coexistence + spinodal for a given FMT model.
static CoexData sweep_coexistence(
    double dx, const arma::rowvec3& box,
    double diameter, const potentials::LennardJones& lj,
    functional::fmt::FMTModel model,
    double T_lo, double T_hi, double dT_coarse, double dT_fine,
    double max_dens, double step, double tol) {

  CoexData cd;
  cd.name = functional::fmt::FMT(model).name();

  // Two-pass sweep: coarse steps, then fine steps near Tc to close the curve.
  std::vector<double> temps;
  for (double kT = T_lo; kT <= T_hi; kT += dT_coarse) {
    temps.push_back(kT);
  }

  double T_last_ok = T_lo;
  for (double kT : temps) {
    auto solver = make_lj_solver(dx, box, diameter, lj, kT, model);

    double rs1 = 0.0;
    double rs2 = 0.0;
    try {
      solver.find_spinodal(max_dens, step, rs1, rs2, tol);
    } catch (const std::exception&) {
      break;
    }
    cd.T_sp.push_back(kT);
    cd.rho_s1.push_back(rs1);
    cd.rho_s2.push_back(rs2);

    double rv = 0.0;
    double rl = 0.0;
    try {
      solver.find_coexistence(max_dens, step, rv, rl, tol);
      cd.T.push_back(kT);
      cd.rho_v.push_back(rv);
      cd.rho_l.push_back(rl);
      T_last_ok = kT;
    } catch (const std::exception&) {
      break;
    }
  }

  // Fine pass: fill in near Tc
  for (double kT = T_last_ok + dT_fine; kT <= T_hi; kT += dT_fine) {
    auto solver = make_lj_solver(dx, box, diameter, lj, kT, model);

    double rs1 = 0.0;
    double rs2 = 0.0;
    try {
      solver.find_spinodal(max_dens, step, rs1, rs2, tol);
    } catch (const std::exception&) {
      break;
    }
    cd.T_sp.push_back(kT);
    cd.rho_s1.push_back(rs1);
    cd.rho_s2.push_back(rs2);

    double rv = 0.0;
    double rl = 0.0;
    try {
      solver.find_coexistence(max_dens, step, rv, rl, tol);
      cd.T.push_back(kT);
      cd.rho_v.push_back(rv);
      cd.rho_l.push_back(rl);
    } catch (const std::exception&) {
      break;
    }
  }

  // Sort spinodal data by temperature (coarse + fine passes may interleave).
  if (cd.T_sp.size() >= 2) {
    std::vector<size_t> order(cd.T_sp.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
      return cd.T_sp[a] < cd.T_sp[b];
    });
    std::vector<double> t_s, r1_s, r2_s;
    double prev = -1.0;
    for (size_t i : order) {
      if (cd.T_sp[i] != prev) {
        t_s.push_back(cd.T_sp[i]);
        r1_s.push_back(cd.rho_s1[i]);
        r2_s.push_back(cd.rho_s2[i]);
        prev = cd.T_sp[i];
      }
    }
    cd.T_sp = std::move(t_s);
    cd.rho_s1 = std::move(r1_s);
    cd.rho_s2 = std::move(r2_s);
  }

  // Estimate critical point from spinodal: where rho_s1 and rho_s2 converge.
  if (cd.T_sp.size() >= 2) {
    // Find the spinodal point with the smallest gap
    size_t i_min = 0;
    double gap_min = cd.rho_s2[0] - cd.rho_s1[0];
    for (size_t i = 1; i < cd.T_sp.size(); ++i) {
      double gap = cd.rho_s2[i] - cd.rho_s1[i];
      if (gap < gap_min) {
        gap_min = gap;
        i_min = i;
      }
    }
    cd.Tc = cd.T_sp[i_min];
    cd.rho_c = 0.5 * (cd.rho_s1[i_min] + cd.rho_s2[i_min]);
  }

  return cd;
}

int main(int argc, char* argv[]) {
#ifdef EXAMPLE_SOURCE_DIR
  std::filesystem::current_path(EXAMPLE_SOURCE_DIR);
#endif
  std::filesystem::create_directories("exports");

  std::string config_path = (argc > 1) ? argv[1] : "config.ini";
  auto cfg = config::ConfigParser(config_path);

  // ── Parameters ────────────────────────────────────────────────────────

  double sigma = cfg.get<double>("potential.sigma");
  double epsilon = cfg.get<double>("potential.epsilon");
  double r_cutoff = cfg.get<double>("potential.r_cutoff");
  double diameter = sigma;
  double dx = cfg.get<double>("grid.dx");
  double box_length = cfg.get<double>("grid.box_length");
  arma::rowvec3 box = {box_length, box_length, box_length};

  double max_dens = cfg.get<double>("coexistence.max_density");
  double step = cfg.get<double>("coexistence.step");
  double tol = cfg.get<double>("coexistence.tolerance");

  potentials::LennardJones lj(sigma, epsilon, r_cutoff);

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "=== LJ fluid phase diagram (mean-field DFT) ===\n"
            << "  sigma = " << sigma << ", epsilon = " << epsilon
            << ", r_c = " << r_cutoff << "\n"
            << "  Grid  = " << box(0) << "^3, dx = " << dx << "\n\n";

  // ── 1. Pressure isotherms (White Bear II) ─────────────────────────────

  // Parse comma-separated temperature list
  std::string temp_str = cfg.get<std::string>("isotherms.temperatures");
  std::vector<double> isotherm_temps;
  {
    std::stringstream ss(temp_str);
    std::string tok;
    while (std::getline(ss, tok, ',')) isotherm_temps.push_back(std::stod(tok));
  }
  int n_rho = static_cast<int>(cfg.get<double>("isotherms.n_rho"));
  double rho_min = cfg.get<double>("isotherms.rho_min");
  double rho_max_iso = cfg.get<double>("isotherms.rho_max");

  std::vector<std::vector<double>> iso_rho(isotherm_temps.size());
  std::vector<std::vector<double>> iso_p(isotherm_temps.size());

  std::cout << "=== Pressure isotherms P*(rho) [White Bear II] ===\n\n";
  for (size_t t = 0; t < isotherm_temps.size(); ++t) {
    double kT = isotherm_temps[t];
    auto solver = make_lj_solver(dx, box, diameter, lj, kT,
                                 functional::fmt::WhiteBearII{});

    std::cout << "T* = " << kT << ":";
    for (int i = 0; i < n_rho; ++i) {
      double rho = rho_min + (rho_max_iso - rho_min) * (i + 1.0) / n_rho;
      double p_star = -solver.grand_potential_density(rho) * kT;
      iso_rho[t].push_back(rho);
      iso_p[t].push_back(p_star);
    }
    std::cout << "  P*(" << rho_min << ") = " << iso_p[t].front()
              << ",  P*(" << rho_max_iso << ") = " << iso_p[t].back() << "\n";
  }

  // ── 2. Coexistence for all four FMT models ────────────────────────────

  std::cout << "\n=== Coexistence sweep (all FMT models) ===\n\n";

  std::vector<functional::fmt::FMTModel> models = {
      functional::fmt::Rosenfeld{},
      functional::fmt::RSLT{},
      functional::fmt::WhiteBearI{},
      functional::fmt::WhiteBearII{},
  };

  std::vector<CoexData> all_coex;
  for (auto& m : models) {
    auto cd = sweep_coexistence(dx, box, diameter, lj, m,
                                cfg.get<double>("sweep.T_lo"), cfg.get<double>("sweep.T_hi"),
                                cfg.get<double>("sweep.dT_coarse"), cfg.get<double>("sweep.dT_fine"),
                                max_dens, step, tol);

    std::cout << cd.name << ": " << cd.T.size() << " coexistence points";
    if (cd.Tc > 0) {
      std::cout << ", T_c ~ " << cd.Tc << ", rho_c ~ " << cd.rho_c;
    }
    std::cout << "\n";

    all_coex.push_back(std::move(cd));
  }

  // Print detailed table for White Bear II
  const auto& wb2 = all_coex.back();
  std::cout << "\n--- White Bear II coexistence data ---\n";
  std::cout << std::setw(8) << "T*"
            << std::setw(14) << "rho_vapor"
            << std::setw(14) << "rho_liquid" << "\n";
  for (size_t i = 0; i < wb2.T.size(); ++i) {
    std::cout << std::setw(8) << wb2.T[i]
              << std::setw(14) << wb2.rho_v[i]
              << std::setw(14) << wb2.rho_l[i] << "\n";
  }

  // ── 3. Plots ──────────────────────────────────────────────────────────

#ifdef DFT_HAS_MATPLOTLIB
  namespace plt = matplotlibcpp;

  // Plot 1: pressure isotherms with van der Waals loops
  {
    plt::figure_size(900, 600);
    std::vector<std::string> styles = {"b-", "c-", "g-", "y-", "r-", "m-", "k--"};
    for (size_t t = 0; t < isotherm_temps.size(); ++t) {
      char label[32];
      std::snprintf(label, sizeof(label), R"($T^* = %.2f$)", isotherm_temps[t]);
      plt::named_plot(label, iso_rho[t], iso_p[t], styles[t % styles.size()]);
    }
    plt::plot({rho_min, rho_max_iso}, {0.0, 0.0},
              {{"color", "gray"}, {"linestyle", ":"}, {"linewidth", "0.8"}});

    plt::xlim(0.0, rho_max_iso + 0.05);
    plt::ylim(-0.6, 3.0);
    plt::xlabel(R"($\rho \sigma^3$)");
    plt::ylabel(R"($P^* = P\sigma^3/\epsilon$)");
    plt::title(R"(Pressure isotherms: LJ fluid (WB-II + mean field))");
    plt::legend();
    plt::grid(true);
    plt::tight_layout();
    plt::save("exports/pressure_isotherms.png");
    plt::close();
    std::cout << "\nPlot: " << std::filesystem::absolute("exports/pressure_isotherms.png") << "\n";
  }

  // Plot 2: coexistence diagram comparison — all four FMT models
  {
    plt::figure_size(900, 650);
    std::vector<std::string> styles = {"b-", "g--", "r-.", "k-"};

    for (size_t m = 0; m < all_coex.size(); ++m) {
      const auto& cd = all_coex[m];
      if (cd.T.empty()) {
        continue;
      }

      // Plot vapor and liquid branches separately to avoid horizontal
      // line artifact at the critical temperature.
      std::string style = styles[m % styles.size()];
      plt::named_plot(cd.name, cd.rho_v, cd.T, style);
      plt::plot(cd.rho_l, cd.T, style);

      // Critical point marker from spinodal convergence
      if (cd.Tc > 0) {
        std::string marker = style.substr(0, 1) + "o";
        plt::plot({cd.rho_c}, {cd.Tc}, marker);
      }
    }

    plt::xlabel(R"($\rho \sigma^3$)");
    plt::ylabel(R"($T^* = k_BT / \epsilon$)");
    plt::title(R"(Liquid-vapor coexistence: LJ fluid (FMT comparison))");
    plt::legend();
    plt::grid(true);
    plt::tight_layout();
    plt::save("exports/coexistence.png");
    plt::close();
    std::cout << "Plot: " << std::filesystem::absolute("exports/coexistence.png") << "\n";
  }

  // Plot 3: spinodal + binodal for White Bear II only (detailed, with splines)
  {
    plt::figure_size(900, 650);
    const auto& cd = all_coex.back();

    // Helper: generate spline-interpolated curve from (T, rho) data.
    // Returns (rho_dense, T_dense) for smooth plotting.
    auto spline_curve = [](const std::vector<double>& t_data,
                           const std::vector<double>& rho_data, int n_pts) {
      std::pair<std::vector<double>, std::vector<double>> result;
      if (t_data.size() < 4) {
        result = {rho_data, t_data};
        return result;
      }
      math::spline::CubicSpline spl(t_data, rho_data);
      double t_lo = t_data.front();
      double t_hi = t_data.back();
      auto& [rho_out, t_out] = result;
      for (int i = 0; i <= n_pts; ++i) {
        double t = t_lo + (t_hi - t_lo) * i / n_pts;
        t_out.push_back(t);
        rho_out.push_back(spl(t));
      }
      return result;
    };

    constexpr int n_spline = 200;

    if (!cd.T.empty()) {
      // Raw data points
      plt::plot(cd.rho_v, cd.T, {{"color", "black"}, {"marker", "o"}, {"linestyle", "none"}, {"markersize", "3"}});
      plt::plot(cd.rho_l, cd.T, {{"color", "black"}, {"marker", "o"}, {"linestyle", "none"}, {"markersize", "3"}});

      // Spline-interpolated curves
      auto [rho_v_s, t_v_s] = spline_curve(cd.T, cd.rho_v, n_spline);
      auto [rho_l_s, t_l_s] = spline_curve(cd.T, cd.rho_l, n_spline);
      plt::named_plot("Binodal (vapor)", rho_v_s, t_v_s, "k-");
      plt::named_plot("Binodal (liquid)", rho_l_s, t_l_s, "k-");
    }

    if (!cd.T_sp.empty()) {
      // Raw data points
      plt::plot(cd.rho_s1, cd.T_sp,
                {{"color", "red"}, {"marker", "s"}, {"linestyle", "none"}, {"markersize", "3"}});
      plt::plot(cd.rho_s2, cd.T_sp,
                {{"color", "red"}, {"marker", "s"}, {"linestyle", "none"}, {"markersize", "3"}});

      // Spline-interpolated curves
      auto [rho_s1_s, t_s1_s] = spline_curve(cd.T_sp, cd.rho_s1, n_spline);
      auto [rho_s2_s, t_s2_s] = spline_curve(cd.T_sp, cd.rho_s2, n_spline);
      plt::named_plot("Spinodal (low)", rho_s1_s, t_s1_s, "r--");
      plt::named_plot("Spinodal (high)", rho_s2_s, t_s2_s, "r--");
    }

    if (cd.Tc > 0) {
      plt::plot({cd.rho_c}, {cd.Tc}, "ko");
    }

    plt::xlabel(R"($\rho \sigma^3$)");
    plt::ylabel(R"($T^* = k_BT / \epsilon$)");
    plt::title(R"(Binodal + spinodal: White Bear II)");
    plt::legend();
    plt::grid(true);
    plt::tight_layout();
    plt::save("exports/binodal_spinodal.png");
    plt::close();
    std::cout << "Plot: " << std::filesystem::absolute("exports/binodal_spinodal.png") << "\n";
  }

  // Plot 4: chemical potential vs density (mu-rho loops)
  {
    plt::figure_size(900, 600);
    std::vector<std::string> mu_styles = {"b-", "c-", "g-", "y-", "r-", "m-", "k--"};
    for (size_t t = 0; t < isotherm_temps.size(); ++t) {
      double kT = isotherm_temps[t];
      auto solver = make_lj_solver(dx, box, diameter, lj, kT,
                                   functional::fmt::WhiteBearII{});

      std::vector<double> rho_v, mu_v;
      for (int i = 0; i < n_rho; ++i) {
        double rho = rho_min + (rho_max_iso - rho_min) * (i + 1.0) / n_rho;
        double mu = solver.chemical_potential(rho) * kT;
        rho_v.push_back(rho);
        mu_v.push_back(mu);
      }
      char label[32];
      std::snprintf(label, sizeof(label), R"($T^* = %.2f$)", kT);
      plt::named_plot(label, rho_v, mu_v, mu_styles[t % mu_styles.size()]);
    }

    plt::xlim(0.0, rho_max_iso + 0.05);
    plt::xlabel(R"($\rho \sigma^3$)");
    plt::ylabel(R"($\mu^* = \mu / \epsilon$)");
    plt::title(R"(Chemical potential isotherms: LJ fluid (WB-II + mean field))");
    plt::legend();
    plt::grid(true);
    plt::tight_layout();
    plt::save("exports/chemical_potential.png");
    plt::close();
    std::cout << "Plot: " << std::filesystem::absolute("exports/chemical_potential.png") << "\n";
  }
#endif

  return 0;
}
