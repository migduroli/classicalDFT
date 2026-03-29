#pragma once

#include <armadillo>
#include <cmath>
#include <string>
#include <variant>

namespace cdft::physics {

  // ── Shared configuration ──────────────────────────────────────────────────

  struct PotentialConfig {
    double sigma = 1.0;
    double epsilon = 1.0;
    double r_cutoff = -1000.0;
    double r_min = 1.0;
    double v_min = -1.0;
    double r_zero = 1.0;
    double epsilon_shift = 0.0;
    double r_attractive_min = 0.0;
    bool bh_perturbation = false;
    double kT = 1.0;

    static constexpr double MAX_VALUE = 1e50;
  };

  // ── Concrete potential types (pure data + formula) ────────────────────────

  struct LennardJones {
    PotentialConfig config;

    LennardJones();
    LennardJones(double sigma, double epsilon, double r_cutoff);

    [[nodiscard]] double raw(double r) const;
    [[nodiscard]] double raw_r2(double r2) const;
    [[nodiscard]] static constexpr double hard_core_diameter() { return 0.0; }
    [[nodiscard]] static std::string name() { return "LennardJones"; }
  };

  struct TenWoldeFrenkel {
    PotentialConfig config;
    double alpha = 50.0;

    TenWoldeFrenkel();
    TenWoldeFrenkel(double sigma, double epsilon, double r_cutoff, double alpha = 50.0);

    [[nodiscard]] double raw(double r) const;
    [[nodiscard]] double raw_r2(double r2) const;
    [[nodiscard]] double hard_core_diameter() const { return config.sigma; }
    [[nodiscard]] static std::string name() { return "TenWoldeFrenkel"; }
  };

  struct WangRamirezDobnikarFrenkel {
    PotentialConfig config;

    WangRamirezDobnikarFrenkel();
    WangRamirezDobnikarFrenkel(double sigma, double epsilon, double r_cutoff);

    [[nodiscard]] double raw(double r) const;
    [[nodiscard]] double raw_r2(double r2) const;
    [[nodiscard]] static constexpr double hard_core_diameter() { return 0.0; }
    [[nodiscard]] static std::string name() { return "WangRamirezDobnikarFrenkel"; }
  };

  // ── Sum type (Python Union[LennardJones, TenWoldeFrenkel, WRDF]) ──────────

  using PairPotential = std::variant<LennardJones, TenWoldeFrenkel, WangRamirezDobnikarFrenkel>;

  // ── Free functions: operate on any potential type via std::visit ───────────

  [[nodiscard]] inline const PotentialConfig& get_config(const PairPotential& pot) {
    return std::visit([](const auto& p) -> const PotentialConfig& { return p.config; }, pot);
  }

  [[nodiscard]] inline PotentialConfig& get_config(PairPotential& pot) {
    return std::visit([](auto& p) -> PotentialConfig& { return p.config; }, pot);
  }

  [[nodiscard]] inline std::string potential_name(const PairPotential& pot) {
    return std::visit([](const auto& p) { return p.name(); }, pot);
  }

  [[nodiscard]] inline double hard_core_diameter(const PairPotential& pot) {
    return std::visit([](const auto& p) { return p.hard_core_diameter(); }, pot);
  }

  // ── Core evaluation (shared cut-and-shift logic) ──────────────────────────

  [[nodiscard]] double evaluate(const PairPotential& pot, double r);
  [[nodiscard]] arma::vec evaluate(const PairPotential& pot, const arma::vec& r);

  [[nodiscard]] double evaluate_r2(const PairPotential& pot, double r2);
  [[nodiscard]] arma::vec evaluate_r2(const PairPotential& pot, const arma::vec& r2);

  // ── WCA / BH splitting ────────────────────────────────────────────────────

  [[nodiscard]] double w_repulsive(const PairPotential& pot, double r);
  [[nodiscard]] arma::vec w_repulsive(const PairPotential& pot, const arma::vec& r);

  [[nodiscard]] double w_attractive(const PairPotential& pot, double r);
  [[nodiscard]] arma::vec w_attractive(const PairPotential& pot, const arma::vec& r);

  [[nodiscard]] double w_attractive_r2(const PairPotential& pot, double r2);

  // ── Configuration ─────────────────────────────────────────────────────────

  void set_wca_limit(PairPotential& pot, double r);
  void set_bh_perturbation(PairPotential& pot);

  // ── Integration (hard-sphere diameter, van der Waals parameter) ───────────

  [[nodiscard]] double find_hard_sphere_diameter(PairPotential& pot, double kT);
  [[nodiscard]] double compute_van_der_waals_integral(PairPotential& pot, double kT);

  // ── Identifier string ─────────────────────────────────────────────────────

  [[nodiscard]] std::string identifier(const PairPotential& pot);

}  // namespace cdft::physics
