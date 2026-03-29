#include "cdft/physics/potentials.hpp"

#include "cdft/numerics/math.hpp"

#include <cmath>

namespace cdft::physics {

  // ── LennardJones ──────────────────────────────────────────────────────────

  LennardJones::LennardJones() {
    config.epsilon_shift = (config.r_cutoff < 0 ? 0.0 : raw(config.r_cutoff));
    config.r_min = std::pow(2.0, 1.0 / 6.0) * config.sigma;
    config.v_min = raw(config.r_min) - config.epsilon_shift;
    config.r_zero = std::pow(0.5 * std::sqrt(1 + config.epsilon_shift) + 0.5, -1.0 / 6.0);
  }

  LennardJones::LennardJones(double sigma, double epsilon, double r_cutoff) {
    config.sigma = sigma;
    config.epsilon = epsilon;
    config.r_cutoff = r_cutoff;
    config.epsilon_shift = (r_cutoff < 0 ? 0.0 : raw(r_cutoff));
    config.r_min = std::pow(2.0, 1.0 / 6.0) * sigma;
    config.v_min = raw(config.r_min) - config.epsilon_shift;
    config.r_zero = std::pow(0.5 * std::sqrt(1 + config.epsilon_shift) + 0.5, -1.0 / 6.0);
  }

  double LennardJones::raw(double r) const {
    double y = config.sigma / r;
    double y6 = y * y * y * y * y * y;
    return 4.0 * config.epsilon * (y6 * y6 - y6);
  }

  double LennardJones::raw_r2(double r2) const {
    double y2 = config.sigma * config.sigma / r2;
    double y6 = y2 * y2 * y2;
    return 4.0 * config.epsilon * (y6 * y6 - y6);
  }

  // ── TenWoldeFrenkel ───────────────────────────────────────────────────────

  TenWoldeFrenkel::TenWoldeFrenkel() {
    config.epsilon_shift = (config.r_cutoff <= 0.0 ? 0.0 : raw(config.r_cutoff));
    config.r_min = config.sigma * std::sqrt(1.0 + std::pow(2.0 / alpha, 1.0 / 3.0));
    config.v_min = raw(config.r_min) - config.epsilon_shift;
    config.r_zero = std::sqrt(1.0 + std::pow(25.0 * std::sqrt(1.0 + config.epsilon_shift) + 25.0, -1.0 / 3.0));
  }

  TenWoldeFrenkel::TenWoldeFrenkel(double sigma, double epsilon, double r_cutoff, double alpha)
      : alpha(alpha) {
    config.sigma = sigma;
    config.epsilon = epsilon;
    config.r_cutoff = r_cutoff;
    config.epsilon_shift = (r_cutoff <= 0.0 ? 0.0 : raw(r_cutoff));
    config.r_min = sigma * std::sqrt(1.0 + std::pow(2.0 / alpha, 1.0 / 3.0));
    config.v_min = raw(config.r_min) - config.epsilon_shift;
    config.r_zero = std::sqrt(1.0 + std::pow(25.0 * std::sqrt(1.0 + config.epsilon_shift) + 25.0, -1.0 / 3.0));
  }

  double TenWoldeFrenkel::raw(double r) const {
    if (r < config.sigma) return PotentialConfig::MAX_VALUE;
    double s = r / config.sigma;
    double y = 1.0 / (s * s - 1.0);
    double y3 = y * y * y;
    return (4.0 * config.epsilon / (alpha * alpha)) * (y3 * y3 - alpha * y3);
  }

  double TenWoldeFrenkel::raw_r2(double r2) const {
    if (r2 < config.sigma * config.sigma) return PotentialConfig::MAX_VALUE;
    double s2 = r2 / (config.sigma * config.sigma);
    double y = 1.0 / (s2 - 1.0);
    double y3 = y * y * y;
    return (4.0 * config.epsilon / (alpha * alpha)) * (y3 * y3 - alpha * y3);
  }

  // ── WangRamirezDobnikarFrenkel ────────────────────────────────────────────

  WangRamirezDobnikarFrenkel::WangRamirezDobnikarFrenkel() {
    config.r_cutoff = 3.0;
    config.epsilon *= 2.0 * std::pow(config.r_cutoff / config.sigma, 2) *
        std::pow(2.0 * ((config.r_cutoff / config.sigma) * (config.r_cutoff / config.sigma) - 1.0) / 3.0, -3.0);
    config.epsilon_shift = 0.0;
    config.r_min = config.r_cutoff *
        std::pow((1.0 + 2.0 * (config.r_cutoff / config.sigma) * (config.r_cutoff / config.sigma)) / 3.0, -0.5);
    config.v_min = raw(config.r_min);
    config.r_zero = 1.0;
  }

  WangRamirezDobnikarFrenkel::WangRamirezDobnikarFrenkel(double sigma, double epsilon, double r_cutoff) {
    config.sigma = sigma;
    config.epsilon = epsilon;
    config.r_cutoff = r_cutoff;
    config.epsilon *= 2.0 * std::pow(r_cutoff / sigma, 2) *
        std::pow(2.0 * ((r_cutoff / sigma) * (r_cutoff / sigma) - 1.0) / 3.0, -3.0);
    config.epsilon_shift = 0.0;
    config.r_min = r_cutoff *
        std::pow((1.0 + 2.0 * (r_cutoff / sigma) * (r_cutoff / sigma)) / 3.0, -0.5);
    config.v_min = raw(config.r_min);
    config.r_zero = 1.0;
  }

  double WangRamirezDobnikarFrenkel::raw(double r) const {
    double y = config.sigma / r;
    double z = config.r_cutoff / r;
    return (r < config.r_cutoff) ? config.epsilon * (y * y - 1.0) * (z * z - 1.0) * (z * z - 1.0) : 0.0;
  }

  double WangRamirezDobnikarFrenkel::raw_r2(double r2) const {
    double y = config.sigma * config.sigma / r2;
    double z = config.r_cutoff * config.r_cutoff / r2;
    return (r2 < config.r_cutoff * config.r_cutoff) ? config.epsilon * (y - 1.0) * (z - 1.0) * (z - 1.0) : 0.0;
  }

  // ── Free functions: core evaluation (shared cut-and-shift) ────────────────

  namespace {
    template <typename P>
    double evaluate_impl(const P& pot, double r) {
      return pot.raw(r) - pot.config.epsilon_shift;
    }

    template <typename P>
    double evaluate_r2_impl(const P& pot, double r2) {
      return pot.raw_r2(r2) - pot.config.epsilon_shift;
    }

    template <typename P>
    double w_repulsive_impl(const P& pot, double r) {
      const auto& c = pot.config;
      if (c.bh_perturbation) {
        return (r < c.r_zero ? evaluate_impl(pot, r) : 0.0);
      }
      return (r < c.r_min ? evaluate_impl(pot, r) - c.v_min : 0.0);
    }

    template <typename P>
    double w_attractive_r2_impl(const P& pot, double r2) {
      const auto& c = pot.config;
      double r_cut2 = c.r_cutoff * c.r_cutoff;
      if (r_cut2 > 0.0 && r2 >= r_cut2) return 0.0;

      if (c.bh_perturbation) {
        return (r2 < c.r_zero * c.r_zero ? 0.0 : evaluate_r2_impl(pot, r2));
      }
      if (r2 < c.r_attractive_min * c.r_attractive_min) return 0.0;
      if (r2 < c.r_min * c.r_min) return c.v_min;
      return evaluate_r2_impl(pot, r2);
    }

    template <typename P>
    double w_attractive_impl(const P& pot, double r) {
      return w_attractive_r2_impl(pot, r * r);
    }
  }  // namespace

  double evaluate(const PairPotential& pot, double r) {
    return std::visit([r](const auto& p) { return evaluate_impl(p, r); }, pot);
  }

  arma::vec evaluate(const PairPotential& pot, const arma::vec& r) {
    arma::vec result = r;
    result.transform([&pot](double ri) { return evaluate(pot, ri); });
    return result;
  }

  double evaluate_r2(const PairPotential& pot, double r2) {
    return std::visit([r2](const auto& p) { return evaluate_r2_impl(p, r2); }, pot);
  }

  arma::vec evaluate_r2(const PairPotential& pot, const arma::vec& r2) {
    arma::vec result = r2;
    result.transform([&pot](double ri) { return evaluate_r2(pot, ri); });
    return result;
  }

  double w_repulsive(const PairPotential& pot, double r) {
    return std::visit([r](const auto& p) { return w_repulsive_impl(p, r); }, pot);
  }

  arma::vec w_repulsive(const PairPotential& pot, const arma::vec& r) {
    arma::vec result = r;
    result.transform([&pot](double ri) { return w_repulsive(pot, ri); });
    return result;
  }

  double w_attractive(const PairPotential& pot, double r) {
    return std::visit([r](const auto& p) { return w_attractive_impl(p, r); }, pot);
  }

  arma::vec w_attractive(const PairPotential& pot, const arma::vec& r) {
    arma::vec result = r;
    result.transform([&pot](double ri) { return w_attractive(pot, ri); });
    return result;
  }

  double w_attractive_r2(const PairPotential& pot, double r2) {
    return std::visit([r2](const auto& p) { return w_attractive_r2_impl(p, r2); }, pot);
  }

  void set_wca_limit(PairPotential& pot, double r) {
    get_config(pot).r_attractive_min = r;
  }

  void set_bh_perturbation(PairPotential& pot) {
    auto& c = get_config(pot);
    c.bh_perturbation = true;
    c.r_attractive_min = c.r_zero;
  }

  double find_hard_sphere_diameter(PairPotential& pot, double kT) {
    auto& c = get_config(pot);
    c.kT = kT;
    double d_hc = hard_core_diameter(pot);
    double r_limit = c.bh_perturbation ? c.r_zero : c.r_min;

    auto kernel = [&pot, kT](double r) {
      return 1.0 - std::exp(-w_repulsive(pot, r) / kT);
    };

    auto result = numerics::integrate_qags(kernel, d_hc, r_limit, 1e-4, 1e-6);
    return d_hc + result.value;
  }

  double compute_van_der_waals_integral(PairPotential& pot, double kT) {
    auto& c = get_config(pot);
    c.kT = kT;
    double prefactor = 2.0 * M_PI / kT;

    auto kernel = [&pot](double r) {
      return r * r * w_attractive(pot, r);
    };

    double limit_superior = std::max(c.r_cutoff, 0.0);

    double integral;
    if (c.bh_perturbation) {
      integral = numerics::integrate_qags(kernel, c.r_zero, limit_superior, 1e-6, 1e-8).value;
    } else {
      integral = numerics::integrate_qags(kernel, c.r_attractive_min, limit_superior, 1e-6, 1e-8).value
                 + numerics::integrate_qags(kernel, c.r_min, limit_superior, 1e-6, 1e-8).value;
    }

    return prefactor * integral;
  }

  std::string identifier(const PairPotential& pot) {
    const auto& c = get_config(pot);
    return potential_name(pot) + "_" + std::to_string(c.sigma) + "_" + std::to_string(c.epsilon) + "_" +
           std::to_string(c.r_cutoff) + "_" + std::to_string(c.r_attractive_min) + "_" +
           std::to_string(c.bh_perturbation);
  }

}  // namespace cdft::physics
