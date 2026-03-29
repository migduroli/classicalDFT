// ── Pair-potential example (modern cdft API) ────────────────────────────────
//
// Demonstrates:
//   - LennardJones, TenWoldeFrenkel, WRDF as variant PairPotential
//   - Cut-and-shift evaluation, WCA splitting
//   - Hard-sphere diameter and van der Waals integral
//   - Vectorized evaluation with arma::vec

#include <cdft.hpp>

#include <iomanip>
#include <iostream>

int main() {
  using namespace cdft::physics;

  // ── Create potentials ───────────────────────────────────────────────────

  auto lj = LennardJones(1.0, 1.0, 2.5);
  auto twf = TenWoldeFrenkel(1.0, 1.0, 2.5, 50.0);
  auto wrdf = WangRamirezDobnikarFrenkel(1.0, 1.0, 2.5);

  // ── Evaluate at a few distances ─────────────────────────────────────────

  std::cout << std::fixed << std::setprecision(8);
  std::cout << "Pair potentials V(r) with sigma=1, epsilon=1, r_cut=2.5\n";
  std::cout << std::string(70, '-') << "\n";
  std::cout << std::setw(8) << "r" << std::setw(18) << "LJ" << std::setw(18) << "tWF" << std::setw(18) << "WRDF\n";

  for (double r : {1.05, 1.12, 1.5, 2.0, 2.5}) {
    std::cout << std::setw(8) << r
              << std::setw(18) << lj.raw(r)
              << std::setw(18) << twf.raw(r)
              << std::setw(18) << wrdf.raw(r) << "\n";
  }

  // ── Variant dispatch with evaluate (cut-and-shift) ──────────────────────

  PairPotential pot = lj;

  std::cout << "\n\nCut-and-shifted LJ via variant dispatch\n";
  std::cout << std::string(40, '-') << "\n";
  std::cout << std::setw(8) << "r" << std::setw(18) << "V_cs(r)\n";

  for (double r : {0.9, 1.0, 1.12, 1.5, 2.0, 2.5, 3.0}) {
    std::cout << std::setw(8) << r << std::setw(18) << evaluate(pot, r) << "\n";
  }

  // ── WCA splitting ──────────────────────────────────────────────────────

  std::cout << "\n\nWCA splitting of LJ potential\n";
  std::cout << std::string(55, '-') << "\n";
  std::cout << std::setw(8) << "r" << std::setw(18) << "V_rep(r)" << std::setw(18) << "V_att(r)\n";

  for (double r : {1.05, 1.12, 1.5, 2.0, 2.5}) {
    std::cout << std::setw(8) << r
              << std::setw(18) << w_repulsive(pot, r)
              << std::setw(18) << w_attractive(pot, r) << "\n";
  }

  // ── Vectorized evaluation ───────────────────────────────────────────────

  auto r_vec = arma::linspace(0.9, 3.0, 50);
  auto v_vec = evaluate(pot, r_vec);

  std::cout << "\n\nVectorized evaluation: " << v_vec.n_elem << " points from r=0.9 to r=3.0\n";
  std::cout << "  V(r_min) = " << v_vec.front() << "\n";
  std::cout << "  V(r_max) = " << v_vec.back() << "\n";

  // ── Hard-sphere diameter and van der Waals integral ─────────────────────

  constexpr double kT = 1.2;
  PairPotential lj_bh = LennardJones(1.0, 1.0, 2.5);
  set_bh_perturbation(lj_bh);
  get_config(lj_bh).kT = kT;

  double d_hs = find_hard_sphere_diameter(lj_bh, kT);
  double a_vdw = compute_van_der_waals_integral(lj_bh, kT);

  std::cout << "\n\nBarker-Henderson perturbation theory (kT = " << kT << ")\n";
  std::cout << std::string(40, '-') << "\n";
  std::cout << "  Hard-sphere diameter : " << d_hs << "\n";
  std::cout << "  van der Waals a      : " << a_vdw << "\n";

  // ── Potential metadata ──────────────────────────────────────────────────

  std::cout << "\n\nPotential metadata\n";
  std::cout << std::string(40, '-') << "\n";

  for (const PairPotential& p : {PairPotential{lj}, PairPotential{twf}, PairPotential{wrdf}}) {
    auto& cfg = get_config(p);
    std::cout << "  " << potential_name(p)
              << " | sigma=" << cfg.sigma
              << " | epsilon=" << cfg.epsilon
              << " | r_cut=" << cfg.r_cutoff
              << " | hc_diam=" << hard_core_diameter(p) << "\n";
  }

  return 0;
}
