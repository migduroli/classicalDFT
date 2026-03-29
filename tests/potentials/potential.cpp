#include "dft/potentials/potential.h"

#include <cmath>
#include <gtest/gtest.h>

using namespace dft::potentials;
using namespace dft::potentials;

// ── FakePotential for testing abstract base class ───────────────────────────

class FakePotential : public Potential {
 private:
  double vr_(double r) const override { return vr2_(r * r); }
  double vr2_(double r2) const override { return r2; }

 public:
  FakePotential() : Potential() {}
  FakePotential(double sigma, double epsilon, double r_cutoff) : Potential(sigma, epsilon, r_cutoff) {}
  double find_hard_core_diameter() const override { return 0; }
  double find_r_min() const override { return 0; }
};

// ── Potential base class (via FakePotential) ────────────────────────────────

TEST(Potential, DefaultConstructorInspectors) {
  auto v = FakePotential();
  EXPECT_DOUBLE_EQ(v.sigma(), DEFAULT_LENGTH_SCALE);
  EXPECT_DOUBLE_EQ(v.epsilon(), DEFAULT_ENERGY_SCALE);
  EXPECT_DOUBLE_EQ(v.epsilon_shift(), DEFAULT_ZERO);
  EXPECT_DOUBLE_EQ(v.r_cutoff(), DEFAULT_CUTOFF);
  EXPECT_DOUBLE_EQ(v.r_min(), DEFAULT_LENGTH_SCALE);
  EXPECT_DOUBLE_EQ(v.v_min(), -DEFAULT_ENERGY_SCALE);
  EXPECT_DOUBLE_EQ(v.r_attractive_min(), DEFAULT_ZERO);
  EXPECT_DOUBLE_EQ(v.r_zero(), DEFAULT_LENGTH_SCALE);
  EXPECT_FALSE(v.bh_perturbation());
  EXPECT_DOUBLE_EQ(v.kT(), DEFAULT_ENERGY_SCALE);
}

TEST(Potential, ParameterizedConstructorInspectors) {
  auto v = FakePotential(2.0, 3.0, 5.0);
  EXPECT_DOUBLE_EQ(v.sigma(), 2.0);
  EXPECT_DOUBLE_EQ(v.epsilon(), 3.0);
  EXPECT_DOUBLE_EQ(v.r_cutoff(), 5.0);
  EXPECT_DOUBLE_EQ(v.epsilon_shift(), DEFAULT_ZERO);
  EXPECT_DOUBLE_EQ(v.r_min(), DEFAULT_ZERO);
  EXPECT_DOUBLE_EQ(v.v_min(), DEFAULT_ZERO);
  EXPECT_DOUBLE_EQ(v.r_attractive_min(), DEFAULT_ZERO);
  EXPECT_DOUBLE_EQ(v.r_zero(), DEFAULT_LENGTH_SCALE);
  EXPECT_FALSE(v.bh_perturbation());
  EXPECT_DOUBLE_EQ(v.kT(), DEFAULT_ENERGY_SCALE);
}

TEST(Potential, VPotentialScalar) {
  auto v = FakePotential();
  EXPECT_DOUBLE_EQ(v.v_potential(0.5), 0.25);
}

TEST(Potential, VPotentialVec) {
  auto v = FakePotential();
  auto result = v.v_potential(arma::vec{0.5, 1.0});
  EXPECT_DOUBLE_EQ(result(0), 0.25);
  EXPECT_DOUBLE_EQ(result(1), 1.0);
}

TEST(Potential, VPotentialR2Scalar) {
  auto v = FakePotential();
  EXPECT_DOUBLE_EQ(v.v_potential_r2(0.25), 0.25);
}

TEST(Potential, VPotentialR2Vec) {
  auto v = FakePotential();
  auto result = v.v_potential_r2(arma::vec{0.25, 1.0});
  EXPECT_DOUBLE_EQ(result(0), 0.25);
  EXPECT_DOUBLE_EQ(result(1), 1.0);
}

TEST(Potential, WRepulsiveScalar) {
  auto v = FakePotential();
  EXPECT_DOUBLE_EQ(v.w_repulsive(0), DEFAULT_ENERGY_SCALE);
  EXPECT_DOUBLE_EQ(v.w_repulsive(0.5), 1.25);
}

TEST(Potential, WRepulsiveVec) {
  auto v = FakePotential();
  auto result = v.w_repulsive(arma::vec{0.0, 0.5});
  EXPECT_DOUBLE_EQ(result(0), DEFAULT_ENERGY_SCALE);
  EXPECT_DOUBLE_EQ(result(1), 1.25);
}

TEST(Potential, WAttractiveScalar) {
  auto v = FakePotential();
  EXPECT_DOUBLE_EQ(v.w_attractive(0), -DEFAULT_ENERGY_SCALE);
  EXPECT_DOUBLE_EQ(v.w_attractive(0.5), -DEFAULT_ENERGY_SCALE);
}

TEST(Potential, WAttractiveVec) {
  auto v = FakePotential();
  auto result = v.w_attractive(arma::vec{0.0, 0.5});
  EXPECT_DOUBLE_EQ(result(0), -DEFAULT_ENERGY_SCALE);
}

TEST(Potential, WAttractiveR2) {
  auto v = FakePotential();
  EXPECT_DOUBLE_EQ(v.w_attractive_r2(0.0), -DEFAULT_ENERGY_SCALE);
}

TEST(Potential, SetWCALimit) {
  auto v = FakePotential();
  v.set_wca_limit(0.5);
  EXPECT_DOUBLE_EQ(v.r_attractive_min(), 0.5);
}

TEST(Potential, SetBHPerturbation) {
  auto v = FakePotential();
  v.set_bh_perturbation();
  EXPECT_TRUE(v.bh_perturbation());
  EXPECT_DOUBLE_EQ(v.r_attractive_min(), DEFAULT_LENGTH_SCALE);
  EXPECT_DOUBLE_EQ(v.w_repulsive(0), DEFAULT_ZERO);
  EXPECT_DOUBLE_EQ(v.w_repulsive(0.5), 0.25);
}

TEST(Potential, FindHardSphereDiameter) {
  auto v = FakePotential();
  v.set_bh_perturbation();
  EXPECT_DOUBLE_EQ(v.find_hard_sphere_diameter(0), DEFAULT_LENGTH_SCALE);
  EXPECT_DOUBLE_EQ(v.find_hard_sphere_diameter(0.1), 0.719752609493357);
}

TEST(Potential, ComputeVanDerWaalsIntegralBH) {
  auto v = FakePotential();
  v.set_bh_perturbation();
  auto vdw = v.compute_van_der_waals_integral(0.5);
  EXPECT_DOUBLE_EQ(vdw, 0.0);
}

TEST(Potential, WAttractiveR2InsideWCALimit) {
  auto v = FakePotential(1.0, 1.0, 5.0);
  v.set_wca_limit(2.0);
  // r_squared = 1.0 < r_attractive_min^2 = 4.0, inside WCA limit → returns 0
  EXPECT_DOUBLE_EQ(v.w_attractive_r2(1.0), 0.0);
}

TEST(Potential, WAttractiveR2BeyondCutoff) {
  auto v = FakePotential(1.0, 1.0, 2.0);
  // r_squared = 10.0 > r_cutoff^2 = 4.0 → returns 0
  EXPECT_DOUBLE_EQ(v.w_attractive_r2(10.0), 0.0);
}

TEST(Potential, OperatorBrackets) {
  auto v = FakePotential();
  EXPECT_DOUBLE_EQ(v(0.5), v.v_potential(0.5));
  auto arma_result = v(arma::vec{0.5, 1.0});
  EXPECT_DOUBLE_EQ(arma_result(0), v.v_potential(0.5));
}

// ── Identifier ──────────────────────────────────────────────────────────────

TEST(Potential, IdentifierUnknownPotential) {
  // FakePotential has potential_id_ default-initialized to LennardJones(0).
  // Override to an out-of-range value to hit the "Unknown" branch.
  class UnknownPotential : public FakePotential {
   public:
    UnknownPotential() { potential_id_ = static_cast<PotentialName>(99); }
  };
  auto u = UnknownPotential();
  EXPECT_NE(u.identifier().find("Unknown"), std::string::npos);
}

TEST(Potential, IdentifierContainsPotentialName) {
  auto lj = LennardJones();
  auto id = lj.identifier();
  EXPECT_NE(id.find("LennardJones"), std::string::npos);
}

TEST(Potential, IdentifierForTenWoldeFrenkel) {
  auto twf = tenWoldeFrenkel();
  auto id = twf.identifier();
  EXPECT_NE(id.find("tenWoldeFrenkel"), std::string::npos);
}

TEST(Potential, IdentifierForWRDF) {
  auto wrdf = WangRamirezDobnikarFrenkel();
  auto id = wrdf.identifier();
  EXPECT_NE(id.find("WangRamirezDobnikarFrenkel"), std::string::npos);
}

// ── LennardJones ────────────────────────────────────────────────────────────

TEST(LennardJones, DefaultConstructorProperties) {
  auto lj = LennardJones();
  EXPECT_EQ(lj.id(), PotentialName::LennardJones);
  EXPECT_NEAR(lj.r_min(), std::pow(2.0, 1.0 / 6.0), 1e-12);
  EXPECT_LT(lj.v_min(), 0.0);
  EXPECT_GT(lj.r_zero(), 0.0);
  EXPECT_DOUBLE_EQ(lj.find_hard_core_diameter(), 0.0);
}

TEST(LennardJones, ParameterizedConstructor) {
  auto lj = LennardJones(1.0, 1.0, 2.5);
  EXPECT_DOUBLE_EQ(lj.sigma(), 1.0);
  EXPECT_DOUBLE_EQ(lj.epsilon(), 1.0);
  EXPECT_DOUBLE_EQ(lj.r_cutoff(), 2.5);
  EXPECT_NE(lj.epsilon_shift(), 0.0);
}

TEST(LennardJones, PotentialAtSigma) {
  auto lj = LennardJones();
  // V(sigma) = 4*eps*(1 - 1) = 0
  EXPECT_NEAR(lj.v_potential(1.0), 0.0, 1e-10);
}

TEST(LennardJones, WCASplitPreservesTotal) {
  auto lj = LennardJones();
  auto x = arma::linspace(0.5, 5, 100);
  auto expected = lj(x);
  arma::vec actual = lj.w_attractive(x) + lj.w_repulsive(x);
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_NEAR(expected(i), actual(i), 1e-6);
  }
}

TEST(LennardJones, WCASplitWithBHPreservesTotal) {
  auto lj = LennardJones();
  lj.set_bh_perturbation();
  auto x = arma::linspace(0.5, 5, 100);
  auto expected = lj(x);
  arma::vec actual = lj.w_attractive(x) + lj.w_repulsive(x);
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_NEAR(expected(i), actual(i), 1e-6);
  }
}

TEST(LennardJones, RepulsivePartNonBH) {
  auto lj = LennardJones();
  double r = 0.9 * lj.r_min();
  EXPECT_DOUBLE_EQ(lj.w_repulsive(r), lj(r) - lj.v_min());
  EXPECT_DOUBLE_EQ(lj.w_repulsive(1.1 * lj.r_min()), 0.0);
}

TEST(LennardJones, RepulsivePartBH) {
  auto lj = LennardJones();
  lj.set_bh_perturbation();
  double r = 0.999 * lj.r_zero();
  EXPECT_DOUBLE_EQ(lj.w_repulsive(r), lj(r));
  EXPECT_DOUBLE_EQ(lj.w_repulsive(1.001 * lj.r_zero()), 0.0);
}

TEST(LennardJones, HardSphereDiameter) {
  auto lj = LennardJones();
  auto d = lj.find_hard_sphere_diameter(1.0);
  EXPECT_GT(d, 0.0);
  EXPECT_LT(d, lj.r_min());
}

TEST(LennardJones, VanDerWaalsIntegralNonBH) {
  auto lj = LennardJones();
  auto vdw = lj.compute_van_der_waals_integral(1.0);
  EXPECT_TRUE(std::isfinite(vdw));
}

TEST(LennardJones, VanDerWaalsIntegralBH) {
  auto lj = LennardJones(1.0, 1.0, 2.5);
  lj.set_bh_perturbation();
  auto vdw = lj.compute_van_der_waals_integral(1.0);
  EXPECT_LT(vdw, 0.0);
}

// ── tenWoldeFrenkel ─────────────────────────────────────────────────────────

TEST(TenWoldeFrenkel, DefaultConstructorProperties) {
  auto twf = tenWoldeFrenkel();
  EXPECT_EQ(twf.id(), PotentialName::tenWoldeFrenkel);
  EXPECT_DOUBLE_EQ(twf.alpha(), DEFAULT_ALPHA_PARAMETER);
  EXPECT_GT(twf.r_min(), twf.sigma());
  EXPECT_LT(twf.v_min(), 0.0);
  EXPECT_DOUBLE_EQ(twf.find_hard_core_diameter(), twf.sigma());
}

TEST(TenWoldeFrenkel, ParameterizedConstructor) {
  auto twf = tenWoldeFrenkel(1.0, 1.0, 3.0, 25.0);
  EXPECT_DOUBLE_EQ(twf.sigma(), 1.0);
  EXPECT_DOUBLE_EQ(twf.epsilon(), 1.0);
  EXPECT_DOUBLE_EQ(twf.r_cutoff(), 3.0);
  EXPECT_DOUBLE_EQ(twf.alpha(), 25.0);
}

TEST(TenWoldeFrenkel, PotentialInsideSigmaIsLarge) {
  auto twf = tenWoldeFrenkel();
  EXPECT_DOUBLE_EQ(twf.v_potential(0.5 * twf.sigma()), MAX_POTENTIAL_VALUE);
}

TEST(TenWoldeFrenkel, PotentialR2InsideSigmaIsLarge) {
  auto twf = tenWoldeFrenkel();
  double r = 0.5 * twf.sigma();
  EXPECT_DOUBLE_EQ(twf.v_potential_r2(r * r), MAX_POTENTIAL_VALUE);
}

TEST(TenWoldeFrenkel, AttractivePartAtMinimum) {
  auto twf = tenWoldeFrenkel();
  EXPECT_DOUBLE_EQ(twf.w_attractive(twf.r_min()), twf.v_min());
}

TEST(TenWoldeFrenkel, AttractivePartBH) {
  auto twf = tenWoldeFrenkel();
  twf.set_bh_perturbation();
  EXPECT_DOUBLE_EQ(twf.w_attractive(0.0), 0.0);
  EXPECT_DOUBLE_EQ(twf.w_attractive(0.999 * twf.r_zero()), 0.0);
  EXPECT_DOUBLE_EQ(twf.w_attractive(twf.r_min()), twf.v_min());
}

TEST(TenWoldeFrenkel, FindRMin) {
  auto twf = tenWoldeFrenkel();
  EXPECT_NEAR(twf.find_r_min(), twf.r_min(), 1e-12);
}

// ── WangRamirezDobnikarFrenkel ──────────────────────────────────────────────

TEST(WangRamirezDobnikarFrenkel, DefaultConstructorProperties) {
  auto wrdf = WangRamirezDobnikarFrenkel();
  EXPECT_EQ(wrdf.id(), PotentialName::WangRamirezDobnikarFrenkel);
  EXPECT_GT(wrdf.r_min(), 0.0);
  EXPECT_LT(wrdf.v_min(), 0.0);
  EXPECT_DOUBLE_EQ(wrdf.r_zero(), 1.0);
  EXPECT_DOUBLE_EQ(wrdf.epsilon_shift(), 0.0);
  EXPECT_DOUBLE_EQ(wrdf.find_hard_core_diameter(), 0.0);
}

TEST(WangRamirezDobnikarFrenkel, ParameterizedConstructor) {
  auto wrdf = WangRamirezDobnikarFrenkel(1.0, 1.0, 3.0);
  EXPECT_DOUBLE_EQ(wrdf.sigma(), 1.0);
  EXPECT_DOUBLE_EQ(wrdf.r_cutoff(), 3.0);
  EXPECT_DOUBLE_EQ(wrdf.r_zero(), 1.0);
}

TEST(WangRamirezDobnikarFrenkel, PotentialBeyondCutoffIsZero) {
  auto wrdf = WangRamirezDobnikarFrenkel();
  EXPECT_DOUBLE_EQ(wrdf.v_potential(wrdf.r_cutoff() + 0.1), 0.0);
}

TEST(WangRamirezDobnikarFrenkel, PotentialR2BeyondCutoffIsZero) {
  auto wrdf = WangRamirezDobnikarFrenkel();
  double r = wrdf.r_cutoff() + 0.1;
  EXPECT_DOUBLE_EQ(wrdf.v_potential_r2(r * r), 0.0);
}

TEST(WangRamirezDobnikarFrenkel, FindRMin) {
  auto wrdf = WangRamirezDobnikarFrenkel();
  EXPECT_NEAR(wrdf.find_r_min(), wrdf.r_min(), 1e-12);
}

TEST(WangRamirezDobnikarFrenkel, HardSphereDiameter) {
  auto wrdf = WangRamirezDobnikarFrenkel();
  auto d = wrdf.find_hard_sphere_diameter(1.0);
  EXPECT_GT(d, 0.0);
}
