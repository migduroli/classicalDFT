#include "classicaldft_bits/physics/potentials/intermolecular/potential.h"

#include <cmath>
#include <gtest/gtest.h>

using namespace dft_core::physics::potentials;
using namespace dft_core::physics::potentials::intermolecular;

// region Cttors:

/**
 * The class FakePotential is required to test the abstract class Potential.
 * This class is a trivial inheritor of the Potential class so that we can test the
 * underlying functionality of the mother class.
 */
class FakePotential : public intermolecular::Potential {
 private:
  double vr_(double r) const override { return vr2_(r * r); }
  double vr2_(double r2) const override { return r2; }

 public:
  FakePotential() : intermolecular::Potential() {}
  FakePotential(double sigma, double epsilon, double r_cutoff) : Potential(sigma, epsilon, r_cutoff) {}
  double find_hard_core_diameter() const override { return 0; }
  double find_r_min() const override { return 0; }
};

TEST(intermolecular_potential, potential_cttor_works_ok) {
  auto v_test = FakePotential();
  EXPECT_DOUBLE_EQ(v_test.sigma(), DEFAULT_LENGTH_SCALE);
  EXPECT_DOUBLE_EQ(v_test.epsilon(), DEFAULT_ENERGY_SCALE);
  EXPECT_DOUBLE_EQ(v_test.epsilon_shift(), DEFAULT_ZERO);
  EXPECT_DOUBLE_EQ(v_test.r_cutoff(), DEFAULT_CUTOFF);
  EXPECT_DOUBLE_EQ(v_test.r_min(), DEFAULT_LENGTH_SCALE);
  EXPECT_DOUBLE_EQ(v_test.v_min(), -DEFAULT_ENERGY_SCALE);
  EXPECT_DOUBLE_EQ(v_test.r_attractive_min(), -DEFAULT_ZERO);
  EXPECT_DOUBLE_EQ(v_test.r_zero(), DEFAULT_LENGTH_SCALE);
  EXPECT_DOUBLE_EQ(v_test.bh_perturbation(), false);
  EXPECT_DOUBLE_EQ(v_test.kT(), DEFAULT_ENERGY_SCALE);

  EXPECT_DOUBLE_EQ(v_test.w_repulsive(0), DEFAULT_ENERGY_SCALE);
  EXPECT_DOUBLE_EQ(v_test.w_repulsive(0.5), 1.25);
  EXPECT_DOUBLE_EQ(v_test.w_attractive(0), -DEFAULT_ENERGY_SCALE);
  EXPECT_DOUBLE_EQ(v_test.w_attractive(0.5), -DEFAULT_ENERGY_SCALE);

  EXPECT_DOUBLE_EQ(v_test.v_potential(0.5), 0.25);

  v_test.set_bh_perturbation();
  EXPECT_DOUBLE_EQ(v_test.r_attractive_min(), DEFAULT_LENGTH_SCALE);
  EXPECT_DOUBLE_EQ(v_test.w_repulsive(0), DEFAULT_ZERO);
  EXPECT_DOUBLE_EQ(v_test.w_repulsive(0.5), 0.25);

  v_test.set_wca_limit(0.5);
  EXPECT_DOUBLE_EQ(v_test.r_attractive_min(), 0.5);

  auto d_hs = v_test.find_hard_sphere_diameter(0);
  EXPECT_DOUBLE_EQ(d_hs, DEFAULT_LENGTH_SCALE);
  d_hs = v_test.find_hard_sphere_diameter(0.1);
  EXPECT_DOUBLE_EQ(d_hs, 0.719752609493357);

  auto vdw = v_test.compute_van_der_waals_integral(0.5);
  EXPECT_DOUBLE_EQ(vdw, 0);
}

TEST(intermolecular_potential, potential_brackets_works) {
  auto lj = LennardJones();
  auto d_expected = lj.v_potential(1);
  auto d_actual = lj(1);
  EXPECT_DOUBLE_EQ(d_expected, d_actual);

  auto x = std::vector<double>{1, 2};
  auto vec_expected = lj.v_potential(x);
  auto vec_actual = lj(x);

  for (size_t i = 0; i < vec_expected.size(); ++i) {
    EXPECT_DOUBLE_EQ(vec_expected[i], vec_actual[i]);
  }

  auto r = arma::linspace(1, 5, 10);
  auto arma_expected = lj.v_potential(r);
  auto arma_actual = lj(r);

  for (size_t i = 0; i < arma_expected.size(); ++i) {
    EXPECT_DOUBLE_EQ(arma_expected(i), arma_actual(i));
  }
}

TEST(intermolecular_potential, potential_attractive_part_ok) {
  auto twf = tenWoldeFrenkel();
  auto v_actual = twf.w_attractive(0);
  EXPECT_DOUBLE_EQ(twf.v_min(), v_actual);

  v_actual = twf.w_attractive(twf.r_min());
  EXPECT_DOUBLE_EQ(twf.v_min(), v_actual);

  v_actual = twf.w_attractive(2 * twf.r_min());
  EXPECT_DOUBLE_EQ(twf(2 * twf.r_min()), v_actual);
}

TEST(intermolecular_potential, potential_attractive_part_bh_ok) {
  auto twf = tenWoldeFrenkel();
  twf.set_bh_perturbation();

  auto v_actual = twf.w_attractive(0.0);
  EXPECT_DOUBLE_EQ(0.0, v_actual);

  v_actual = twf.w_attractive(0.999 * twf.r_zero());
  EXPECT_DOUBLE_EQ(0.0, v_actual);

  v_actual = twf.w_attractive(1.001 * twf.r_zero());
  EXPECT_DOUBLE_EQ(twf(1.001 * twf.r_zero()), v_actual);

  v_actual = twf.w_attractive(twf.r_min());
  EXPECT_DOUBLE_EQ(twf.v_min(), v_actual);

  v_actual = twf.w_attractive(2 * twf.r_min());
  EXPECT_DOUBLE_EQ(twf(2 * twf.r_min()), v_actual);
}

TEST(intermolecular_potential, potential_repulsive_part_ok) {
  auto lj = LennardJones();
  auto r = 0.9 * lj.r_min();

  auto actual = lj.w_repulsive(r);
  auto expected = lj(r) - lj.v_min();
  EXPECT_DOUBLE_EQ(expected, actual);

  r = 1.1 * lj.r_min();
  actual = lj.w_repulsive(r);
  expected = 0.0;
  EXPECT_DOUBLE_EQ(expected, actual);

  lj.set_bh_perturbation();

  r = 0.999 * lj.r_zero();
  actual = lj.w_repulsive(r);
  expected = lj(r);
  EXPECT_DOUBLE_EQ(expected, actual);

  r = 1.001 * lj.r_zero();
  actual = lj.w_repulsive(r);
  expected = 0.0;
  EXPECT_DOUBLE_EQ(expected, actual);
}

TEST(intermolecular_potential, WCA_split_works_ok) {
  auto lj = LennardJones();
  auto x = arma::linspace(0.5, 5, 100);
  auto expected = lj(x);
  arma::vec actual = lj.w_attractive(x) + lj.w_repulsive(x);

  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_NEAR(expected(i), actual(i), 1e-6);
  }
}

TEST(intermolecular_potential, WCA_split_works_with_BH_ok) {
  auto lj = LennardJones();
  lj.set_bh_perturbation();

  auto x = arma::linspace(0.5, 5, 100);
  auto expected = lj(x);
  arma::vec actual = lj.w_attractive(x) + lj.w_repulsive(x);

  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_NEAR(expected(i), actual(i), 1e-6);
  }
}
// endregion