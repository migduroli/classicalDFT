#include "cdft/physics/potentials.hpp"

#include <gtest/gtest.h>

#include <cmath>

namespace cdft::physics {

  TEST(LennardJonesTest, DefaultConstruction) {
    LennardJones lj;
    EXPECT_DOUBLE_EQ(lj.config.sigma, 1.0);
    EXPECT_DOUBLE_EQ(lj.config.epsilon, 1.0);
  }

  TEST(LennardJonesTest, RawAtSigma) {
    LennardJones lj;
    double v = lj.raw(1.0);
    EXPECT_DOUBLE_EQ(v, 0.0);
  }

  TEST(LennardJonesTest, RawAtMinimum) {
    LennardJones lj;
    double r_min = std::pow(2.0, 1.0 / 6.0);
    double v = lj.raw(r_min);
    EXPECT_NEAR(v, -1.0, 1e-12);
  }

  TEST(LennardJonesTest, EvaluateWithShift) {
    LennardJones lj(1.0, 1.0, 2.5);
    PairPotential pot = lj;
    double v_at_cutoff = evaluate(pot, 2.5);
    EXPECT_NEAR(v_at_cutoff, 0.0, 1e-12);
  }

  TEST(LennardJonesTest, VariantDispatch) {
    PairPotential pot = LennardJones(1.0, 1.0, 2.5);
    EXPECT_EQ(potential_name(pot), "LennardJones");
    EXPECT_DOUBLE_EQ(hard_core_diameter(pot), 0.0);
  }

  TEST(TenWoldeFrenkelTest, NameAndHardCore) {
    TenWoldeFrenkel twf(1.0, 1.0, 2.5);
    PairPotential pot = twf;
    EXPECT_EQ(potential_name(pot), "TenWoldeFrenkel");
    EXPECT_DOUBLE_EQ(hard_core_diameter(pot), 1.0);
  }

  TEST(WRDFTest, NameAndHardCore) {
    WangRamirezDobnikarFrenkel wrdf;
    PairPotential pot = wrdf;
    EXPECT_EQ(potential_name(pot), "WangRamirezDobnikarFrenkel");
    EXPECT_DOUBLE_EQ(hard_core_diameter(pot), 0.0);
  }

  TEST(PotentialSplitTest, RepulsiveZeroOutsideRmin) {
    PairPotential pot = LennardJones(1.0, 1.0, 2.5);
    double wr = w_repulsive(pot, 2.0);
    EXPECT_DOUBLE_EQ(wr, 0.0);
  }

  TEST(PotentialSplitTest, AttractiveZeroOutsideCutoff) {
    PairPotential pot = LennardJones(1.0, 1.0, 2.5);
    double wa = w_attractive(pot, 3.0);
    EXPECT_DOUBLE_EQ(wa, 0.0);
  }

  TEST(PotentialSplitTest, VectorEvaluate) {
    PairPotential pot = LennardJones(1.0, 1.0, 2.5);
    arma::vec r = {1.0, 1.5, 2.0, 2.5};
    arma::vec v = evaluate(pot, r);
    EXPECT_EQ(v.n_elem, 4u);
    EXPECT_NEAR(v(3), 0.0, 1e-12);
  }

  TEST(PotentialIntegrationTest, FindHardSphereDiameter) {
    PairPotential pot = LennardJones(1.0, 1.0, 2.5);
    double d = find_hard_sphere_diameter(pot, 1.0);
    EXPECT_GT(d, 0.0);
    EXPECT_LT(d, 1.5);
  }

}  // namespace cdft::physics
