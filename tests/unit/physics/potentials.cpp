#include "dft/physics/potentials.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <numbers>

using namespace dft::physics::potentials;

// LennardJones factory

TEST_CASE("LennardJones factory computes derived quantities", "[potentials]") {
  auto lj = make_lennard_jones(1.0, 1.0, 2.5);

  CHECK(lj.sigma == 1.0);
  CHECK(lj.epsilon == 1.0);
  CHECK(lj.r_cutoff == 2.5);
  CHECK(lj.r_min == Catch::Approx(std::pow(2.0, 1.0 / 6.0)).margin(1e-12));
  CHECK(lj.epsilon_shift != 0.0);
  CHECK(lj.v_min < 0.0);
}

TEST_CASE("LennardJones without cutoff has zero shift", "[potentials]") {
  auto lj = make_lennard_jones(1.0, 1.0);

  CHECK(lj.epsilon_shift == 0.0);
}

// TenWoldeFrenkel factory

TEST_CASE("TenWoldeFrenkel factory sets alpha and hard core", "[potentials]") {
  auto twf = make_ten_wolde_frenkel(1.0, 1.0, 2.5, 50.0);

  CHECK(twf.alpha == 50.0);
  CHECK(twf.r_min > twf.sigma);
  CHECK(twf.v_min < 0.0);
}

// WangRamirezDobnikarFrenkel factory

TEST_CASE("WRDF factory rescales epsilon", "[potentials]") {
  auto w = make_wang_ramirez_dobnikar_frenkel(1.0, 1.0, 3.0);

  CHECK(w.epsilon_effective > 0.0);
  CHECK(w.r_min > 0.0);
  CHECK(w.v_min < 0.0);
}

// Pair potential

TEST_CASE("LennardJones potential at r_min is -epsilon", "[potentials]") {
  auto lj = make_lennard_jones(1.0, 1.0);
  double r = lj.r_min;

  CHECK(lj(r) == Catch::Approx(-1.0).margin(1e-12));
}

TEST_CASE("LennardJones from_r2 matches operator()", "[potentials]") {
  auto lj = make_lennard_jones(1.0, 1.0, 2.5);
  double r = 1.5;

  CHECK(lj.from_r2(r * r) == Catch::Approx(lj(r)).margin(1e-12));
}

TEST_CASE("TenWoldeFrenkel diverges inside hard core", "[potentials]") {
  auto twf = make_ten_wolde_frenkel(1.0, 1.0, 2.5);

  CHECK(twf(0.5) == MAX_POTENTIAL_VALUE);
  CHECK(twf.from_r2(0.25) == MAX_POTENTIAL_VALUE);
}

TEST_CASE("WRDF vanishes at and beyond r_cutoff", "[potentials]") {
  auto w = make_wang_ramirez_dobnikar_frenkel(1.0, 1.0, 3.0);

  CHECK(w(3.0) == 0.0);
  CHECK(w(4.0) == 0.0);
  CHECK(w.from_r2(9.0) == 0.0);
}

// Cut-and-shifted energy via variant

TEST_CASE("energy() returns shifted LJ value", "[potentials]") {
  auto lj = make_lennard_jones(1.0, 1.0, 2.5);

  double shifted = lj(1.5) - lj.epsilon_shift;
  CHECK(lj.energy(1.5) == Catch::Approx(shifted).margin(1e-12));
}

// Hard-core diameter

TEST_CASE("LJ has zero hard-core diameter", "[potentials]") {
  auto lj = make_lennard_jones(1.0, 1.0, 2.5);
  CHECK(lj.hard_core_diameter() == 0.0);
}

TEST_CASE("TenWoldeFrenkel has sigma hard-core diameter", "[potentials]") {
  auto twf = make_ten_wolde_frenkel(1.0, 1.0, 2.5);
  CHECK(twf.hard_core_diameter() == 1.0);
}

// Name

TEST_CASE("name() returns correct potential name", "[potentials]") {
  CHECK(LennardJones::NAME == "LennardJones");
  CHECK(TenWoldeFrenkel::NAME == "TenWoldeFrenkel");
  CHECK(WangRamirezDobnikarFrenkel::NAME == "WangRamirezDobnikarFrenkel");
}

// r_min and v_min

TEST_CASE("r_min and v_min of LJ", "[potentials]") {
  auto lj = make_lennard_jones(1.0, 1.0);
  CHECK(lj.r_min == Catch::Approx(std::pow(2.0, 1.0 / 6.0)).margin(1e-12));
  CHECK(lj.v_min == Catch::Approx(-1.0).margin(1e-12));
}

// Repulsive / attractive split (WCA)

TEST_CASE("WCA repulsive is nonzero inside r_min, zero outside", "[potentials]") {
  auto lj = make_lennard_jones(1.0, 1.0, 2.5);

  double r_inner = 1.0;
  double r_outer = 1.5;

  CHECK(lj.repulsive(r_inner, SplitScheme::WeeksChandlerAndersen) > 0.0);
  CHECK(lj.repulsive(r_outer, SplitScheme::WeeksChandlerAndersen) == 0.0);
}

TEST_CASE("WCA attractive equals v_min inside r_min", "[potentials]") {
  auto lj = make_lennard_jones(1.0, 1.0, 2.5);

  CHECK(lj.attractive(1.0, SplitScheme::WeeksChandlerAndersen) == Catch::Approx(lj.v_min).margin(1e-10));
}

TEST_CASE("WCA attractive is zero beyond cutoff", "[potentials]") {
  auto lj = make_lennard_jones(1.0, 1.0, 2.5);
  CHECK(lj.attractive(3.0, SplitScheme::WeeksChandlerAndersen) == 0.0);
}

// BH split

TEST_CASE("BH repulsive is nonzero inside r_zero, zero at r_min", "[potentials]") {
  auto lj = make_lennard_jones(1.0, 1.0, 2.5);

  CHECK(lj.repulsive(0.9, SplitScheme::BarkerHenderson) != 0.0);
  CHECK(lj.repulsive(lj.r_min, SplitScheme::BarkerHenderson) == 0.0);
}

// Hard-sphere diameter via integration

TEST_CASE("LJ hard-sphere diameter is approximately sigma at kT=1", "[potentials]") {
  auto lj = make_lennard_jones(1.0, 1.0, 2.5);
  double d_hs = lj.hard_sphere_diameter(1.0, SplitScheme::WeeksChandlerAndersen);

  CHECK(d_hs == Catch::Approx(1.0).margin(0.15));
  CHECK(d_hs > 0.0);
  CHECK(d_hs < 1.5);
}

// vdw_integral

TEST_CASE("LJ vdw integral is negative at kT=1", "[potentials]") {
  auto lj = make_lennard_jones(1.0, 1.0, 2.5);
  double a = lj.vdw_integral(1.0, SplitScheme::WeeksChandlerAndersen);

  CHECK(a < 0.0);
}

TEST_CASE("vdw_integral returns zero without cutoff", "[potentials]") {
  auto lj = make_lennard_jones(1.0, 1.0);
  double a = lj.vdw_integral(1.0, SplitScheme::WeeksChandlerAndersen);

  CHECK(a == 0.0);
}

// TenWoldeFrenkel: potential and potential_r2 outside hard core

TEST_CASE("TenWoldeFrenkel potential at r_min matches v_min + shift", "[potentials]") {
  auto twf = make_ten_wolde_frenkel(1.0, 1.0, 2.5, 50.0);
  double v = twf(twf.r_min);

  CHECK(v == Catch::Approx(twf.v_min + twf.epsilon_shift).margin(1e-10));
}

TEST_CASE("TenWoldeFrenkel from_r2 matches operator()", "[potentials]") {
  auto twf = make_ten_wolde_frenkel(1.0, 1.0, 2.5);
  double r = 1.5;

  CHECK(twf.from_r2(r * r) == Catch::Approx(twf(r)).margin(1e-12));
}

// WRDF: operator() and from_r2 inside cutoff

TEST_CASE("WRDF potential is negative near r_min", "[potentials]") {
  auto w = make_wang_ramirez_dobnikar_frenkel(1.0, 1.0, 3.0);

  CHECK(w(w.r_min) == Catch::Approx(w.v_min).margin(1e-10));
}

TEST_CASE("WRDF from_r2 matches operator()", "[potentials]") {
  auto w = make_wang_ramirez_dobnikar_frenkel(1.0, 1.0, 3.0);
  double r = 2.0;

  CHECK(w.from_r2(r * r) == Catch::Approx(w(r)).margin(1e-12));
}

// energy() via variant for tWF and WRDF

TEST_CASE("energy() for TenWoldeFrenkel", "[potentials]") {
  auto twf = make_ten_wolde_frenkel(1.0, 1.0, 2.5);

  double r = 1.5;
  double expected = twf(r) - twf.epsilon_shift;
  CHECK(twf.energy(r) == Catch::Approx(expected).margin(1e-12));
}

TEST_CASE("energy() for WRDF returns potential value", "[potentials]") {
  auto w = make_wang_ramirez_dobnikar_frenkel(1.0, 1.0, 3.0);

  double r = 2.0;
  CHECK(w.energy(r) == Catch::Approx(w(r)).margin(1e-12));
}

// WRDF hard-core diameter is zero

TEST_CASE("WRDF has zero hard-core diameter", "[potentials]") {
  auto w = make_wang_ramirez_dobnikar_frenkel(1.0, 1.0, 3.0);
  CHECK(w.hard_core_diameter() == 0.0);
}

// WRDF repulsive/attractive splits

TEST_CASE("WRDF WCA repulsive is nonzero inside r_min", "[potentials]") {
  auto w = make_wang_ramirez_dobnikar_frenkel(1.0, 1.0, 3.0);

  CHECK(w.repulsive(0.5, SplitScheme::WeeksChandlerAndersen) > 0.0);
  CHECK(w.repulsive(w.r_min + 0.1, SplitScheme::WeeksChandlerAndersen) == 0.0);
}

TEST_CASE("WRDF BH repulsive returns raw energy inside r_min", "[potentials]") {
  auto w = make_wang_ramirez_dobnikar_frenkel(1.0, 1.0, 3.0);

  CHECK(w.repulsive(0.5, SplitScheme::BarkerHenderson) > 0.0);
  CHECK(w.repulsive(w.r_min + 0.1, SplitScheme::BarkerHenderson) == 0.0);
}

TEST_CASE("WRDF WCA attractive equals v_min inside r_min", "[potentials]") {
  auto w = make_wang_ramirez_dobnikar_frenkel(1.0, 1.0, 3.0);

  CHECK(w.attractive(0.5, SplitScheme::WeeksChandlerAndersen) == Catch::Approx(w.v_min).margin(1e-10));
}

TEST_CASE("WRDF BH attractive is zero inside r_min", "[potentials]") {
  auto w = make_wang_ramirez_dobnikar_frenkel(1.0, 1.0, 3.0);

  CHECK(w.attractive(0.5, SplitScheme::BarkerHenderson) == 0.0);
  CHECK(w.attractive(w.r_min + 0.1, SplitScheme::BarkerHenderson) < 0.0);
}

TEST_CASE("WRDF attractive is zero beyond cutoff", "[potentials]") {
  auto w = make_wang_ramirez_dobnikar_frenkel(1.0, 1.0, 3.0);
  CHECK(w.attractive(4.0, SplitScheme::WeeksChandlerAndersen) == 0.0);
}

// LJ BH attractive paths

TEST_CASE("LJ BH attractive is zero inside r_zero", "[potentials]") {
  auto lj = make_lennard_jones(1.0, 1.0, 2.5);

  CHECK(lj.attractive(0.9, SplitScheme::BarkerHenderson) == 0.0);
  CHECK(lj.attractive(lj.r_zero + 0.01, SplitScheme::BarkerHenderson) != 0.0);
}

// tWF repulsive/attractive

TEST_CASE("TenWoldeFrenkel WCA repulsive/attractive splits", "[potentials]") {
  auto twf = make_ten_wolde_frenkel(1.0, 1.0, 2.5, 50.0);

  CHECK(twf.repulsive(twf.sigma + 0.01, SplitScheme::WeeksChandlerAndersen) > 0.0);
  CHECK(twf.repulsive(twf.r_min + 0.1, SplitScheme::WeeksChandlerAndersen) == 0.0);
  CHECK(twf.attractive(twf.r_min + 0.1, SplitScheme::WeeksChandlerAndersen) < 0.0);
}

TEST_CASE("TenWoldeFrenkel BH split", "[potentials]") {
  auto twf = make_ten_wolde_frenkel(1.0, 1.0, 2.5, 50.0);

  CHECK(twf.repulsive(twf.sigma + 0.01, SplitScheme::BarkerHenderson) != 0.0);
  CHECK(twf.attractive(twf.r_zero + 0.01, SplitScheme::BarkerHenderson) != 0.0);
  CHECK(twf.attractive(twf.sigma + 0.001, SplitScheme::BarkerHenderson) == 0.0);
}

// Hard-sphere diameter and vdw for non-LJ types

TEST_CASE("TenWoldeFrenkel hard-sphere diameter is approximately sigma", "[potentials]") {
  auto twf = make_ten_wolde_frenkel(1.0, 1.0, 2.5, 50.0);
  double d_hs = twf.hard_sphere_diameter(1.0, SplitScheme::WeeksChandlerAndersen);

  CHECK(d_hs > 0.9);
  CHECK(d_hs < 1.2);
}

TEST_CASE("TenWoldeFrenkel BH hard-sphere diameter", "[potentials]") {
  auto twf = make_ten_wolde_frenkel(1.0, 1.0, 2.5, 50.0);
  double d_hs = twf.hard_sphere_diameter(1.0, SplitScheme::BarkerHenderson);

  CHECK(d_hs > 0.9);
  CHECK(d_hs < 1.2);
}

TEST_CASE("WRDF hard-sphere diameter", "[potentials]") {
  auto w = make_wang_ramirez_dobnikar_frenkel(1.0, 1.0, 3.0);
  double d_hs = w.hard_sphere_diameter(1.0, SplitScheme::WeeksChandlerAndersen);

  CHECK(d_hs > 0.0);
  CHECK(d_hs < 2.0);
}

TEST_CASE("WRDF BH hard-sphere diameter", "[potentials]") {
  auto w = make_wang_ramirez_dobnikar_frenkel(1.0, 1.0, 3.0);
  double d_hs = w.hard_sphere_diameter(1.0, SplitScheme::BarkerHenderson);

  CHECK(d_hs > 0.0);
  CHECK(d_hs < 2.0);
}

TEST_CASE("TenWoldeFrenkel vdw integral is negative", "[potentials]") {
  auto twf = make_ten_wolde_frenkel(1.0, 1.0, 2.5, 50.0);
  double a = twf.vdw_integral(1.0, SplitScheme::WeeksChandlerAndersen);

  CHECK(a < 0.0);
}

TEST_CASE("WRDF vdw integral is negative", "[potentials]") {
  auto w = make_wang_ramirez_dobnikar_frenkel(1.0, 1.0, 3.0);
  double a = w.vdw_integral(1.0, SplitScheme::WeeksChandlerAndersen);

  CHECK(a < 0.0);
}

TEST_CASE("LJ BH vdw integral", "[potentials]") {
  auto lj = make_lennard_jones(1.0, 1.0, 2.5);
  double a = lj.vdw_integral(1.0, SplitScheme::BarkerHenderson);

  CHECK(a < 0.0);
}

TEST_CASE("WRDF BH vdw integral", "[potentials]") {
  auto w = make_wang_ramirez_dobnikar_frenkel(1.0, 1.0, 3.0);
  double a = w.vdw_integral(1.0, SplitScheme::BarkerHenderson);

  CHECK(a < 0.0);
}

// r_min and v_min for tWF and WRDF via variant

TEST_CASE("r_min and v_min for tWF", "[potentials]") {
  auto twf = make_ten_wolde_frenkel(1.0, 1.0, 2.5, 50.0);

  CHECK(twf.r_min == Catch::Approx(twf.r_min).margin(1e-12));
  CHECK(twf.v_min == Catch::Approx(twf.v_min).margin(1e-12));
}

TEST_CASE("r_min and v_min for WRDF", "[potentials]") {
  auto w = make_wang_ramirez_dobnikar_frenkel(1.0, 1.0, 3.0);

  CHECK(w.r_min > 0.0);
  CHECK(w.v_min < 0.0);
}
