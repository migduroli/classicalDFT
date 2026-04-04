// Cross-validation of potentials against Jim's code.

#include "dft.hpp"
#include "legacy/classicaldft.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace dft;
namespace pot = physics::potentials;
using Catch::Approx;

static constexpr double SIGMA = 1.0;
static constexpr double EPS = 1.0;
static constexpr double LJ_RCUT = 2.5;
static constexpr double TWF_RCUT = 2.5;
static constexpr double TWF_ALPHA = 50.0;
static constexpr double WHDF_RCUT = 3.0;

TEST_CASE("LJ derived quantities match legacy", "[integration][potentials]") {
  auto jim = legacy::potentials::make_LJ(SIGMA, EPS, LJ_RCUT);
  auto ours = pot::make_lennard_jones(SIGMA, EPS, LJ_RCUT);
  CHECK(ours.epsilon_shift == Approx(jim.shift));
  CHECK(ours.r_min == Approx(jim.rmin));
  CHECK(ours.v_min == Approx(jim.Vmin));
  CHECK(ours.r_zero == Approx(jim.r0));
  CHECK(pot::hard_core_diameter(pot::Potential{ours}) == Approx(jim.getHardCore()));
}

TEST_CASE("tWF derived quantities match legacy", "[integration][potentials]") {
  auto jim = legacy::potentials::make_tWF(SIGMA, EPS, TWF_RCUT, TWF_ALPHA);
  auto ours = pot::make_ten_wolde_frenkel(SIGMA, EPS, TWF_RCUT, TWF_ALPHA);
  CHECK(ours.epsilon_shift == Approx(jim.shift));
  CHECK(ours.r_min == Approx(jim.rmin));
  CHECK(ours.v_min == Approx(jim.Vmin));
  CHECK(ours.r_zero == Approx(jim.r0));
  CHECK(pot::hard_core_diameter(pot::Potential{ours}) == Approx(jim.getHardCore()));
}

TEST_CASE("WHDF derived quantities match legacy", "[integration][potentials]") {
  auto jim = legacy::potentials::make_WHDF(SIGMA, EPS, WHDF_RCUT);
  auto ours = pot::make_wang_ramirez_dobnikar_frenkel(SIGMA, EPS, WHDF_RCUT);
  CHECK(ours.epsilon_effective == Approx(jim.eps_rescaled));
  CHECK(ours.r_min == Approx(jim.rmin));
  CHECK(ours.v_min == Approx(jim.Vmin));
  CHECK(pot::hard_core_diameter(pot::Potential{ours}) == Approx(jim.getHardCore()));
}

TEST_CASE("LJ vr(r) matches legacy at 100 points", "[integration][potentials]") {
  auto jim = legacy::potentials::make_LJ(SIGMA, EPS, LJ_RCUT);
  auto ours = pot::make_lennard_jones(SIGMA, EPS, LJ_RCUT);
  for (int i = 1; i <= 100; ++i) {
    double r = 0.8 + i * 0.02;
    CHECK(pot::potential(ours, r) == Approx(legacy::potentials::LJ::vr(SIGMA, EPS, r)).margin(1e-10));
  }
}

TEST_CASE("tWF vr(r) matches legacy at 100 points", "[integration][potentials]") {
  auto ours = pot::make_ten_wolde_frenkel(SIGMA, EPS, TWF_RCUT, TWF_ALPHA);
  for (int i = 1; i <= 100; ++i) {
    double r = 0.8 + i * 0.02;
    if (r <= SIGMA + 1e-6)
      continue;
    CHECK(pot::potential(ours, r) == Approx(legacy::potentials::tWF::vr(SIGMA, EPS, TWF_ALPHA, r)).margin(1e-10));
  }
}

TEST_CASE("WHDF vr(r) matches legacy at 100 points", "[integration][potentials]") {
  auto jim = legacy::potentials::make_WHDF(SIGMA, EPS, WHDF_RCUT);
  auto ours = pot::make_wang_ramirez_dobnikar_frenkel(SIGMA, EPS, WHDF_RCUT);
  for (int i = 1; i <= 100; ++i) {
    double r = 0.8 + i * 0.02;
    if (r >= WHDF_RCUT)
      continue;
    CHECK(
        pot::potential(ours, r) ==
        Approx(legacy::potentials::WHDF::vr(jim.eps_rescaled, SIGMA, WHDF_RCUT, r)).margin(1e-10)
    );
  }
}

TEST_CASE("LJ V(r), V0(r), Watt(r) match legacy", "[integration][potentials]") {
  auto jim = legacy::potentials::make_LJ(SIGMA, EPS, LJ_RCUT);
  auto ours = pot::make_lennard_jones(SIGMA, EPS, LJ_RCUT);
  pot::Potential p{ours};
  for (int i = 1; i <= 100; ++i) {
    double r = 0.8 + i * 0.02;
    CHECK(pot::energy(p, r) == Approx(legacy::potentials::V(jim, r)).margin(1e-10));
    CHECK(
        pot::repulsive(p, r, pot::SplitScheme::WeeksChandlerAndersen) ==
        Approx(legacy::potentials::V0(jim, r)).margin(1e-10)
    );
    CHECK(
        pot::attractive(p, r, pot::SplitScheme::WeeksChandlerAndersen) ==
        Approx(legacy::potentials::Watt(jim, r)).margin(1e-10)
    );
  }
}

TEST_CASE("tWF V(r), V0(r), Watt(r) match legacy", "[integration][potentials]") {
  auto jim = legacy::potentials::make_tWF(SIGMA, EPS, TWF_RCUT, TWF_ALPHA);
  auto ours = pot::make_ten_wolde_frenkel(SIGMA, EPS, TWF_RCUT, TWF_ALPHA);
  pot::Potential p{ours};
  for (int i = 1; i <= 100; ++i) {
    double r = 0.8 + i * 0.02;
    if (r <= SIGMA + 1e-6)
      continue;
    CHECK(pot::energy(p, r) == Approx(legacy::potentials::V(jim, r)).margin(1e-10));
    CHECK(
        pot::repulsive(p, r, pot::SplitScheme::WeeksChandlerAndersen) ==
        Approx(legacy::potentials::V0(jim, r)).margin(1e-10)
    );
    CHECK(
        pot::attractive(p, r, pot::SplitScheme::WeeksChandlerAndersen) ==
        Approx(legacy::potentials::Watt(jim, r)).margin(1e-10)
    );
  }
}

TEST_CASE("WHDF V(r), V0(r), Watt(r) match legacy", "[integration][potentials]") {
  auto jim = legacy::potentials::make_WHDF(SIGMA, EPS, WHDF_RCUT);
  auto ours = pot::make_wang_ramirez_dobnikar_frenkel(SIGMA, EPS, WHDF_RCUT);
  pot::Potential p{ours};
  for (int i = 1; i <= 100; ++i) {
    double r = 0.8 + i * 0.02;
    CHECK(pot::energy(p, r) == Approx(legacy::potentials::V(jim, r)).margin(1e-10));
    CHECK(
        pot::repulsive(p, r, pot::SplitScheme::WeeksChandlerAndersen) ==
        Approx(legacy::potentials::V0(jim, r)).margin(1e-10)
    );
    CHECK(
        pot::attractive(p, r, pot::SplitScheme::WeeksChandlerAndersen) ==
        Approx(legacy::potentials::Watt(jim, r)).margin(1e-10)
    );
  }
}

TEST_CASE("HSD matches legacy at multiple temperatures", "[integration][potentials]") {
  auto jim_lj = legacy::potentials::make_LJ(SIGMA, EPS, LJ_RCUT);
  auto jim_twf = legacy::potentials::make_tWF(SIGMA, EPS, TWF_RCUT, TWF_ALPHA);
  auto jim_whdf = legacy::potentials::make_WHDF(SIGMA, EPS, WHDF_RCUT);

  pot::Potential plj{pot::make_lennard_jones(SIGMA, EPS, LJ_RCUT)};
  pot::Potential ptwf{pot::make_ten_wolde_frenkel(SIGMA, EPS, TWF_RCUT, TWF_ALPHA)};
  pot::Potential pwhdf{pot::make_wang_ramirez_dobnikar_frenkel(SIGMA, EPS, WHDF_RCUT)};

  for (double kT : {0.5, 0.7, 1.0, 1.5, 2.0, 5.0}) {
    CHECK(
        pot::hard_sphere_diameter(plj, kT, pot::SplitScheme::WeeksChandlerAndersen) ==
        Approx(legacy::potentials::getHSD(jim_lj, kT)).epsilon(1e-6)
    );
    CHECK(
        pot::hard_sphere_diameter(ptwf, kT, pot::SplitScheme::WeeksChandlerAndersen) ==
        Approx(legacy::potentials::getHSD(jim_twf, kT)).epsilon(1e-6)
    );
    CHECK(
        pot::hard_sphere_diameter(pwhdf, kT, pot::SplitScheme::WeeksChandlerAndersen) ==
        Approx(legacy::potentials::getHSD(jim_whdf, kT)).epsilon(1e-6)
    );
  }
}

TEST_CASE("a_vdw matches legacy at multiple temperatures", "[integration][potentials]") {
  auto jim_lj = legacy::potentials::make_LJ(SIGMA, EPS, LJ_RCUT);
  auto jim_twf = legacy::potentials::make_tWF(SIGMA, EPS, TWF_RCUT, TWF_ALPHA);
  auto jim_whdf = legacy::potentials::make_WHDF(SIGMA, EPS, WHDF_RCUT);

  pot::Potential plj{pot::make_lennard_jones(SIGMA, EPS, LJ_RCUT)};
  pot::Potential ptwf{pot::make_ten_wolde_frenkel(SIGMA, EPS, TWF_RCUT, TWF_ALPHA)};
  pot::Potential pwhdf{pot::make_wang_ramirez_dobnikar_frenkel(SIGMA, EPS, WHDF_RCUT)};

  for (double kT : {0.5, 0.7, 1.0, 1.5, 2.0, 5.0}) {
    CHECK(
        pot::vdw_integral(plj, kT, pot::SplitScheme::WeeksChandlerAndersen) ==
        Approx(legacy::potentials::getVDW(jim_lj, kT)).epsilon(1e-6)
    );
    CHECK(
        pot::vdw_integral(ptwf, kT, pot::SplitScheme::WeeksChandlerAndersen) ==
        Approx(legacy::potentials::getVDW(jim_twf, kT)).epsilon(1e-6)
    );
    CHECK(
        pot::vdw_integral(pwhdf, kT, pot::SplitScheme::WeeksChandlerAndersen) ==
        Approx(legacy::potentials::getVDW(jim_whdf, kT)).epsilon(1e-6)
    );
  }
}
