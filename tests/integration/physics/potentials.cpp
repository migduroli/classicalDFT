// Cross-validation of potentials against legacy code.

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
  auto ref = legacy::potentials::make_LJ(SIGMA, EPS, LJ_RCUT);
  auto ours = pot::make_lennard_jones(SIGMA, EPS, LJ_RCUT);
  CHECK(ours.epsilon_shift == Approx(ref.shift));
  CHECK(ours.r_min == Approx(ref.rmin));
  CHECK(ours.v_min == Approx(ref.Vmin));
  CHECK(ours.r_zero == Approx(ref.r0));
  CHECK(ours.hard_core_diameter() == Approx(ref.getHardCore()));
}

TEST_CASE("tWF derived quantities match legacy", "[integration][potentials]") {
  auto ref = legacy::potentials::make_tWF(SIGMA, EPS, TWF_RCUT, TWF_ALPHA);
  auto ours = pot::make_ten_wolde_frenkel(SIGMA, EPS, TWF_RCUT, TWF_ALPHA);
  CHECK(ours.epsilon_shift == Approx(ref.shift));
  CHECK(ours.r_min == Approx(ref.rmin));
  CHECK(ours.v_min == Approx(ref.Vmin));
  CHECK(ours.r_zero == Approx(ref.r0));
  CHECK(ours.hard_core_diameter() == Approx(ref.getHardCore()));
}

TEST_CASE("WHDF derived quantities match legacy", "[integration][potentials]") {
  auto ref = legacy::potentials::make_WHDF(SIGMA, EPS, WHDF_RCUT);
  auto ours = pot::make_wang_ramirez_dobnikar_frenkel(SIGMA, EPS, WHDF_RCUT);
  CHECK(ours.epsilon_effective == Approx(ref.eps_rescaled));
  CHECK(ours.r_min == Approx(ref.rmin));
  CHECK(ours.v_min == Approx(ref.Vmin));
  CHECK(ours.hard_core_diameter() == Approx(ref.getHardCore()));
}

TEST_CASE("LJ vr(r) matches legacy at 100 points", "[integration][potentials]") {
  auto ref = legacy::potentials::make_LJ(SIGMA, EPS, LJ_RCUT);
  auto ours = pot::make_lennard_jones(SIGMA, EPS, LJ_RCUT);
  for (int i = 1; i <= 100; ++i) {
    double r = 0.8 + i * 0.02;
    CHECK(ours(r) == Approx(legacy::potentials::LJ::vr(SIGMA, EPS, r)).margin(1e-10));
  }
}

TEST_CASE("tWF vr(r) matches legacy at 100 points", "[integration][potentials]") {
  auto ours = pot::make_ten_wolde_frenkel(SIGMA, EPS, TWF_RCUT, TWF_ALPHA);
  for (int i = 1; i <= 100; ++i) {
    double r = 0.8 + i * 0.02;
    if (r <= SIGMA + 1e-6)
      continue;
    CHECK(ours(r) == Approx(legacy::potentials::tWF::vr(SIGMA, EPS, TWF_ALPHA, r)).margin(1e-10));
  }
}

TEST_CASE("WHDF vr(r) matches legacy at 100 points", "[integration][potentials]") {
  auto ref = legacy::potentials::make_WHDF(SIGMA, EPS, WHDF_RCUT);
  auto ours = pot::make_wang_ramirez_dobnikar_frenkel(SIGMA, EPS, WHDF_RCUT);
  for (int i = 1; i <= 100; ++i) {
    double r = 0.8 + i * 0.02;
    if (r >= WHDF_RCUT)
      continue;
    CHECK(ours(r) == Approx(legacy::potentials::WHDF::vr(ref.eps_rescaled, SIGMA, WHDF_RCUT, r)).margin(1e-10));
  }
}

TEST_CASE("LJ V(r), V0(r), Watt(r) match legacy", "[integration][potentials]") {
  auto ref = legacy::potentials::make_LJ(SIGMA, EPS, LJ_RCUT);
  auto ours = pot::make_lennard_jones(SIGMA, EPS, LJ_RCUT);
  for (int i = 1; i <= 100; ++i) {
    double r = 0.8 + i * 0.02;
    CHECK(ours.energy(r) == Approx(legacy::potentials::V(ref, r)).margin(1e-10));
    CHECK(
        ours.repulsive(r, pot::SplitScheme::WeeksChandlerAndersen)
        == Approx(legacy::potentials::V0(ref, r)).margin(1e-10)
    );
    CHECK(
        ours.attractive(r, pot::SplitScheme::WeeksChandlerAndersen)
        == Approx(legacy::potentials::Watt(ref, r)).margin(1e-10)
    );
  }
}

TEST_CASE("tWF V(r), V0(r), Watt(r) match legacy", "[integration][potentials]") {
  auto ref = legacy::potentials::make_tWF(SIGMA, EPS, TWF_RCUT, TWF_ALPHA);
  auto ours = pot::make_ten_wolde_frenkel(SIGMA, EPS, TWF_RCUT, TWF_ALPHA);
  for (int i = 1; i <= 100; ++i) {
    double r = 0.8 + i * 0.02;
    if (r <= SIGMA + 1e-6)
      continue;
    CHECK(ours.energy(r) == Approx(legacy::potentials::V(ref, r)).margin(1e-10));
    CHECK(
        ours.repulsive(r, pot::SplitScheme::WeeksChandlerAndersen)
        == Approx(legacy::potentials::V0(ref, r)).margin(1e-10)
    );
    CHECK(
        ours.attractive(r, pot::SplitScheme::WeeksChandlerAndersen)
        == Approx(legacy::potentials::Watt(ref, r)).margin(1e-10)
    );
  }
}

TEST_CASE("WHDF V(r), V0(r), Watt(r) match legacy", "[integration][potentials]") {
  auto ref = legacy::potentials::make_WHDF(SIGMA, EPS, WHDF_RCUT);
  auto ours = pot::make_wang_ramirez_dobnikar_frenkel(SIGMA, EPS, WHDF_RCUT);
  for (int i = 1; i <= 100; ++i) {
    double r = 0.8 + i * 0.02;
    CHECK(ours.energy(r) == Approx(legacy::potentials::V(ref, r)).margin(1e-10));
    CHECK(
        ours.repulsive(r, pot::SplitScheme::WeeksChandlerAndersen)
        == Approx(legacy::potentials::V0(ref, r)).margin(1e-10)
    );
    CHECK(
        ours.attractive(r, pot::SplitScheme::WeeksChandlerAndersen)
        == Approx(legacy::potentials::Watt(ref, r)).margin(1e-10)
    );
  }
}

TEST_CASE("HSD matches legacy at multiple temperatures", "[integration][potentials]") {
  auto ref_lj = legacy::potentials::make_LJ(SIGMA, EPS, LJ_RCUT);
  auto ref_twf = legacy::potentials::make_tWF(SIGMA, EPS, TWF_RCUT, TWF_ALPHA);
  auto ref_whdf = legacy::potentials::make_WHDF(SIGMA, EPS, WHDF_RCUT);

  auto plj = pot::make_lennard_jones(SIGMA, EPS, LJ_RCUT);
  auto ptwf = pot::make_ten_wolde_frenkel(SIGMA, EPS, TWF_RCUT, TWF_ALPHA);
  auto pwhdf = pot::make_wang_ramirez_dobnikar_frenkel(SIGMA, EPS, WHDF_RCUT);

  for (double kT : {0.5, 0.7, 1.0, 1.5, 2.0, 5.0}) {
    CHECK(
        plj.hard_sphere_diameter(kT, pot::SplitScheme::WeeksChandlerAndersen)
        == Approx(legacy::potentials::getHSD(ref_lj, kT)).epsilon(1e-6)
    );
    CHECK(
        ptwf.hard_sphere_diameter(kT, pot::SplitScheme::WeeksChandlerAndersen)
        == Approx(legacy::potentials::getHSD(ref_twf, kT)).epsilon(1e-6)
    );
    CHECK(
        pwhdf.hard_sphere_diameter(kT, pot::SplitScheme::WeeksChandlerAndersen)
        == Approx(legacy::potentials::getHSD(ref_whdf, kT)).epsilon(1e-6)
    );
  }
}

TEST_CASE("a_vdw matches legacy at multiple temperatures", "[integration][potentials]") {
  auto ref_lj = legacy::potentials::make_LJ(SIGMA, EPS, LJ_RCUT);
  auto ref_twf = legacy::potentials::make_tWF(SIGMA, EPS, TWF_RCUT, TWF_ALPHA);
  auto ref_whdf = legacy::potentials::make_WHDF(SIGMA, EPS, WHDF_RCUT);

  auto plj = pot::make_lennard_jones(SIGMA, EPS, LJ_RCUT);
  auto ptwf = pot::make_ten_wolde_frenkel(SIGMA, EPS, TWF_RCUT, TWF_ALPHA);
  auto pwhdf = pot::make_wang_ramirez_dobnikar_frenkel(SIGMA, EPS, WHDF_RCUT);

  for (double kT : {0.5, 0.7, 1.0, 1.5, 2.0, 5.0}) {
    CHECK(
        plj.vdw_integral(kT, pot::SplitScheme::WeeksChandlerAndersen)
        == Approx(legacy::potentials::getVDW(ref_lj, kT)).epsilon(1e-6)
    );
    CHECK(
        ptwf.vdw_integral(kT, pot::SplitScheme::WeeksChandlerAndersen)
        == Approx(legacy::potentials::getVDW(ref_twf, kT)).epsilon(1e-6)
    );
    CHECK(
        pwhdf.vdw_integral(kT, pot::SplitScheme::WeeksChandlerAndersen)
        == Approx(legacy::potentials::getVDW(ref_whdf, kT)).epsilon(1e-6)
    );
  }
}
