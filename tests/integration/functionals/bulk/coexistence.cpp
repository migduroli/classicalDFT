// Cross-validation of spinodal/coexistence solvers against Jim's code.

#include "dft.hpp"
#include "legacy/classicaldft.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace dft;
using Catch::Approx;

static constexpr double SIGMA = 1.0;
static constexpr double EPS = 1.0;
static constexpr double RCUT = 2.5;

struct SolverFixture {
  physics::Model model;
  functionals::bulk::BulkThermodynamics eos;
  legacy::solver::EOS legacy_eos;

  SolverFixture(double kT)
      : model{
            .grid = make_grid(0.1, {6.0, 6.0, 6.0}),
            .species = {Species{.name = "LJ", .hard_sphere_diameter = SIGMA}},
            .interactions = {{
                .species_i = 0, .species_j = 0,
                .potential = physics::potentials::make_lennard_jones(SIGMA, EPS, RCUT),
                .split = physics::potentials::SplitScheme::WeeksChandlerAndersen,
            }},
            .temperature = kT,
        },
        eos(functionals::bulk::make_bulk_thermodynamics(
            model.species,
            functionals::make_bulk_weights(functionals::fmt::WhiteBearII{}, model.interactions, kT))),
        legacy_eos{
            .pressure = [this](double rho) {
              return eos.pressure(arma::vec{rho});
            },
            .chemical_potential = [this](double rho) {
              return eos.chemical_potential(arma::vec{rho}, 0);
            },
        } {}
};

TEST_CASE("Spinodal matches legacy at kT=0.7", "[integration][solver]") {
  SolverFixture fix(0.7);
  auto ours = functionals::bulk::PhaseSearch{ .rho_max = 1.0, .rho_scan_step = 0.005 }.find_spinodal(fix.eos);
  auto jims = legacy::solver::findSpinodal(fix.legacy_eos, 1.0, 0.005);
  REQUIRE(ours.has_value());
  CHECK(ours->rho_low == Approx(jims.xs1).epsilon(1e-6));
  CHECK(ours->rho_high == Approx(jims.xs2).epsilon(1e-6));
}

TEST_CASE("Spinodal matches legacy at kT=0.8", "[integration][solver]") {
  SolverFixture fix(0.8);
  auto ours = functionals::bulk::PhaseSearch{ .rho_max = 1.0, .rho_scan_step = 0.005 }.find_spinodal(fix.eos);
  auto jims = legacy::solver::findSpinodal(fix.legacy_eos, 1.0, 0.005);
  REQUIRE(ours.has_value());
  CHECK(ours->rho_low == Approx(jims.xs1).epsilon(1e-6));
  CHECK(ours->rho_high == Approx(jims.xs2).epsilon(1e-6));
}

TEST_CASE("Spinodal matches legacy at kT=0.9", "[integration][solver]") {
  SolverFixture fix(0.9);
  auto ours = functionals::bulk::PhaseSearch{ .rho_max = 1.0, .rho_scan_step = 0.005 }.find_spinodal(fix.eos);
  auto jims = legacy::solver::findSpinodal(fix.legacy_eos, 1.0, 0.005);
  REQUIRE(ours.has_value());
  CHECK(ours->rho_low == Approx(jims.xs1).epsilon(1e-6));
  CHECK(ours->rho_high == Approx(jims.xs2).epsilon(1e-6));
}

TEST_CASE("Coexistence matches legacy at kT=0.7", "[integration][solver]") {
  SolverFixture fix(0.7);
  auto ours = functionals::bulk::PhaseSearch{ .rho_max = 1.0, .rho_scan_step = 0.005 }.find_coexistence(fix.eos);
  auto jims = legacy::solver::findCoex(fix.legacy_eos, 1.0, 0.005);
  REQUIRE(ours.has_value());
  CHECK(ours->rho_vapor == Approx(jims.x1).epsilon(1e-6));
  CHECK(ours->rho_liquid == Approx(jims.x2).epsilon(1e-6));
}

TEST_CASE("Coexistence matches legacy at kT=0.8", "[integration][solver]") {
  SolverFixture fix(0.8);
  auto ours = functionals::bulk::PhaseSearch{ .rho_max = 1.0, .rho_scan_step = 0.005 }.find_coexistence(fix.eos);
  auto jims = legacy::solver::findCoex(fix.legacy_eos, 1.0, 0.005);
  REQUIRE(ours.has_value());
  CHECK(ours->rho_vapor == Approx(jims.x1).epsilon(1e-6));
  CHECK(ours->rho_liquid == Approx(jims.x2).epsilon(1e-6));
}

TEST_CASE("Coexistence matches legacy at kT=0.9", "[integration][solver]") {
  SolverFixture fix(0.9);
  auto ours = functionals::bulk::PhaseSearch{ .rho_max = 1.0, .rho_scan_step = 0.005 }.find_coexistence(fix.eos);
  auto jims = legacy::solver::findCoex(fix.legacy_eos, 1.0, 0.005);
  REQUIRE(ours.has_value());
  CHECK(ours->rho_vapor == Approx(jims.x1).epsilon(1e-6));
  CHECK(ours->rho_liquid == Approx(jims.x2).epsilon(1e-6));
}

TEST_CASE("Coexistence has equal pressure and chemical potential", "[integration][solver]") {
  SolverFixture fix(0.8);
  auto coex = functionals::bulk::PhaseSearch{ .rho_max = 1.0, .rho_scan_step = 0.005 }.find_coexistence(fix.eos);
  REQUIRE(coex.has_value());
  double p_v = fix.eos.pressure(arma::vec{ coex->rho_vapor });
  double p_l = fix.eos.pressure(arma::vec{ coex->rho_liquid });
  double mu_v = fix.eos.chemical_potential(arma::vec{ coex->rho_vapor }, 0);
  double mu_l = fix.eos.chemical_potential(arma::vec{ coex->rho_liquid }, 0);
  CHECK(p_v == Approx(p_l).epsilon(1e-6));
  CHECK(mu_v == Approx(mu_l).epsilon(1e-6));
}
