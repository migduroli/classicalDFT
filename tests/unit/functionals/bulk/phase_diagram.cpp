#include "dft/functionals/bulk/phase_diagram.hpp"

#include "dft/functionals/bulk/thermodynamics.hpp"
#include "dft/functionals/fmt/models.hpp"
#include "dft/functionals/functionals.hpp"
#include "dft/physics/interactions.hpp"
#include "dft/physics/potentials.hpp"

#include <armadillo>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace dft::functionals::bulk;
using namespace dft::physics;
using namespace dft::physics::potentials;
using dft::Species;
using dft::functionals::make_bulk_weights;
using dft::functionals::Weights;
using dft::functionals::fmt::Rosenfeld;

// A strongly attractive LJ system that has a clear van der Waals loop.

static constexpr double SIGMA = 1.0;
static constexpr double EPSILON = 1.0;
static constexpr double R_CUTOFF = 2.5;
static constexpr double KT = 0.8;

static const std::vector<Species> SPECIES = {{.name = "LJ", .hard_sphere_diameter = SIGMA}};

static auto make_lj_interactions() -> std::vector<Interaction> {
  auto pot = make_lennard_jones(SIGMA, EPSILON, R_CUTOFF);
  return {{.species_i = 0, .species_j = 0, .potential = pot}};
}

static auto make_lj_weights() -> Weights {
  return make_bulk_weights(Rosenfeld{}, make_lj_interactions(), KT);
}

static auto make_lj_weights_at(double kT) -> Weights {
  return make_bulk_weights(Rosenfeld{}, make_lj_interactions(), kT);
}

static auto make_lj_eos() -> BulkThermodynamics {
  return make_bulk_thermodynamics(SPECIES, make_lj_weights());
}

static auto make_lj_eos_at(double kT) -> BulkThermodynamics {
  return make_bulk_thermodynamics(SPECIES, make_lj_weights_at(kT));
}

static const PhaseSearch CONFIG{.rho_max = 0.9, .rho_scan_step = 0.005};

// density_from_chemical_potential

TEST_CASE("density_from_chemical_potential finds ideal gas density", "[phase_diagram]") {
  auto eos = make_bulk_thermodynamics(
      SPECIES, make_bulk_weights(Rosenfeld{}, {}, 1.0)
  );

  double target_mu = ideal::chemical_potential(0.001);
  auto result = density_from_chemical_potential(target_mu, 0.001, eos);
  REQUIRE(result.has_value());
  CHECK(*result == Catch::Approx(0.001).epsilon(1e-2));
}

TEST_CASE("density_from_chemical_potential returns nullopt for bad guess", "[phase_diagram]") {
  auto eos = make_bulk_thermodynamics(
      SPECIES, make_bulk_weights(Rosenfeld{}, {}, 1.0)
  );

  // A negative initial guess should fail
  auto result = density_from_chemical_potential(100.0, -1.0, eos);
  CHECK_FALSE(result.has_value());
}

// Spinodal

TEST_CASE("find_spinodal returns two densities for attractive LJ system", "[phase_diagram]") {
  auto eos = make_lj_eos();
  auto result = CONFIG.find_spinodal(eos);

  REQUIRE(result.has_value());
  CHECK(result->rho_low > 0.0);
  CHECK(result->rho_high > result->rho_low);
  CHECK(result->rho_high < CONFIG.rho_max);
}

TEST_CASE("spinodal densities bound the unstable region", "[phase_diagram]") {
  auto eos = make_lj_eos();
  auto sp = CONFIG.find_spinodal(eos);
  REQUIRE(sp.has_value());

  // At the spinodal, dP/drho ~ 0.
  // P at rho_low should be a local maximum
  double p_low = eos.pressure(arma::vec{sp->rho_low});
  double p_left = eos.pressure(arma::vec{sp->rho_low - 0.001});
  double p_right = eos.pressure(arma::vec{sp->rho_low + 0.001});
  CHECK(p_low >= p_left - 1e-6);
  CHECK(p_low >= p_right - 1e-6);

  // P at rho_high should be a local minimum
  double p_high = eos.pressure(arma::vec{sp->rho_high});
  double p_left2 = eos.pressure(arma::vec{sp->rho_high - 0.001});
  double p_right2 = eos.pressure(arma::vec{sp->rho_high + 0.001});
  CHECK(p_high <= p_left2 + 1e-6);
  CHECK(p_high <= p_right2 + 1e-6);
}

TEST_CASE("find_spinodal returns nullopt for purely repulsive system", "[phase_diagram]") {
  auto eos = make_bulk_thermodynamics(
      SPECIES, make_bulk_weights(Rosenfeld{}, {}, 1.0)
  );

  auto result = CONFIG.find_spinodal(eos);
  CHECK_FALSE(result.has_value());
}

// Coexistence

TEST_CASE("find_coexistence returns vapor and liquid densities for LJ", "[phase_diagram]") {
  auto eos = make_lj_eos();
  auto result = CONFIG.find_coexistence(eos);

  REQUIRE(result.has_value());
  CHECK(result->rho_vapor > 0.0);
  CHECK(result->rho_liquid > result->rho_vapor);
}

TEST_CASE("coexistence densities satisfy equal pressure", "[phase_diagram]") {
  auto eos = make_lj_eos();
  auto coex = CONFIG.find_coexistence(eos);
  REQUIRE(coex.has_value());

  double p_v = eos.pressure(arma::vec{coex->rho_vapor});
  double p_l = eos.pressure(arma::vec{coex->rho_liquid});
  CHECK(p_v == Catch::Approx(p_l).epsilon(1e-6));
}

TEST_CASE("coexistence densities satisfy equal chemical potential", "[phase_diagram]") {
  auto eos = make_lj_eos();
  auto coex = CONFIG.find_coexistence(eos);
  REQUIRE(coex.has_value());

  double mu_v = eos.chemical_potential(arma::vec{coex->rho_vapor}, 0);
  double mu_l = eos.chemical_potential(arma::vec{coex->rho_liquid}, 0);
  CHECK(mu_v == Catch::Approx(mu_l).epsilon(1e-6));
}

TEST_CASE("coexistence densities lie outside spinodal densities", "[phase_diagram]") {
  auto eos = make_lj_eos();
  auto sp = CONFIG.find_spinodal(eos);
  auto coex = CONFIG.find_coexistence(eos);
  REQUIRE(sp.has_value());
  REQUIRE(coex.has_value());

  CHECK(coex->rho_vapor < sp->rho_low);
  CHECK(coex->rho_liquid > sp->rho_high);
}

TEST_CASE("find_coexistence returns nullopt for pure hard spheres", "[phase_diagram]") {
  auto eos = make_bulk_thermodynamics(
      SPECIES, make_bulk_weights(Rosenfeld{}, {}, 1.0)
  );

  auto result = CONFIG.find_coexistence(eos);
  CHECK_FALSE(result.has_value());
}

// Higher temperature reduces the coexistence gap

TEST_CASE("higher temperature reduces coexistence density gap", "[phase_diagram]") {
  auto eos_low = make_lj_eos_at(0.7);
  auto eos_high = make_lj_eos_at(0.8);

  auto coex_low = CONFIG.find_coexistence(eos_low);
  auto coex_high = CONFIG.find_coexistence(eos_high);

  REQUIRE(coex_low.has_value());
  REQUIRE(coex_high.has_value());

  double gap_low = coex_low->rho_liquid - coex_low->rho_vapor;
  double gap_high = coex_high->rho_liquid - coex_high->rho_vapor;

  CHECK(gap_high < gap_low);
}

// trace_coexistence

TEST_CASE("trace_coexistence follows the coexistence curve", "[phase_diagram]") {
  auto eos = make_lj_eos_at(0.75);
  auto coex = CONFIG.find_coexistence(eos);
  REQUIRE(coex.has_value());

  EoSFactory factory = [](double kT) -> BulkThermodynamics { return make_lj_eos_at(kT); };

  dft::algorithms::continuation::Continuation cont_config{
      .initial_step = 0.005,
      .max_step = 0.02,
      .min_step = 1e-4,
      .newton = {.max_iterations = 100, .tolerance = 1e-8},
  };

  auto curve = _internal::trace_coexistence(
      *coex,
      0.75,
      factory,
      cont_config,
      [](const dft::algorithms::continuation::CurvePoint& p) { return p.lambda > 0.8; }
  );

  // Should have traced at least a few points
  CHECK(curve.size() >= 2);

  // Each point on the curve should satisfy equal pressure and mu
  for (const auto& pt : curve) {
    auto eos_at_T = make_lj_eos_at(pt.lambda);
    double pv = eos_at_T.pressure(arma::vec{pt.x(0)});
    double pl = eos_at_T.pressure(arma::vec{pt.x(1)});
    CHECK(pv == Catch::Approx(pl).epsilon(1e-4));
  }
}

// binodal

static const EoSFactory EOS_FACTORY = [](double kT) -> BulkThermodynamics { return make_lj_eos_at(kT); };

TEST_CASE("binodal returns a coexistence curve for LJ system", "[phase_diagram]") {
  PhaseDiagramBuilder pd_config{
      .start_temperature = 0.6,
      .search = CONFIG,
  };

  auto result = pd_config.binodal(EOS_FACTORY);

  REQUIRE(result.has_value());
  CHECK(result->temperature.n_elem >= 5);
  CHECK(result->rho_vapor.n_elem == result->temperature.n_elem);
  CHECK(result->rho_liquid.n_elem == result->temperature.n_elem);
  CHECK(result->critical_temperature > 0.7);
  CHECK(result->critical_density > 0.0);
}

TEST_CASE("binodal vapor density is less than liquid at all temperatures", "[phase_diagram]") {
  PhaseDiagramBuilder pd_config{
      .start_temperature = 0.6,
      .search = CONFIG,
  };

  auto result = pd_config.binodal(EOS_FACTORY);
  REQUIRE(result.has_value());

  for (arma::uword i = 0; i < result->temperature.n_elem; ++i) {
    CHECK(result->rho_vapor(i) < result->rho_liquid(i));
  }
}

TEST_CASE("binodal temperature is monotonically increasing", "[phase_diagram]") {
  PhaseDiagramBuilder pd_config{
      .start_temperature = 0.6,
      .search = CONFIG,
  };

  auto result = pd_config.binodal(EOS_FACTORY);
  REQUIRE(result.has_value());

  for (arma::uword i = 1; i < result->temperature.n_elem; ++i) {
    CHECK(result->temperature(i) >= result->temperature(i - 1));
  }
}

TEST_CASE("binodal satisfies equal pressure along the curve", "[phase_diagram]") {
  PhaseDiagramBuilder pd_config{
      .start_temperature = 0.6,
      .search = CONFIG,
  };

  auto result = pd_config.binodal(EOS_FACTORY);
  REQUIRE(result.has_value());

  for (arma::uword i = 0; i < result->temperature.n_elem; ++i) {
    auto eos_at_T = EOS_FACTORY(result->temperature(i));
    double pv = eos_at_T.pressure(arma::vec{result->rho_vapor(i)});
    double pl = eos_at_T.pressure(arma::vec{result->rho_liquid(i)});
    CHECK(pv == Catch::Approx(pl).epsilon(1e-3));
  }
}

TEST_CASE("binodal returns nullopt for purely repulsive system", "[phase_diagram]") {
  EoSFactory hs_factory = [](double kT) -> BulkThermodynamics {
    return make_bulk_thermodynamics(SPECIES, make_bulk_weights(Rosenfeld{}, {}, kT));
  };

  auto result = PhaseDiagramBuilder{}.binodal(hs_factory);
  CHECK_FALSE(result.has_value());
}

// spinodal

TEST_CASE("spinodal returns a curve for LJ system", "[phase_diagram]") {
  PhaseDiagramBuilder pd_config{
      .start_temperature = 0.6,
      .search = CONFIG,
  };

  auto result = pd_config.spinodal_curve(EOS_FACTORY);

  REQUIRE(result.has_value());
  CHECK(result->temperature.n_elem >= 5);
  CHECK(result->rho_low.n_elem == result->temperature.n_elem);
  CHECK(result->rho_high.n_elem == result->temperature.n_elem);
  CHECK(result->critical_temperature > 0.7);
  CHECK(result->critical_density > 0.0);
}

TEST_CASE("spinodal low density is less than high density", "[phase_diagram]") {
  PhaseDiagramBuilder pd_config{
      .start_temperature = 0.6,
      .search = CONFIG,
  };

  auto result = pd_config.spinodal_curve(EOS_FACTORY);
  REQUIRE(result.has_value());

  for (arma::uword i = 0; i < result->temperature.n_elem; ++i) {
    CHECK(result->rho_low(i) < result->rho_high(i));
  }
}

TEST_CASE("spinodal lies inside the binodal", "[phase_diagram]") {
  PhaseDiagramBuilder pd_config{
      .start_temperature = 0.6,
      .search = CONFIG,
  };

  auto b = pd_config.binodal(EOS_FACTORY);
  auto s = pd_config.spinodal_curve(EOS_FACTORY);
  REQUIRE(b.has_value());
  REQUIRE(s.has_value());

  // At start temperature, spinodal should be inside binodal.
  CHECK(s->rho_low(0) > b->rho_vapor(0));
  CHECK(s->rho_high(0) < b->rho_liquid(0));
}

TEST_CASE("spinodal returns nullopt for purely repulsive system", "[phase_diagram]") {
  EoSFactory hs_factory = [](double kT) -> BulkThermodynamics {
    return make_bulk_thermodynamics(SPECIES, make_bulk_weights(Rosenfeld{}, {}, kT));
  };

  auto result = PhaseDiagramBuilder{}.spinodal_curve(hs_factory);
  CHECK_FALSE(result.has_value());
}

// phase_diagram

TEST_CASE("phase_diagram returns both binodal and spinodal", "[phase_diagram]") {
  PhaseDiagramBuilder pd_config{
      .start_temperature = 0.6,
      .search = CONFIG,
  };

  auto result = phase_diagram(EOS_FACTORY, pd_config);

  REQUIRE(result.has_value());
  CHECK_FALSE(result->binodal.temperature.is_empty());
  CHECK_FALSE(result->spinodal.temperature.is_empty());
  CHECK(result->critical_temperature > 0.7);
  CHECK(result->critical_density > 0.0);
}

TEST_CASE("phase_diagram returns nullopt for purely repulsive system", "[phase_diagram]") {
  EoSFactory hs_factory = [](double kT) -> BulkThermodynamics {
    return make_bulk_thermodynamics(SPECIES, make_bulk_weights(Rosenfeld{}, {}, kT));
  };

  auto result = phase_diagram(hs_factory);
  CHECK_FALSE(result.has_value());
}

// interpolate

TEST_CASE("interpolate returns phase boundaries at a known temperature", "[phase_diagram]") {
  PhaseDiagramBuilder pd_config{
      .start_temperature = 0.6,
      .search = CONFIG,
  };

  auto pd = phase_diagram(EOS_FACTORY, pd_config);
  REQUIRE(pd.has_value());

  auto pb = pd->interpolate(0.8);
  CHECK(pb.temperature == 0.8);
  CHECK(std::isfinite(pb.binodal_vapor));
  CHECK(std::isfinite(pb.binodal_liquid));
  CHECK(std::isfinite(pb.spinodal_low));
  CHECK(std::isfinite(pb.spinodal_high));
  CHECK(pb.binodal_vapor < pb.spinodal_low);
  CHECK(pb.spinodal_high < pb.binodal_liquid);
}

TEST_CASE("interpolate returns NaN above the critical temperature", "[phase_diagram]") {
  PhaseDiagramBuilder pd_config{
      .start_temperature = 0.6,
      .search = CONFIG,
  };

  auto pd = phase_diagram(EOS_FACTORY, pd_config);
  REQUIRE(pd.has_value());

  auto pb = pd->interpolate(pd->critical_temperature + 0.5);
  CHECK(std::isnan(pb.binodal_vapor));
  CHECK(std::isnan(pb.binodal_liquid));
}

TEST_CASE("interpolate returns NaN below the traced range", "[phase_diagram]") {
  PhaseDiagramBuilder pd_config{
      .start_temperature = 0.6,
      .search = CONFIG,
  };

  auto pd = phase_diagram(EOS_FACTORY, pd_config);
  REQUIRE(pd.has_value());

  auto pb = pd->interpolate(0.1);
  CHECK(std::isnan(pb.binodal_vapor));
}
