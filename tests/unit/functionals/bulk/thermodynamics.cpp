#include "dft/functionals/bulk/thermodynamics.hpp"

#include "dft/functionals/fmt/models.hpp"
#include "dft/functionals/functionals.hpp"
#include "dft/physics/interactions.hpp"
#include "dft/physics/model.hpp"
#include "dft/physics/potentials.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>
#include <numbers>

using namespace dft::functionals;
using namespace dft::functionals::bulk;
using namespace dft::functionals::fmt;
using namespace dft::physics;
using namespace dft::physics::potentials;
using dft::Species;

static constexpr double DX = 0.1;
static constexpr double DIAMETER = 1.0;
static constexpr double KT = 1.0;
static const dft::Grid GRID = dft::Grid{.dx = DX, .box_size = {1.6, 1.6, 1.6}, .shape = {16, 16, 16}};
static const std::vector<Species> SPECIES = {{.name = "HS", .hard_sphere_diameter = DIAMETER}};

static auto make_hs_weights(const FMTModel& model = Rosenfeld{}) -> Weights {
  Model m{.grid = GRID, .species = SPECIES, .temperature = KT};
  return make_weights(model, m);
}

// Ideal gas

TEST_CASE("ideal free energy density of unit density is -1", "[bulk][thermodynamics]") {
  arma::vec rho = {1.0};
  CHECK(ideal::free_energy_density(rho) == Catch::Approx(-1.0).margin(1e-14));
}

TEST_CASE("ideal free energy density is rho*(ln(rho) - 1)", "[bulk][thermodynamics]") {
  double rho0 = 0.5;
  arma::vec rho = {rho0};
  double expected = rho0 * (std::log(rho0) - 1.0);
  CHECK(ideal::free_energy_density(rho) == Catch::Approx(expected).margin(1e-14));
}

TEST_CASE("ideal chemical potential is ln(rho)", "[bulk][thermodynamics]") {
  double rho0 = 0.3;
  CHECK(ideal::chemical_potential(rho0) == Catch::Approx(std::log(rho0)).margin(1e-14));
}

TEST_CASE("ideal free energy density is additive for two species", "[bulk][thermodynamics]") {
  arma::vec rho = {0.3, 0.2};
  double expected = 0.3 * (std::log(0.3) - 1.0) + 0.2 * (std::log(0.2) - 1.0);
  CHECK(ideal::free_energy_density(rho) == Catch::Approx(expected).margin(1e-14));
}

// Hard-sphere excess: single species should match FMT single-species function

TEST_CASE("hard sphere bulk free energy matches fmt::free_energy_density", "[bulk][thermodynamics]") {
  double rho0 = 0.5;
  arma::vec rho = {rho0};

  auto [label, model] = GENERATE(table<std::string, FMTModel>({
      {"Rosenfeld", Rosenfeld{}},
      {"RSLT", RSLT{}},
      {"WhiteBearI", WhiteBearI{}},
      {"WhiteBearII", WhiteBearII{}},
  }));

  CAPTURE(label);
  double bulk_val = hard_sphere::free_energy_density(model, rho, SPECIES);
  double fmt_val = model.free_energy_density(rho0, DIAMETER);
  CHECK(bulk_val == Catch::Approx(fmt_val).epsilon(1e-14));
}

TEST_CASE("hard sphere excess mu matches fmt::excess_chemical_potential for single species", "[bulk][thermodynamics]") {
  double rho0 = 0.4;
  arma::vec rho = {rho0};

  auto [label, model] = GENERATE(table<std::string, FMTModel>({
      {"Rosenfeld", Rosenfeld{}},
      {"WhiteBearII", WhiteBearII{}},
  }));

  CAPTURE(label);
  double bulk_mu = hard_sphere::excess_chemical_potential(model, rho, SPECIES, 0);
  double fmt_mu = model.excess_chemical_potential(rho0, DIAMETER);
  CHECK(bulk_mu == Catch::Approx(fmt_mu).epsilon(1e-14));
}

TEST_CASE("hard sphere excess vanishes at zero density", "[bulk][thermodynamics]") {
  arma::vec rho = {1e-15};
  double f = hard_sphere::free_energy_density(Rosenfeld{}, rho, SPECIES);
  CHECK(std::abs(f) < 1e-10);
}

// Mean-field: self-interaction

TEST_CASE("mean field free energy density is 0.5 * a_vdw * rho^2 for self", "[bulk][thermodynamics]") {
  auto pot = make_lennard_jones(1.0, 1.0, 2.5);
  std::vector<Interaction> interactions = {
      {.species_i = 0, .species_j = 0, .potential = pot},
  };
  auto mfw = make_mean_field_weights(GRID, interactions, KT);

  double rho0 = 0.3;
  arma::vec rho = {rho0};
  double f = mean_field::free_energy_density(mfw, rho);
  double expected = 0.5 * mfw.interactions[0].a_vdw * rho0 * rho0;
  CHECK(f == Catch::Approx(expected).epsilon(1e-14));
}

TEST_CASE("mean field chemical potential is a_vdw * rho for self", "[bulk][thermodynamics]") {
  auto pot = make_lennard_jones(1.0, 1.0, 2.5);
  std::vector<Interaction> interactions = {
      {.species_i = 0, .species_j = 0, .potential = pot},
  };
  auto mfw = make_mean_field_weights(GRID, interactions, KT);

  double rho0 = 0.3;
  arma::vec rho = {rho0};
  double mu = mean_field::chemical_potential(mfw, rho, 0);
  double expected = mfw.interactions[0].a_vdw * rho0;
  CHECK(mu == Catch::Approx(expected).epsilon(1e-14));
}

TEST_CASE("mean field free energy density is zero with no interactions", "[bulk][thermodynamics]") {
  MeanFieldWeights mfw;
  arma::vec rho = {0.5};
  CHECK(mean_field::free_energy_density(mfw, rho) == 0.0);
}

// Composed: thermodynamic consistency

TEST_CASE("pressure equals rho*mu - f for single species (Gibbs-Duhem)", "[bulk][thermodynamics]") {
  double rho0 = 0.3;
  arma::vec rho = {rho0};
  auto w = make_hs_weights(Rosenfeld{});

  double p = pressure(rho, SPECIES, w);
  double f = free_energy_density(rho, SPECIES, w);
  double mu = chemical_potential(rho, SPECIES, w, 0);

  CHECK(p == Catch::Approx(rho0 * mu - f).epsilon(1e-12));
}

TEST_CASE("grand potential density is negative of pressure", "[bulk][thermodynamics]") {
  double rho0 = 0.4;
  arma::vec rho = {rho0};
  auto w = make_hs_weights(WhiteBearII{});

  double omega = grand_potential_density(rho, SPECIES, w);
  double p = pressure(rho, SPECIES, w);
  CHECK(omega == Catch::Approx(-p).epsilon(1e-14));
}

TEST_CASE("pressure is positive for moderate density", "[bulk][thermodynamics]") {
  double rho0 = 0.4;
  arma::vec rho = {rho0};
  auto w = make_hs_weights();

  double p = pressure(rho, SPECIES, w);
  CHECK(p > 0.0);
}

TEST_CASE("pressure of ideal gas is rho", "[bulk][thermodynamics]") {
  // At very low density, excess contributions vanish.
  double rho0 = 1e-6;
  arma::vec rho = {rho0};
  auto w = make_hs_weights();

  double p = pressure(rho, SPECIES, w);
  CHECK(p == Catch::Approx(rho0).epsilon(1e-4));
}

// chemical_potentials returns all species

TEST_CASE("chemical_potentials vector matches individual calls", "[bulk][thermodynamics]") {
  std::vector<Species> species2 = {
      {.name = "A", .hard_sphere_diameter = 1.0},
      {.name = "B", .hard_sphere_diameter = 0.8},
  };
  Model model{.grid = GRID, .species = species2, .temperature = KT};
  auto w = make_weights(Rosenfeld{}, model);

  arma::vec rho = {0.3, 0.2};
  auto mu = chemical_potentials(rho, species2, w);

  CHECK(mu(0) == Catch::Approx(chemical_potential(rho, species2, w, 0)).epsilon(1e-14));
  CHECK(mu(1) == Catch::Approx(chemical_potential(rho, species2, w, 1)).epsilon(1e-14));
}

// Multi-species Gibbs-Duhem: P = sum_i rho_i * mu_i - f

TEST_CASE("pressure satisfies Gibbs-Duhem for two species", "[bulk][thermodynamics]") {
  std::vector<Species> species2 = {
      {.name = "A", .hard_sphere_diameter = 1.0},
      {.name = "B", .hard_sphere_diameter = 0.8},
  };
  Model model{.grid = GRID, .species = species2, .temperature = KT};
  auto w = make_weights(Rosenfeld{}, model);

  arma::vec rho = {0.3, 0.2};
  double p = pressure(rho, species2, w);
  double f = free_energy_density(rho, species2, w);
  auto mu = chemical_potentials(rho, species2, w);

  double expected = arma::dot(rho, mu) - f;
  CHECK(p == Catch::Approx(expected).epsilon(1e-12));
}

// With mean-field: Gibbs-Duhem still holds

TEST_CASE("pressure satisfies Gibbs-Duhem with mean-field", "[bulk][thermodynamics]") {
  auto pot = make_lennard_jones(1.0, 1.0, 2.5);
  std::vector<Interaction> interactions = {
      {.species_i = 0, .species_j = 0, .potential = pot},
  };
  Model model{.grid = GRID, .species = SPECIES, .interactions = interactions, .temperature = KT};
  auto w = make_weights(Rosenfeld{}, model);

  double rho0 = 0.3;
  arma::vec rho = {rho0};
  double p = pressure(rho, SPECIES, w);
  double f = free_energy_density(rho, SPECIES, w);
  double mu = chemical_potential(rho, SPECIES, w, 0);

  CHECK(p == Catch::Approx(rho0 * mu - f).epsilon(1e-12));
}

// Mean-field lowers pressure (attractive interactions)

TEST_CASE("mean-field attractive interactions lower pressure", "[bulk][thermodynamics]") {
  auto pot = make_lennard_jones(1.0, 1.0, 2.5);
  std::vector<Interaction> interactions = {
      {.species_i = 0, .species_j = 0, .potential = pot},
  };

  Model model_no_mf{.grid = GRID, .species = SPECIES, .temperature = KT};
  Model model_mf{.grid = GRID, .species = SPECIES, .interactions = interactions, .temperature = KT};
  auto w_no_mf = make_weights(Rosenfeld{}, model_no_mf);
  auto w_mf = make_weights(Rosenfeld{}, model_mf);

  double rho0 = 0.3;
  arma::vec rho = {rho0};

  double p_no_mf = pressure(rho, SPECIES, w_no_mf);
  double p_mf = pressure(rho, SPECIES, w_mf);

  CHECK(p_mf < p_no_mf);
}

// Mean-field: cross-interaction

TEST_CASE("mean field cross-interaction free energy density", "[bulk][thermodynamics]") {
  auto pot = make_lennard_jones(1.0, 1.0, 2.5);
  std::vector<Species> species2 = {
      {.name = "A", .hard_sphere_diameter = 1.0},
      {.name = "B", .hard_sphere_diameter = 0.8},
  };
  std::vector<Interaction> interactions = {
      {.species_i = 0, .species_j = 1, .potential = pot},
  };
  auto mfw = make_mean_field_weights(GRID, interactions, KT);

  arma::vec rho = {0.3, 0.2};
  double f = mean_field::free_energy_density(mfw, rho);
  double expected = mfw.interactions[0].a_vdw * rho(0) * rho(1);
  CHECK(f == Catch::Approx(expected).epsilon(1e-14));
}

TEST_CASE("mean field cross-interaction chemical potential", "[bulk][thermodynamics]") {
  auto pot = make_lennard_jones(1.0, 1.0, 2.5);
  std::vector<Species> species2 = {
      {.name = "A", .hard_sphere_diameter = 1.0},
      {.name = "B", .hard_sphere_diameter = 0.8},
  };
  std::vector<Interaction> interactions = {
      {.species_i = 0, .species_j = 1, .potential = pot},
  };
  auto mfw = make_mean_field_weights(GRID, interactions, KT);

  arma::vec rho = {0.3, 0.2};
  // mu for species 0 = 0.5 * a_vdw * rho(1)
  double mu0 = mean_field::chemical_potential(mfw, rho, 0);
  CHECK(mu0 == Catch::Approx(0.5 * mfw.interactions[0].a_vdw * rho(1)).epsilon(1e-14));
  // mu for species 1 = 0.5 * a_vdw * rho(0)
  double mu1 = mean_field::chemical_potential(mfw, rho, 1);
  CHECK(mu1 == Catch::Approx(0.5 * mfw.interactions[0].a_vdw * rho(0)).epsilon(1e-14));
}
