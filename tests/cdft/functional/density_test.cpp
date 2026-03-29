#include "cdft/functional/density.hpp"

#include <gtest/gtest.h>

#include <cmath>

namespace cdft::functional {

  TEST(DensityFieldTest, Construction) {
    arma::rowvec3 box = {2.0, 2.0, 2.0};
    DensityField d(0.5, box);
    EXPECT_DOUBLE_EQ(d.spacing(), 0.5);
    EXPECT_EQ(d.shape().size(), 3u);
    EXPECT_EQ(d.shape()[0], 4);
    EXPECT_EQ(d.shape()[1], 4);
    EXPECT_EQ(d.shape()[2], 4);
    EXPECT_EQ(d.size(), 64u);
  }

  TEST(DensityFieldTest, CellVolume) {
    arma::rowvec3 box = {2.0, 2.0, 2.0};
    DensityField d(0.5, box);
    EXPECT_DOUBLE_EQ(d.cell_volume(), 0.125);
  }

  TEST(DensityFieldTest, SetAndGet) {
    arma::rowvec3 box = {2.0, 2.0, 2.0};
    DensityField d(0.5, box);
    arma::vec rho(64, arma::fill::ones);
    d.set(rho);
    EXPECT_DOUBLE_EQ(d.values()(0), 1.0);
  }

  TEST(DensityFieldTest, SetSingleValue) {
    arma::rowvec3 box = {2.0, 2.0, 2.0};
    DensityField d(0.5, box);
    d.set(0, 5.0);
    EXPECT_DOUBLE_EQ(d.values()(0), 5.0);
  }

  TEST(DensityFieldTest, NumberOfAtoms) {
    arma::rowvec3 box = {2.0, 2.0, 2.0};
    DensityField d(0.5, box);
    arma::vec rho(64, arma::fill::value(1.0));
    d.set(rho);
    double n = d.number_of_atoms();
    EXPECT_NEAR(n, 64 * 0.125, 1e-10);
  }

  TEST(DensityFieldTest, Scale) {
    arma::rowvec3 box = {2.0, 2.0, 2.0};
    DensityField d(0.5, box);
    arma::vec rho(64, arma::fill::value(2.0));
    d.set(rho);
    d.scale(0.5);
    EXPECT_DOUBLE_EQ(d.values()(0), 1.0);
  }

  TEST(DensityFieldTest, InvalidSpacingThrows) {
    arma::rowvec3 box = {2.0, 2.0, 2.0};
    EXPECT_THROW(DensityField(-0.5, box), std::invalid_argument);
  }

  TEST(SpeciesTest, Construction) {
    arma::rowvec3 box = {2.0, 2.0, 2.0};
    DensityField d(0.5, box);
    Species sp(std::move(d), 1.0);
    EXPECT_DOUBLE_EQ(sp.chemical_potential(), 1.0);
    EXPECT_EQ(sp.force().n_elem, 64u);
  }

  TEST(SpeciesTest, AliasRoundtrip) {
    arma::rowvec3 box = {2.0, 2.0, 2.0};
    DensityField d(0.5, box);
    arma::vec rho(64, arma::fill::value(0.1));
    d.set(rho);
    Species sp(std::move(d));

    arma::vec alias = sp.density_alias();
    sp.set_density_from_alias(alias);
    const arma::vec& recovered = sp.density().values();
    for (arma::uword i = 0; i < recovered.n_elem; ++i) {
      EXPECT_NEAR(recovered(i), 0.1, 1e-6);
    }
  }

}  // namespace cdft::functional
