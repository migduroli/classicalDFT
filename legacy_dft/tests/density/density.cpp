#include "classicaldft_bits/density/density.h"

#include <cmath>
#include <filesystem>
#include <gtest/gtest.h>
#include <numbers>

using namespace dft::density;

// ── Construction ────────────────────────────────────────────────────────────

TEST(Density, ConstructsWithCorrectSize) {
  Density d(0.5, {2.0, 3.0, 4.0});
  // shape: 4 x 6 x 8 = 192
  EXPECT_EQ(d.size(), 192U);
  EXPECT_EQ(d.shape()[0], 4);
  EXPECT_EQ(d.shape()[1], 6);
  EXPECT_EQ(d.shape()[2], 8);
}

TEST(Density, ThrowsOnNonPositiveDx) {
  EXPECT_THROW(Density(0.0, {2.0, 2.0, 2.0}), std::invalid_argument);
  EXPECT_THROW(Density(-0.5, {2.0, 2.0, 2.0}), std::invalid_argument);
}

TEST(Density, ThrowsOnNonCommensurateBox) {
  EXPECT_THROW(Density(0.3, {1.0, 1.0, 1.0}), std::invalid_argument);
}

TEST(Density, DensityInitializedToZero) {
  Density d(1.0, {3.0, 3.0, 3.0});
  EXPECT_DOUBLE_EQ(d.min(), 0.0);
  EXPECT_DOUBLE_EQ(d.max(), 0.0);
  EXPECT_DOUBLE_EQ(arma::accu(d.values()), 0.0);
}

TEST(Density, ExternalFieldInitializedToZero) {
  Density d(1.0, {3.0, 3.0, 3.0});
  EXPECT_DOUBLE_EQ(arma::accu(d.external_field()), 0.0);
}

TEST(Density, MeshAccessorsConsistent) {
  Density d(0.5, {2.0, 3.0, 4.0});
  EXPECT_EQ(d.shape()[0], 4);
  EXPECT_EQ(d.shape()[1], 6);
  EXPECT_EQ(d.shape()[2], 8);
  EXPECT_DOUBLE_EQ(d.dx(), 0.5);
  EXPECT_DOUBLE_EQ(d.cell_volume(), 0.125);
}

// ── Set / get ───────────────────────────────────────────────────────────────

TEST(Density, SetVectorCopiesValues) {
  Density d(1.0, {3.0, 3.0, 3.0});
  arma::vec rho(d.size(), arma::fill::ones);
  d.set(rho);
  EXPECT_DOUBLE_EQ(d.values()(0), 1.0);
  EXPECT_DOUBLE_EQ(d.values()(d.size() - 1), 1.0);
}

TEST(Density, SetVectorThrowsOnSizeMismatch) {
  Density d(1.0, {3.0, 3.0, 3.0});
  arma::vec rho(d.size() + 1, arma::fill::zeros);
  EXPECT_THROW(d.set(rho), std::invalid_argument);
}

TEST(Density, SetSingleElement) {
  Density d(1.0, {3.0, 3.0, 3.0});
  d.set(0, 42.0);
  EXPECT_DOUBLE_EQ(d.values()(0), 42.0);
  // rest still zero
  EXPECT_DOUBLE_EQ(d.values()(1), 0.0);
}

TEST(Density, SetSingleElementThrowsOutOfRange) {
  Density d(1.0, {3.0, 3.0, 3.0});
  EXPECT_THROW(d.set(d.size(), 1.0), std::out_of_range);
}

TEST(Density, ScaleMultipliesAllElements) {
  Density d(1.0, {3.0, 3.0, 3.0});
  arma::vec rho(d.size(), arma::fill::ones);
  d.set(rho);
  d.scale(3.0);
  EXPECT_DOUBLE_EQ(d.values()(0), 3.0);
  EXPECT_DOUBLE_EQ(d.values()(d.size() - 1), 3.0);
}

// ── Number of atoms ─────────────────────────────────────────────────────────

TEST(Density, NumberOfAtomsUniformDensity) {
  double dx = 0.5;
  arma::rowvec3 box = {4.0, 4.0, 4.0};
  Density d(dx, box);

  double rho0 = 0.8;
  arma::vec rho(d.size(), arma::fill::value(rho0));
  d.set(rho);

  double expected = rho0 * box(0) * box(1) * box(2);
  EXPECT_NEAR(d.number_of_atoms(), expected, 1e-12);
}

TEST(Density, NumberOfAtomsZeroDensity) {
  Density d(1.0, {3.0, 3.0, 3.0});
  EXPECT_DOUBLE_EQ(d.number_of_atoms(), 0.0);
}

// ── External field energy ───────────────────────────────────────────────────

TEST(Density, ExternalFieldEnergyUniform) {
  double dx = 1.0;
  arma::rowvec3 box = {3.0, 3.0, 3.0};
  Density d(dx, box);

  arma::vec rho(d.size(), arma::fill::value(2.0));
  d.set(rho);
  d.external_field().fill(0.5);

  double dV = d.cell_volume();
  double expected = arma::dot(rho, d.external_field()) * dV;
  EXPECT_NEAR(d.external_field_energy(), expected, 1e-12);
}

TEST(Density, ExternalFieldEnergyZeroField) {
  Density d(1.0, {3.0, 3.0, 3.0});
  arma::vec rho(d.size(), arma::fill::value(1.0));
  d.set(rho);
  EXPECT_DOUBLE_EQ(d.external_field_energy(), 0.0);
}

// ── FFT ─────────────────────────────────────────────────────────────────────

TEST(Density, ForwardFFTConstantDensityOnlyDC) {
  double dx = 1.0;
  arma::rowvec3 box = {4.0, 4.0, 4.0};
  Density d(dx, box);

  double rho0 = 1.5;
  arma::vec rho(d.size(), arma::fill::value(rho0));
  d.set(rho);
  d.forward_fft();

  auto fourier = d.fft().fourier();
  // DC component (index 0) should be rho0 * N
  double dc_mag = std::abs(fourier[0]);
  EXPECT_NEAR(dc_mag, rho0 * d.size(), 1e-10);

  // All other components should be approximately zero
  for (long k = 1; k < d.fft().fourier_total(); ++k) {
    EXPECT_NEAR(std::abs(fourier[k]), 0.0, 1e-10);
  }
}

TEST(Density, ForwardFFTSineWaveHasPeaks) {
  double dx = 1.0;
  arma::rowvec3 box = {8.0, 8.0, 8.0};
  Density d(dx, box);

  long nx = 8, ny = 8, nz = 8;
  arma::vec rho(d.size());
  // Sine wave along z with one full period
  for (long ix = 0; ix < nx; ++ix) {
    for (long iy = 0; iy < ny; ++iy) {
      for (long iz = 0; iz < nz; ++iz) {
        rho(d.flat_index(ix, iy, iz)) = std::sin(2.0 * std::numbers::pi * iz / nz);
      }
    }
  }
  d.set(rho);
  d.forward_fft();

  auto fourier = d.fft().fourier();
  // DC should be near zero
  EXPECT_NEAR(std::abs(fourier[0]), 0.0, 1e-10);

  // There should be nonzero components (the sine peaks)
  double max_mag = 0.0;
  for (long k = 1; k < d.fft().fourier_total(); ++k) {
    max_mag = std::max(max_mag, std::abs(fourier[k]));
  }
  EXPECT_GT(max_mag, 1.0);
}

// ── Center of mass ──────────────────────────────────────────────────────────

TEST(Density, CenterOfMassUniformDensity) {
  double dx = 1.0;
  arma::rowvec3 box = {4.0, 4.0, 4.0};
  Density d(dx, box);

  arma::vec rho(d.size(), arma::fill::value(1.0));
  d.set(rho);

  auto com = d.center_of_mass();
  // Uniform density on grid 0,1,2,3: COM = 1.5
  EXPECT_NEAR(com(0), 1.5, 1e-12);
  EXPECT_NEAR(com(1), 1.5, 1e-12);
  EXPECT_NEAR(com(2), 1.5, 1e-12);
}

TEST(Density, CenterOfMassSinglePeak) {
  double dx = 1.0;
  arma::rowvec3 box = {4.0, 4.0, 4.0};
  Density d(dx, box);

  // Place all mass at grid position (ix=2, iy=1, iz=3)
  // Physical coords: (2.0, 1.0, 3.0)
  d.set(d.flat_index(2, 1, 3), 1.0);

  auto com = d.center_of_mass();
  EXPECT_NEAR(com(0), 2.0, 1e-12);
  EXPECT_NEAR(com(1), 1.0, 1e-12);
  EXPECT_NEAR(com(2), 3.0, 1e-12);
}

// ── Min / max ───────────────────────────────────────────────────────────────

TEST(Density, MinMaxAfterSet) {
  Density d(1.0, {3.0, 3.0, 3.0});
  d.set(0, -5.0);
  d.set(1, 10.0);
  EXPECT_DOUBLE_EQ(d.min(), -5.0);
  EXPECT_DOUBLE_EQ(d.max(), 10.0);
}

// ── Save / load ─────────────────────────────────────────────────────────────

TEST(Density, SaveLoadRoundTrip) {
  double dx = 1.0;
  arma::rowvec3 box = {3.0, 3.0, 3.0};
  Density d1(dx, box);

  arma::vec rho = arma::randu(d1.size());
  d1.set(rho);

  std::string tmpfile = "density_test_roundtrip.bin";
  d1.save(tmpfile);

  Density d2(dx, box);
  d2.load(tmpfile);

  for (arma::uword i = 0; i < d1.size(); ++i) {
    EXPECT_DOUBLE_EQ(d1.values()(i), d2.values()(i));
  }

  std::filesystem::remove(tmpfile);
}

TEST(Density, LoadThrowsOnBadFile) {
  Density d(1.0, {3.0, 3.0, 3.0});
  EXPECT_THROW(d.load("nonexistent_file_xyz.bin"), std::runtime_error);
}

TEST(Density, LoadThrowsOnSizeMismatch) {
  Density d1(1.0, {3.0, 3.0, 3.0});
  d1.values().fill(1.0);
  std::string tmpfile = "density_test_sizemismatch.bin";
  d1.save(tmpfile);

  Density d2(1.0, {4.0, 4.0, 4.0});  // different size
  EXPECT_THROW(d2.load(tmpfile), std::invalid_argument);
  std::filesystem::remove(tmpfile);
}

// ── Mutable access ──────────────────────────────────────────────────────────

TEST(Density, MutableValuesAccess) {
  Density d(1.0, {3.0, 3.0, 3.0});
  d.values().fill(7.0);
  EXPECT_DOUBLE_EQ(d.values()(0), 7.0);
  // 27 points * 7.0 * dV(1.0) = 189.0
  EXPECT_NEAR(d.number_of_atoms(), 7.0 * d.size() * d.cell_volume(), 1e-12);
}

TEST(Density, MutableExternalFieldAccess) {
  Density d(1.0, {3.0, 3.0, 3.0});
  d.external_field().fill(3.0);
  d.values().fill(1.0);
  EXPECT_NEAR(d.external_field_energy(), 3.0 * d.size() * d.cell_volume(), 1e-12);
}
