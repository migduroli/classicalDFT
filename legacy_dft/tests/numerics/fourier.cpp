#include "classicaldft_bits/numerics/fourier.h"

#include <cmath>
#include <complex>
#include <gtest/gtest.h>
#include <numeric>
#include <vector>

namespace fourier = dft::numerics::fourier;

// ── FourierTransform ────────────────────────────────────────────────────

TEST(FourierTransform, ConstructionAndZeros) {
  fourier::FourierTransform plan({4, 4, 4});

  EXPECT_EQ(plan.shape(), (std::vector<long>{4, 4, 4}));
  EXPECT_EQ(plan.total(), 64);
  EXPECT_EQ(plan.fourier_total(), 48);
  EXPECT_EQ(plan.real().size(), 64U);
  EXPECT_EQ(plan.fourier().size(), 48U);

  for (auto v : plan.real()) {
    EXPECT_DOUBLE_EQ(v, 0.0);
  }
}

TEST(FourierTransform, ZeroDimensionsThrow) {
  EXPECT_THROW(fourier::FourierTransform({0, 0, 0}), std::invalid_argument);
  EXPECT_THROW(fourier::FourierTransform({0, 4, 4}), std::invalid_argument);
}

TEST(FourierTransform, WrongShapeSizeThrows) {
  EXPECT_THROW(fourier::FourierTransform({4, 4}), std::invalid_argument);
  EXPECT_THROW(fourier::FourierTransform({4, 4, 4, 4}), std::invalid_argument);
}

TEST(FourierTransform, RoundTrip) {
  fourier::FourierTransform plan({8, 8, 8});

  auto r = plan.real();
  auto s = plan.shape();
  for (long i = 0; i < s[0]; ++i) {
    for (long j = 0; j < s[1]; ++j) {
      for (long k = 0; k < s[2]; ++k) {
        r[k + s[2] * (j + s[1] * i)] = std::sin(2.0 * M_PI * static_cast<double>(i) / s[0]);
      }
    }
  }

  std::vector<double> original(r.begin(), r.end());

  plan.forward();
  plan.backward();
  plan.scale(1.0 / static_cast<double>(plan.total()));

  auto result = plan.real();
  for (std::size_t i = 0; i < original.size(); ++i) {
    EXPECT_NEAR(result[i], original[i], 1e-12);
  }
}

TEST(FourierTransform, ParsevalTheorem) {
  fourier::FourierTransform plan({4, 4, 8});

  auto r = plan.real();
  for (std::size_t i = 0; i < r.size(); ++i) {
    r[i] = static_cast<double>(i % 7) - 3.0;
  }

  double real_energy = 0.0;
  for (auto v : plan.real()) {
    real_energy += v * v;
  }

  plan.forward();

  auto s = plan.shape();
  auto f = plan.fourier();
  double fourier_energy = 0.0;
  const long nz_half = s[2] / 2 + 1;
  for (long i = 0; i < s[0]; ++i) {
    for (long j = 0; j < s[1]; ++j) {
      for (long k = 0; k < nz_half; ++k) {
        auto idx = k + nz_half * (j + s[1] * i);
        double mag2 = std::norm(f[idx]);
        if (k == 0 || k == s[2] / 2) {
          fourier_energy += mag2;
        } else {
          fourier_energy += 2.0 * mag2;
        }
      }
    }
  }

  double n = static_cast<double>(plan.total());
  EXPECT_NEAR(fourier_energy / n, real_energy, 1e-10);
}

TEST(FourierTransform, MoveConstruction) {
  fourier::FourierTransform plan({4, 4, 4});
  plan.real()[0] = 42.0;

  fourier::FourierTransform moved(std::move(plan));
  EXPECT_EQ(moved.shape(), (std::vector<long>{4, 4, 4}));
  EXPECT_DOUBLE_EQ(moved.real()[0], 42.0);
}

TEST(FourierTransform, MoveAssignment) {
  fourier::FourierTransform plan1({4, 4, 4});
  fourier::FourierTransform plan2({2, 2, 2});
  plan1.real()[0] = 99.0;

  plan2 = std::move(plan1);
  EXPECT_EQ(plan2.shape(), (std::vector<long>{4, 4, 4}));
  EXPECT_EQ(plan2.total(), 64);
  EXPECT_DOUBLE_EQ(plan2.real()[0], 99.0);
}

TEST(FourierTransform, Scale) {
  fourier::FourierTransform plan({2, 2, 2});
  auto r = plan.real();
  for (std::size_t i = 0; i < r.size(); ++i) {
    r[i] = 1.0;
  }
  plan.scale(3.0);
  for (auto v : plan.real()) {
    EXPECT_DOUBLE_EQ(v, 3.0);
  }
}

TEST(FourierTransform, ZerosAfterWrite) {
  fourier::FourierTransform plan({4, 4, 4});
  plan.real()[0] = 42.0;
  plan.zeros();
  for (auto v : plan.real()) {
    EXPECT_DOUBLE_EQ(v, 0.0);
  }
}

TEST(FourierTransform, ConstAccessors) {
  fourier::FourierTransform plan({4, 4, 4});
  plan.real()[0] = 5.0;
  plan.forward();

  const auto& const_plan = plan;
  EXPECT_EQ(const_plan.real().size(), 64U);
  EXPECT_DOUBLE_EQ(const_plan.real()[0], 5.0);
  EXPECT_EQ(const_plan.fourier().size(), 48U);
  EXPECT_EQ(const_plan.shape(), (std::vector<long>{4, 4, 4}));
}

// ── FourierConvolution ──────────────────────────────────────────────────────

TEST(FourierConvolution, DeltaConvolution) {
  fourier::FourierConvolution conv({4, 4, 4});

  auto a = conv.input_a();
  for (std::size_t i = 0; i < a.size(); ++i) {
    a[i] = static_cast<double>(i + 1);
  }

  auto b = conv.input_b();
  std::fill(b.begin(), b.end(), 0.0);
  b[0] = 1.0;

  conv.execute();

  auto result = conv.result();
  for (std::size_t i = 0; i < result.size(); ++i) {
    EXPECT_NEAR(result[i], a[i], 1e-10);
  }
}

TEST(FourierConvolution, ConstantConvolution) {
  fourier::FourierConvolution conv({4, 4, 4});

  auto a = conv.input_a();
  auto b = conv.input_b();
  std::fill(a.begin(), a.end(), 2.0);
  std::fill(b.begin(), b.end(), 3.0);

  conv.execute();

  double expected = 2.0 * 3.0 * static_cast<double>(conv.total());
  auto result = conv.result();
  for (auto v : result) {
    EXPECT_NEAR(v, expected, 1e-10);
  }
}

TEST(FourierConvolution, ShapeAccessor) {
  fourier::FourierConvolution conv({4, 4, 4});
  EXPECT_EQ(conv.shape(), (std::vector<long>{4, 4, 4}));
  EXPECT_EQ(conv.total(), 64);
}
