#include "cdft/numerics/fourier.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <numbers>

namespace cdft::numerics {

  TEST(FourierTransformTest, Construction) {
    FourierTransform ft({4, 4, 4});
    EXPECT_EQ(ft.total(), 64);
    EXPECT_EQ(ft.fourier_total(), 4 * 4 * 3);
  }

  TEST(FourierTransformTest, ZerosInitialized) {
    FourierTransform ft({4, 4, 4});
    auto r = ft.real();
    for (auto v : r) EXPECT_DOUBLE_EQ(v, 0.0);
  }

  TEST(FourierTransformTest, ForwardBackwardRoundtrip) {
    FourierTransform ft({4, 4, 4});
    auto r = ft.real();
    r[0] = 1.0;
    r[1] = 2.0;
    r[2] = 3.0;

    ft.forward();
    ft.backward();
    ft.scale(1.0 / static_cast<double>(ft.total()));

    auto r2 = ft.real();
    EXPECT_NEAR(r2[0], 1.0, 1e-12);
    EXPECT_NEAR(r2[1], 2.0, 1e-12);
    EXPECT_NEAR(r2[2], 3.0, 1e-12);
  }

  TEST(FourierTransformTest, MoveSemantics) {
    FourierTransform ft1({4, 4, 4});
    ft1.real()[0] = 42.0;
    FourierTransform ft2 = std::move(ft1);
    EXPECT_NEAR(ft2.real()[0], 42.0, 1e-12);
  }

  TEST(FourierConvolutionTest, DeltaConvolution) {
    FourierConvolution conv({4, 4, 4});
    auto a = conv.input_a();
    auto b = conv.input_b();

    // Set a to a delta: all 1.0 (constant), b to a constant
    for (auto& v : a) v = 1.0;
    for (auto& v : b) v = 2.0;

    conv.execute();
    auto result = conv.result();

    // Convolution of two constants: result should be 2 * total_size
    // Actually: circular convolution of constant f=1 with constant g=2 is 2*N at each point,
    // divided by N from our normalization = 2.0 * N / N = 128.0 / 64 ... no.
    // Discrete circular: (f*g)[n] = sum_k f[k]*g[n-k] / N with our normalization
    // For f=1, g=2: each point = 2*64 / 64 = 2.0. Because scale by 1/N.
    for (auto v : result) {
      EXPECT_NEAR(v, 2.0 * 64.0, 1e-8);
    }
  }

  TEST(FourierTransformTest, InvalidShapeThrows) {
    EXPECT_THROW(FourierTransform({4, 4}), std::invalid_argument);
  }

}  // namespace cdft::numerics
