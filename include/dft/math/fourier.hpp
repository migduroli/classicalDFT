#ifndef DFT_MATH_FOURIER_HPP
#define DFT_MATH_FOURIER_HPP

#include <armadillo>
#include <complex>
#include <fftw3.h>
#include <memory>
#include <span>
#include <vector>

namespace dft::math {

  // FFTW RAII helpers

  struct FftwFree {
    void operator()(double* p) const noexcept {
      if (p)
        fftw_free(p);
    }

    void operator()(fftw_complex* p) const noexcept {
      if (p)
        fftw_free(p);
    }
  };

  struct FftwPlanDeleter {
    void operator()(fftw_plan_s* p) const noexcept {
      if (p)
        fftw_destroy_plan(p);
    }
  };

  using FftwRealPtr =
      std::unique_ptr<double[], FftwFree>; // NOLINT(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  using FftwComplexPtr =
      std::unique_ptr<fftw_complex[], FftwFree>; // NOLINT(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  using FftwPlanPtr = std::unique_ptr<fftw_plan_s, FftwPlanDeleter>;

  // RAII wrapper around a forward r2c + backward c2r FFTW3 plan pair
  // on a 3D grid. Provides zero-copy std::span access to both buffers.
  // Move-only (FFTW plans cannot be copied).
  class FourierTransform {
   public:
    FourierTransform() = default;
    explicit FourierTransform(std::vector<long> shape);
    ~FourierTransform();

    FourierTransform(FourierTransform&&) noexcept;
    FourierTransform& operator=(FourierTransform&&) noexcept;
    FourierTransform(const FourierTransform&) = delete;
    FourierTransform& operator=(const FourierTransform&) = delete;

    [[nodiscard]] auto shape() const -> const std::vector<long>&;
    [[nodiscard]] auto total() const -> long;
    [[nodiscard]] auto fourier_total() const -> long;

    auto real() -> std::span<double>;
    [[nodiscard]] auto real() const -> std::span<const double>;

    auto fourier() -> std::span<std::complex<double>>;
    [[nodiscard]] auto fourier() const -> std::span<const std::complex<double>>;

    void forward();
    void backward();
    void zeros();
    void scale(double factor);

    // Armadillo bridge: copy an arma::vec into the real buffer.
    void set_real(const arma::vec& v);

    // Armadillo bridge: copy an arma::cx_vec into the Fourier buffer.
    void set_fourier(const arma::cx_vec& v);

    // Armadillo bridge: wrap the real buffer as a non-owning arma::vec.
    [[nodiscard]] auto real_vec() const -> arma::vec;

    // Armadillo bridge: wrap the Fourier buffer as a non-owning arma::cx_vec.
    [[nodiscard]] auto fourier_vec() const -> arma::cx_vec;

   private:
    std::vector<long> shape_;
    FftwRealPtr real_data_;
    FftwComplexPtr fourier_data_;
    FftwPlanPtr forward_;
    FftwPlanPtr backward_;
  };

  // 3D cyclic convolution: c(r) = IFFT[FFT(a) * FFT(b)] / N.
  // Owns three FourierTransform objects. Reusable by writing into
  // input_a()/input_b() and calling execute().
  class FourierConvolution {
   public:
    FourierConvolution() = default;
    explicit FourierConvolution(std::vector<long> shape);

    auto input_a() -> std::span<double>;
    auto input_b() -> std::span<double>;
    [[nodiscard]] auto result() const -> std::span<const double>;

    void execute();

    [[nodiscard]] auto shape() const -> const std::vector<long>&;
    [[nodiscard]] auto total() const -> long;

   private:
    FourierTransform a_;
    FourierTransform b_;
    FourierTransform c_;
  };

} // namespace dft::math

#endif // DFT_MATH_FOURIER_HPP
