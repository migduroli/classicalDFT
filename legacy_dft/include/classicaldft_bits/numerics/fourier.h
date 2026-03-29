#ifndef CLASSICALDFT_NUMERICS_FOURIER_H
#define CLASSICALDFT_NUMERICS_FOURIER_H

#include <complex>
#include <fftw3.h>
#include <span>
#include <stdexcept>
#include <vector>

namespace dft::numerics::fourier {

  /**
   * @brief RAII wrapper around a pair of FFTW3 plans (forward r2c + backward c2r)
   * and their associated FFTW-aligned buffers.
   *
   * Provides zero-copy std::span access to the real-space and Fourier-space data.
   * Move-only (FFTW plans cannot be meaningfully copied).
   */
  class FourierTransform {
   public:
    FourierTransform() = default;

    /**
     * @brief Constructs a FourierTransform for a 3D grid whose dimensions are given by shape
     * @param shape Vector of grid points per axis (must have exactly 3 elements)
     */
    explicit FourierTransform(std::vector<long> shape);

    ~FourierTransform();

    FourierTransform(FourierTransform&& other) noexcept;
    FourierTransform& operator=(FourierTransform&& other) noexcept;

    FourierTransform(const FourierTransform&) = delete;
    FourierTransform& operator=(const FourierTransform&) = delete;

    // region Inspectors:

    [[nodiscard]] const std::vector<long>& shape() const;

    /**
     * Total number of real-space grid points: product of shape elements
     */
    [[nodiscard]] long total() const;

    /**
     * Total number of complex Fourier coefficients: shape[0] * shape[1] * (shape[2]/2 + 1)
     */
    [[nodiscard]] long fourier_total() const;

    // endregion

    // region Accessors:

    std::span<double> real();
    [[nodiscard]] std::span<const double> real() const;

    std::span<std::complex<double>> fourier();
    [[nodiscard]] std::span<const std::complex<double>> fourier() const;

    // endregion

    // region Methods:

    void forward();
    void backward();

    void zeros();
    void scale(double factor);

    // endregion

   private:
    void release() noexcept;

    std::vector<long> shape_ = {};
    double* real_data_ = nullptr;
    fftw_complex* fourier_data_ = nullptr;
    fftw_plan forward_ = nullptr;
    fftw_plan backward_ = nullptr;
  };

  /**
   * @brief 3D cyclic convolution via Fourier transform: c(r) = IFFT[FFT(a) . FFT(b)] / N.
   *
   * Owns three FourierTransform objects. Reusable across multiple convolutions on the
   * same grid by writing into input_a()/input_b() and calling execute().
   */
  class FourierConvolution {
   public:
    FourierConvolution() = default;

    explicit FourierConvolution(std::vector<long> shape);

    std::span<double> input_a();
    std::span<double> input_b();
    [[nodiscard]] std::span<const double> result() const;

    void execute();

    [[nodiscard]] const std::vector<long>& shape() const;
    [[nodiscard]] long total() const;

   private:
    FourierTransform a_;
    FourierTransform b_;
    FourierTransform c_;
  };

}  // namespace dft::numerics::fourier

#endif  // CLASSICALDFT_NUMERICS_FOURIER_H
