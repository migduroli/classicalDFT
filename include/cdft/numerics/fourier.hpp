#pragma once

#include <complex>
#include <fftw3.h>
#include <span>
#include <stdexcept>
#include <vector>

namespace cdft::numerics {

  class FourierTransform {
   public:
    FourierTransform() = default;
    explicit FourierTransform(std::vector<long> shape);
    ~FourierTransform();

    FourierTransform(FourierTransform&& other) noexcept;
    FourierTransform& operator=(FourierTransform&& other) noexcept;

    FourierTransform(const FourierTransform&) = delete;
    FourierTransform& operator=(const FourierTransform&) = delete;

    [[nodiscard]] const std::vector<long>& shape() const;
    [[nodiscard]] long total() const;
    [[nodiscard]] long fourier_total() const;

    std::span<double> real();
    [[nodiscard]] std::span<const double> real() const;

    std::span<std::complex<double>> fourier();
    [[nodiscard]] std::span<const std::complex<double>> fourier() const;

    void forward();
    void backward();

    void zeros();
    void scale(double factor);

   private:
    void release() noexcept;

    std::vector<long> shape_ = {};
    double* real_data_ = nullptr;
    fftw_complex* fourier_data_ = nullptr;
    fftw_plan forward_ = nullptr;
    fftw_plan backward_ = nullptr;
  };

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

}  // namespace cdft::numerics
