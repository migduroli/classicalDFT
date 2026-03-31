#include "dft/math/fourier.h"

#include <algorithm>
#include <cstring>
#include <numeric>

namespace dft::math::fourier {

  // ── FourierTransform ─────────────────────────────────────────────────

  FourierTransform::FourierTransform(std::vector<long> shape) : shape_(std::move(shape)) {
    if (shape_.size() != 3) {
      throw std::invalid_argument("FourierTransform: shape must have exactly 3 elements");
    }
    if (total() == 0) {
      throw std::invalid_argument("FourierTransform: grid dimensions must be nonzero");
    }

    real_data_.reset(static_cast<double*>(fftw_malloc(sizeof(double) * total())));
    fourier_data_.reset(static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * fourier_total())));

    if (!real_data_ || !fourier_data_) {
      throw std::runtime_error("FourierTransform: fftw_malloc failed");
    }

    forward_.reset(fftw_plan_dft_r2c_3d(
        static_cast<int>(shape_[0]),
        static_cast<int>(shape_[1]),
        static_cast<int>(shape_[2]),
        real_data_.get(),
        fourier_data_.get(),
        FFTW_ESTIMATE
    ));

    backward_.reset(fftw_plan_dft_c2r_3d(
        static_cast<int>(shape_[0]),
        static_cast<int>(shape_[1]),
        static_cast<int>(shape_[2]),
        fourier_data_.get(),
        real_data_.get(),
        FFTW_ESTIMATE
    ));

    zeros();
  }

  FourierTransform::~FourierTransform() = default;

  FourierTransform::FourierTransform(FourierTransform&& other) noexcept = default;

  FourierTransform& FourierTransform::operator=(FourierTransform&& other) noexcept = default;

  const std::vector<long>& FourierTransform::shape() const {
    return shape_;
  }

  long FourierTransform::total() const {
    return std::accumulate(begin(shape_), end(shape_), 1L, std::multiplies<>());
  }

  long FourierTransform::fourier_total() const {
    return shape_[0] * shape_[1] * (shape_[2] / 2 + 1);
  }

  std::span<double> FourierTransform::real() {
    return {real_data_.get(), static_cast<std::size_t>(total())};
  }

  std::span<const double> FourierTransform::real() const {
    return {real_data_.get(), static_cast<std::size_t>(total())};
  }

  std::span<std::complex<double>> FourierTransform::fourier() {
    return {reinterpret_cast<std::complex<double>*>(fourier_data_.get()), static_cast<std::size_t>(fourier_total())};
  }

  std::span<const std::complex<double>> FourierTransform::fourier() const {
    return {
        reinterpret_cast<const std::complex<double>*>(fourier_data_.get()), static_cast<std::size_t>(fourier_total())};
  }

  void FourierTransform::forward() {
    fftw_execute(forward_.get());
  }

  void FourierTransform::backward() {
    fftw_execute(backward_.get());
  }

  void FourierTransform::zeros() {
    std::memset(real_data_.get(), 0, sizeof(double) * total());
    std::memset(fourier_data_.get(), 0, sizeof(fftw_complex) * fourier_total());
  }

  void FourierTransform::scale(double factor) {
    auto r = real();
    for (auto& v : r) {
      v *= factor;
    }
  }

  // ── FourierConvolution ────────────────────────────────────────────────────

  FourierConvolution::FourierConvolution(std::vector<long> shape) : a_(shape), b_(shape), c_(std::move(shape)) {}

  std::span<double> FourierConvolution::input_a() {
    return a_.real();
  }

  std::span<double> FourierConvolution::input_b() {
    return b_.real();
  }

  std::span<const double> FourierConvolution::result() const {
    return c_.real();
  }

  void FourierConvolution::execute() {
    a_.forward();
    b_.forward();

    auto fa = a_.fourier();
    auto fb = b_.fourier();
    auto fc = c_.fourier();
    for (std::size_t i = 0; i < fc.size(); ++i) {
      fc[i] = fa[i] * fb[i];
    }

    c_.backward();
    c_.scale(1.0 / static_cast<double>(c_.total()));
  }

  const std::vector<long>& FourierConvolution::shape() const {
    return a_.shape();
  }

  long FourierConvolution::total() const {
    return a_.total();
  }

}  // namespace dft::math::fourier
