#include "classicaldft_bits/numerics/fourier.h"

#include <algorithm>
#include <cstring>
#include <numeric>

namespace dft::numerics::fourier {

  // ── FourierTransform ─────────────────────────────────────────────────

  FourierTransform::FourierTransform(std::vector<long> shape) : shape_(std::move(shape)) {
    if (shape_.size() != 3) {
      throw std::invalid_argument("FourierTransform: shape must have exactly 3 elements");
    }
    if (total() == 0) {
      throw std::invalid_argument("FourierTransform: grid dimensions must be nonzero");
    }

    real_data_ = static_cast<double*>(fftw_malloc(sizeof(double) * total()));
    fourier_data_ = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * fourier_total()));

    if (real_data_ == nullptr || fourier_data_ == nullptr) {
      release();
      throw std::runtime_error("FourierTransform: fftw_malloc failed");
    }

    forward_ = fftw_plan_dft_r2c_3d(
        static_cast<int>(shape_[0]),
        static_cast<int>(shape_[1]),
        static_cast<int>(shape_[2]),
        real_data_,
        fourier_data_,
        FFTW_ESTIMATE
    );

    backward_ = fftw_plan_dft_c2r_3d(
        static_cast<int>(shape_[0]),
        static_cast<int>(shape_[1]),
        static_cast<int>(shape_[2]),
        fourier_data_,
        real_data_,
        FFTW_ESTIMATE
    );

    zeros();
  }

  FourierTransform::~FourierTransform() {
    release();
  }

  FourierTransform::FourierTransform(FourierTransform&& other) noexcept
      : shape_(std::move(other.shape_)),
        real_data_(other.real_data_),
        fourier_data_(other.fourier_data_),
        forward_(other.forward_),
        backward_(other.backward_) {
    other.real_data_ = nullptr;
    other.fourier_data_ = nullptr;
    other.forward_ = nullptr;
    other.backward_ = nullptr;
  }

  FourierTransform& FourierTransform::operator=(FourierTransform&& other) noexcept {
    if (this != &other) {
      release();
      shape_ = std::move(other.shape_);
      real_data_ = other.real_data_;
      fourier_data_ = other.fourier_data_;
      forward_ = other.forward_;
      backward_ = other.backward_;
      other.real_data_ = nullptr;
      other.fourier_data_ = nullptr;
      other.forward_ = nullptr;
      other.backward_ = nullptr;
    }
    return *this;
  }

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
    return {real_data_, static_cast<std::size_t>(total())};
  }

  std::span<const double> FourierTransform::real() const {
    return {real_data_, static_cast<std::size_t>(total())};
  }

  std::span<std::complex<double>> FourierTransform::fourier() {
    return {reinterpret_cast<std::complex<double>*>(fourier_data_), static_cast<std::size_t>(fourier_total())};
  }

  std::span<const std::complex<double>> FourierTransform::fourier() const {
    return {reinterpret_cast<const std::complex<double>*>(fourier_data_), static_cast<std::size_t>(fourier_total())};
  }

  void FourierTransform::forward() {
    fftw_execute(forward_);
  }

  void FourierTransform::backward() {
    fftw_execute(backward_);
  }

  void FourierTransform::zeros() {
    std::memset(real_data_, 0, sizeof(double) * total());
    std::memset(fourier_data_, 0, sizeof(fftw_complex) * fourier_total());
  }

  void FourierTransform::scale(double factor) {
    auto r = real();
    for (auto& v : r) {
      v *= factor;
    }
  }

  void FourierTransform::release() noexcept {
    if (forward_ != nullptr) {
      fftw_destroy_plan(forward_);
    }
    if (backward_ != nullptr) {
      fftw_destroy_plan(backward_);
    }
    if (real_data_ != nullptr) {
      fftw_free(real_data_);
    }
    if (fourier_data_ != nullptr) {
      fftw_free(fourier_data_);
    }
    forward_ = nullptr;
    backward_ = nullptr;
    real_data_ = nullptr;
    fourier_data_ = nullptr;
    shape_.clear();
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

}  // namespace dft::numerics::fourier
