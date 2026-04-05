#include "dft/math/fourier.hpp"

#include <algorithm>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <thread>

namespace dft::math {

  namespace {
    std::once_flag fftw_threads_flag; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

    void init_fftw_threading() {
#ifdef DFT_FFTW_THREADS
      if (fftw_init_threads()) {
        fftw_make_planner_thread_safe();
        auto hw = std::jthread::hardware_concurrency();
        fftw_plan_with_nthreads(static_cast<int>(hw > 0 ? hw : 1));
      }
#endif
    }
  } // namespace

  FourierTransform::FourierTransform(std::vector<long> shape) : shape_(std::move(shape)) {
    std::call_once(fftw_threads_flag, init_fftw_threading);
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
  FourierTransform::FourierTransform(FourierTransform&&) noexcept = default;
  FourierTransform& FourierTransform::operator=(FourierTransform&&) noexcept = default;

  auto FourierTransform::shape() const -> const std::vector<long>& {
    return shape_;
  }

  auto FourierTransform::total() const -> long {
    return std::accumulate(shape_.begin(), shape_.end(), 1L, std::multiplies<>());
  }

  auto FourierTransform::fourier_total() const -> long {
    return shape_[0] * shape_[1] * (shape_[2] / 2 + 1);
  }

  auto FourierTransform::real() -> std::span<double> {
    return {real_data_.get(), static_cast<std::size_t>(total())};
  }

  auto FourierTransform::real() const -> std::span<const double> {
    return {real_data_.get(), static_cast<std::size_t>(total())};
  }

  auto FourierTransform::fourier() -> std::span<std::complex<double>> {
    return {reinterpret_cast<std::complex<double>*>(fourier_data_.get()), static_cast<std::size_t>(fourier_total())};
  }

  auto FourierTransform::fourier() const -> std::span<const std::complex<double>> {
    return {
        reinterpret_cast<const std::complex<double>*>(fourier_data_.get()),
        static_cast<std::size_t>(fourier_total())
    };
  }

  void FourierTransform::forward() {
    fftw_execute(forward_.get());
  }

  void FourierTransform::backward() {
    fftw_execute(backward_.get());
  }

  void FourierTransform::zeros() {
    auto r = real();
    std::ranges::fill(r, 0.0);
    auto f = fourier();
    std::ranges::fill(f, std::complex<double>{0.0, 0.0});
  }

  void FourierTransform::scale(double factor) {
    auto r = real();
    for (auto& v : r) {
      v *= factor;
    }
  }

  void FourierTransform::set_real(const arma::vec& v) {
    auto r = real();
    auto n = std::min(v.n_elem, static_cast<arma::uword>(r.size()));
    std::copy_n(v.memptr(), n, r.data());
  }

  void FourierTransform::set_fourier(const arma::cx_vec& v) {
    auto f = fourier();
    auto n = std::min(v.n_elem, static_cast<arma::uword>(f.size()));
    std::copy_n(reinterpret_cast<const std::complex<double>*>(v.memptr()), n, f.data());
  }

  auto FourierTransform::real_vec() const -> arma::vec {
    auto r = real();
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast, modernize-return-braced-init-list)
    return arma::vec(const_cast<double*>(r.data()), static_cast<arma::uword>(r.size()), true);
  }

  auto FourierTransform::fourier_vec() const -> arma::cx_vec {
    auto f = fourier();
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast, modernize-return-braced-init-list)
    return arma::cx_vec(const_cast<std::complex<double>*>(f.data()), static_cast<arma::uword>(f.size()), true);
  }

  // FourierConvolution

  FourierConvolution::FourierConvolution(std::vector<long> shape) : a_(shape), b_(shape), c_(std::move(shape)) {}

  auto FourierConvolution::input_a() -> std::span<double> {
    return a_.real();
  }

  auto FourierConvolution::input_b() -> std::span<double> {
    return b_.real();
  }

  auto FourierConvolution::result() const -> std::span<const double> {
    return c_.real();
  }

  void FourierConvolution::execute() {
    a_.forward();
    b_.forward();

    arma::cx_vec fc = a_.fourier_vec() % b_.fourier_vec();
    c_.set_fourier(fc);

    c_.backward();
    c_.scale(1.0 / static_cast<double>(c_.total()));
  }

  auto FourierConvolution::shape() const -> const std::vector<long>& {
    return a_.shape();
  }

  auto FourierConvolution::total() const -> long {
    return a_.total();
  }

} // namespace dft::math
