#include "classicaldft_bits/physics/fmt/convolution.h"

#include <algorithm>

namespace dft_core::physics::fmt {

  ConvolutionField::ConvolutionField(std::vector<long> shape)
      : weight_(shape),
        scratch_(shape),
        field_(static_cast<arma::uword>(weight_.total()), arma::fill::zeros),
        derivative_(static_cast<arma::uword>(weight_.total()), arma::fill::zeros) {}

  void ConvolutionField::set_weight_from_real(const arma::vec& w) {
    if (static_cast<long>(w.n_elem) != weight_.total()) {
      throw std::invalid_argument("ConvolutionField::set_weight_from_real: size mismatch");
    }
    auto real = weight_.real();
    std::copy_n(w.memptr(), w.n_elem, real.data());
    weight_.forward();

    double inv_n = 1.0 / static_cast<double>(weight_.total());
    for (auto& c : weight_.fourier()) {
      c *= inv_n;
    }
  }

  void ConvolutionField::convolve(std::span<const std::complex<double>> rho_fourier) {
    auto wk = weight_.fourier();
    auto out = scratch_.fourier();
    for (long i = 0; i < weight_.fourier_total(); ++i) {
      out[i] = rho_fourier[i] * wk[i];
    }
    scratch_.backward();

    auto real = scratch_.real();
    std::copy_n(real.data(), field_.n_elem, field_.memptr());
  }

  void ConvolutionField::accumulate(
      std::span<std::complex<double>> output_fourier,
      bool conjugate
  ) {
    auto real = scratch_.real();
    std::copy_n(derivative_.memptr(), derivative_.n_elem, real.data());
    scratch_.forward();

    auto wk = weight_.fourier();
    auto dk = scratch_.fourier();
    if (conjugate) {
      for (long i = 0; i < weight_.fourier_total(); ++i) {
        output_fourier[i] += std::conj(wk[i]) * dk[i];
      }
    } else {
      for (long i = 0; i < weight_.fourier_total(); ++i) {
        output_fourier[i] += wk[i] * dk[i];
      }
    }
  }

}  // namespace dft_core::physics::fmt
