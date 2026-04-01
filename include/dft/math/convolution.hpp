#ifndef DFT_MATH_CONVOLUTION_HPP
#define DFT_MATH_CONVOLUTION_HPP

#include "dft/math/fourier.hpp"

#include <armadillo>
#include <complex>
#include <span>
#include <vector>

namespace dft::math {

  // Computes a weighted density by Schur product in Fourier space:
  // n(r) = IFFT[rho_k * weight_k].
  [[nodiscard]] inline auto convolve(
      std::span<const std::complex<double>> weight_k, std::span<const std::complex<double>> rho_k,
      const std::vector<long>& shape
  ) -> arma::vec {
    FourierTransform scratch(shape);
    auto out = scratch.fourier();
    for (std::size_t i = 0; i < out.size(); ++i) {
      out[i] = rho_k[i] * weight_k[i];
    }
    scratch.backward();

    auto real = scratch.real();
    return arma::vec(real.data(), static_cast<arma::uword>(real.size()));
  }

  // Back-convolution of a functional derivative through a weight:
  // Returns weight_k * FFT(derivative) in Fourier space.
  // If conjugate is true, uses conj(weight_k) for parity-odd weights.
  [[nodiscard]] inline auto back_convolve(
      std::span<const std::complex<double>> weight_k, const arma::vec& derivative,
      const std::vector<long>& shape, bool conjugate = false
  ) -> std::vector<std::complex<double>> {
    FourierTransform scratch(shape);
    auto real = scratch.real();
    std::copy_n(derivative.memptr(), derivative.n_elem, real.data());
    scratch.forward();

    auto dk = scratch.fourier();
    std::vector<std::complex<double>> result(dk.size());
    if (conjugate) {
      for (std::size_t i = 0; i < dk.size(); ++i) {
        result[i] = std::conj(weight_k[i]) * dk[i];
      }
    } else {
      for (std::size_t i = 0; i < dk.size(); ++i) {
        result[i] = weight_k[i] * dk[i];
      }
    }
    return result;
  }

}  // namespace dft::math

#endif  // DFT_MATH_CONVOLUTION_HPP
