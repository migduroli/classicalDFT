#ifndef DFT_MATH_CONVOLUTION_HPP
#define DFT_MATH_CONVOLUTION_HPP

#include "dft/math/fourier.hpp"

#include <armadillo>
#include <complex>
#include <span>

namespace dft::math {

  // Computes a weighted density by Schur product in Fourier space:
  // n(r) = IFFT[rho_k * weight_k]. The scratch transform is used as
  // workspace. Returns the real-space result as an arma::vec.
  [[nodiscard]] inline auto convolve(
      std::span<const std::complex<double>> weight_k, std::span<const std::complex<double>> rho_k,
      FourierTransform& scratch
  ) -> arma::vec {
    auto out = scratch.fourier();
    for (std::size_t i = 0; i < out.size(); ++i) {
      out[i] = rho_k[i] * weight_k[i];
    }
    scratch.backward();

    auto real = scratch.real();
    return arma::vec(real.data(), static_cast<arma::uword>(real.size()));
  }

  // Accumulates the back-convolution of a functional derivative into a
  // Fourier-space force buffer: force_k += weight_k * FFT(derivative).
  // If conjugate is true, uses conj(weight_k) for parity-odd weights.
  inline void accumulate(
      std::span<const std::complex<double>> weight_k, const arma::vec& derivative, FourierTransform& scratch,
      std::span<std::complex<double>> force_k, bool conjugate = false
  ) {
    auto real = scratch.real();
    std::copy_n(derivative.memptr(), derivative.n_elem, real.data());
    scratch.forward();

    auto dk = scratch.fourier();
    if (conjugate) {
      for (std::size_t i = 0; i < force_k.size(); ++i) {
        force_k[i] += std::conj(weight_k[i]) * dk[i];
      }
    } else {
      for (std::size_t i = 0; i < force_k.size(); ++i) {
        force_k[i] += weight_k[i] * dk[i];
      }
    }
  }

}  // namespace dft::math

#endif  // DFT_MATH_CONVOLUTION_HPP
