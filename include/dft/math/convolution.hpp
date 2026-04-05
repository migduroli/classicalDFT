#ifndef DFT_MATH_CONVOLUTION_HPP
#define DFT_MATH_CONVOLUTION_HPP

#include "dft/math/fourier.hpp"

#include <armadillo>
#include <vector>

namespace dft::math {

  // Computes a weighted density by Schur product in Fourier space:
  // n(r) = IFFT[rho_k .* weight_k].
  [[nodiscard]] inline auto convolve(
      std::span<const std::complex<double>> weight_k,
      std::span<const std::complex<double>> rho_k,
      const std::vector<long>& shape
  ) -> arma::vec {
    auto n = static_cast<arma::uword>(weight_k.size());
    arma::cx_vec w(const_cast<std::complex<double>*>(weight_k.data()), n, false, true);
    arma::cx_vec r(const_cast<std::complex<double>*>(rho_k.data()), n, false, true);

    FourierTransform scratch(shape);
    scratch.set_fourier(w % r);
    scratch.backward();
    return scratch.real_vec();
  }

  // Back-convolution of a functional derivative through a weight:
  // Returns weight_k .* FFT(derivative) in Fourier space as arma::cx_vec.
  // If conjugate is true, uses conj(weight_k) for parity-odd weights.
  [[nodiscard]] inline auto back_convolve(
      std::span<const std::complex<double>> weight_k,
      const arma::vec& derivative,
      const std::vector<long>& shape,
      bool conjugate = false
  ) -> arma::cx_vec {
    FourierTransform scratch(shape);
    scratch.set_real(derivative);
    scratch.forward();

    auto n = static_cast<arma::uword>(weight_k.size());
    arma::cx_vec w(const_cast<std::complex<double>*>(weight_k.data()), n, false, true);
    arma::cx_vec dk = scratch.fourier_vec();

    if (conjugate) {
      return arma::cx_vec(arma::conj(w) % dk);
    }
    return arma::cx_vec(w % dk);
  }

}  // namespace dft::math

#endif  // DFT_MATH_CONVOLUTION_HPP
