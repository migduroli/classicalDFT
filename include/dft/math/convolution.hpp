#ifndef DFT_MATH_CONVOLUTION_HPP
#define DFT_MATH_CONVOLUTION_HPP

#include "dft/math/fourier.hpp"

#include <armadillo>
#include <vector>

namespace dft::math {

  struct ConvolutionWorkspace {
    FourierTransform scratch;

    explicit ConvolutionWorkspace(std::vector<long> shape) : scratch(std::move(shape)) {}
  };

  // Computes a weighted density by Schur product in Fourier space:
  // n(r) = IFFT[rho_k .* weight_k].
  [[nodiscard]] inline auto convolve(
      std::span<const std::complex<double>> weight_k,
      std::span<const std::complex<double>> rho_k,
      ConvolutionWorkspace& workspace
  ) -> arma::vec {
    auto scratch_k = workspace.scratch.fourier();
    for (std::size_t i = 0; i < scratch_k.size(); ++i) {
      scratch_k[i] = weight_k[i] * rho_k[i];
    }

    workspace.scratch.backward();
    return workspace.scratch.real_vec();
  }

  [[nodiscard]] inline auto convolve(
      std::span<const std::complex<double>> weight_k,
      std::span<const std::complex<double>> rho_k,
      const std::vector<long>& shape
  ) -> arma::vec {
    ConvolutionWorkspace workspace(shape);
    return convolve(weight_k, rho_k, workspace);
  }

  // Back-convolution of a functional derivative through a weight:
  // Returns weight_k .* FFT(derivative) in Fourier space as arma::cx_vec.
  // If conjugate is true, uses conj(weight_k) for parity-odd weights.
  [[nodiscard]] inline auto back_convolve(
      std::span<const std::complex<double>> weight_k,
      const arma::vec& derivative,
      ConvolutionWorkspace& workspace,
      bool conjugate = false
  ) -> arma::cx_vec {
    workspace.scratch.set_real(derivative);
    workspace.scratch.forward();

    auto dk = workspace.scratch.fourier();
    arma::cx_vec result(static_cast<arma::uword>(weight_k.size()));
    for (arma::uword i = 0; i < result.n_elem; ++i) {
      auto w = conjugate ? std::conj(weight_k[i]) : weight_k[i];
      result(i) = w * dk[i];
    }
    return result;
  }

  [[nodiscard]] inline auto back_convolve(
      std::span<const std::complex<double>> weight_k,
      const arma::vec& derivative,
      const std::vector<long>& shape,
      bool conjugate = false
  ) -> arma::cx_vec {
    ConvolutionWorkspace workspace(shape);
    return back_convolve(weight_k, derivative, workspace, conjugate);
  }

} // namespace dft::math

#endif // DFT_MATH_CONVOLUTION_HPP
