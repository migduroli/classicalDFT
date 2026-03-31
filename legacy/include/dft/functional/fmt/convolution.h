#ifndef DFT_FUNCTIONAL_FMT_CONVOLUTION_H
#define DFT_FUNCTIONAL_FMT_CONVOLUTION_H

#include "dft/math/fourier.h"

#include <armadillo>
#include <complex>
#include <span>
#include <vector>

namespace dft::functional::fmt {

  /**
   * @brief A single convolution channel in the FMT pipeline.
   *
   * Bundles three quantities for one component $\alpha$:
   *  - **weight**: the pre-FFT'd weight function $\hat{w}_\alpha(\mathbf{k})$
   *  - **field**: the real-space weighted density $n_\alpha(\mathbf{r})$
   *  - **derivative**: the functional derivative $\partial\Phi/\partial n_\alpha(\mathbf{r})$
   *
   * Forward path (density to weighted density):
   *   `convolve(FFT(\rho))` sets `field = IFFT[\hat\rho \cdot \hat{w}_\alpha]`.
   *
   * Backward path (derivative to force contribution):
   *   `accumulate(output)` adds $\hat{w}_\alpha \cdot \text{FFT}(d\Phi/dn_\alpha)$ to output.
   */
  class ConvolutionField {
   public:
    ConvolutionField() = default;
    explicit ConvolutionField(const std::vector<long>& shape);
    ~ConvolutionField() = default;

    ConvolutionField(ConvolutionField&&) noexcept = default;             // NOLINT(bugprone-exception-escape)
    ConvolutionField& operator=(ConvolutionField&&) noexcept = default;  // NOLINT(bugprone-exception-escape)

    ConvolutionField(const ConvolutionField&) = delete;
    ConvolutionField& operator=(const ConvolutionField&) = delete;

    // ── Weight setup ────────────────────────────────────────────────────────

    /**
     * @brief Copies a real-space weight into the FFT buffer, transforms it,
     * and normalises by $1/N$.
     */
    void set_weight_from_real(const arma::vec& w);

    // ── Forward path ────────────────────────────────────────────────────────

    /**
     * @brief Computes the weighted density by Schur product in Fourier space.
     *
     * Given $\hat\rho(\mathbf{k})$, computes
     * $n_\alpha(\mathbf{r}) = \text{IFFT}[\hat\rho \cdot \hat{w}_\alpha]$.
     */
    void convolve(std::span<const std::complex<double>> rho_fourier);

    // ── Backward path ───────────────────────────────────────────────────────

    /**
     * @brief Accumulates $\hat{w}_\alpha \cdot \text{FFT}(d\Phi/dn_\alpha)$
     * into the given output Fourier buffer.
     *
     * @param conjugate If true, uses $\hat{w}_\alpha^*$ (for parity-odd weights).
     */
    void accumulate(std::span<std::complex<double>> output_fourier, bool conjugate = false);

    // ── Accessors ───────────────────────────────────────────────────────────

    [[nodiscard]] const arma::vec& field() const noexcept { return field_; }
    [[nodiscard]] arma::vec& field() noexcept { return field_; }

    [[nodiscard]] const arma::vec& derivative() const noexcept { return derivative_; }
    [[nodiscard]] arma::vec& derivative() noexcept { return derivative_; }

    [[nodiscard]] const math::fourier::FourierTransform& weight() const noexcept { return weight_; }

    /**
     * @brief Mutable access to the weight's Fourier coefficients.
     *
     * Used by Weights::generate() to populate the weights directly in k-space.
     */
    [[nodiscard]] std::span<std::complex<double>> weight_fourier() { return weight_.fourier(); }

    [[nodiscard]] long total() const noexcept { return weight_.total(); }

   private:
    math::fourier::FourierTransform weight_;
    math::fourier::FourierTransform scratch_;
    arma::vec field_;
    arma::vec derivative_;
  };

}  // namespace dft::functional::fmt

#endif  // DFT_FUNCTIONAL_FMT_CONVOLUTION_H
