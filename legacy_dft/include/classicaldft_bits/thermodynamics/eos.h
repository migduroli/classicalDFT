#ifndef CLASSICALDFT_THERMODYNAMICS_EOS_H
#define CLASSICALDFT_THERMODYNAMICS_EOS_H

#include "classicaldft_bits/thermodynamics/enskog.h"

#include <array>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <string>

namespace dft::thermodynamics::eos {

  /**
   * @brief Abstract equation of state for a single-component fluid.
   *
   * All methods are functions of number density $\rho$ (in units where $d = 1$)
   * and return dimensionless quantities (in units of $k_BT$).
   *
   * Subclasses implement the excess free energy per particle $\phi^{\text{ex}}(\rho)$
   * and its first three density derivatives. The base class provides derived
   * thermodynamic quantities from these.
   */
  class EquationOfState {
   public:
    explicit EquationOfState(double kT) : kT_(kT) {
      if (kT <= 0.0) {
        throw std::invalid_argument("Temperature must be positive");
      }
    }

    virtual ~EquationOfState() = default;

    EquationOfState(const EquationOfState&) = default;
    EquationOfState& operator=(const EquationOfState&) = default;
    EquationOfState(EquationOfState&&) = default;
    EquationOfState& operator=(EquationOfState&&) = default;

    // ── Abstract interface: excess free energy per particle and derivatives ──

    [[nodiscard]] virtual double excess_free_energy_per_particle(double density) const = 0;
    [[nodiscard]] virtual double d_excess_free_energy_per_particle(double density) const = 0;
    [[nodiscard]] virtual double d2_excess_free_energy_per_particle(double density) const = 0;
    [[nodiscard]] virtual double d3_excess_free_energy_per_particle(double density) const = 0;

    [[nodiscard]] virtual std::string name() const = 0;

    // ── Derived quantities ──────────────────────────────────────────────────

    /**
     * @brief Total free energy per particle $f(\rho) / k_BT = \ln\rho - 1 + \phi^{\text{ex}}(\rho)$.
     */
    [[nodiscard]] double free_energy_per_particle(double density) const {
      return std::log(density) - 1.0 + excess_free_energy_per_particle(density);
    }

    /**
     * @brief Excess free energy per unit volume $f^{\text{ex}}(\rho) / (k_BT\,V) = \rho\,\phi^{\text{ex}}(\rho)$.
     */
    [[nodiscard]] double excess_free_energy_density(double density) const {
      return density * excess_free_energy_per_particle(density);
    }

    /**
     * @brief First derivative of excess free energy density.
     *
     * $\frac{d}{d\rho}[\rho\,\phi^{\text{ex}}] = \phi^{\text{ex}} + \rho\,\phi^{\text{ex}\prime}$
     */
    [[nodiscard]] double d_excess_free_energy_density(double density) const {
      return excess_free_energy_per_particle(density) + density * d_excess_free_energy_per_particle(density);
    }

    /**
     * @brief Second derivative of excess free energy density.
     *
     * $\frac{d^2}{d\rho^2}[\rho\,\phi^{\text{ex}}] = 2\phi^{\text{ex}\prime} + \rho\,\phi^{\text{ex}\prime\prime}$
     */
    [[nodiscard]] double d2_excess_free_energy_density(double density) const {
      return 2.0 * d_excess_free_energy_per_particle(density) + density * d2_excess_free_energy_per_particle(density);
    }

    /**
     * @brief Pressure divided by $n\,k_BT$: $P/(\rho\,k_BT) = 1 + \rho\,\phi^{\text{ex}\prime}$.
     */
    [[nodiscard]] double pressure(double density) const {
      return 1.0 + density * d_excess_free_energy_per_particle(density);
    }

    [[nodiscard]] double temperature() const noexcept { return kT_; }

   protected:
    double kT_;
  };

  // ── Null EOS (ideal gas) ──────────────────────────────────────────────────

  class IdealGas final : public EquationOfState {
   public:
    explicit IdealGas(double kT) : EquationOfState(kT) {}

    [[nodiscard]] double excess_free_energy_per_particle(double /*density*/) const override { return 0.0; }
    [[nodiscard]] double d_excess_free_energy_per_particle(double /*density*/) const override { return 0.0; }
    [[nodiscard]] double d2_excess_free_energy_per_particle(double /*density*/) const override { return 0.0; }
    [[nodiscard]] double d3_excess_free_energy_per_particle(double /*density*/) const override { return 0.0; }
    [[nodiscard]] std::string name() const override { return "IdealGas"; }
  };

  // ── Percus-Yevick (compressibility route) ─────────────────────────────────

  class PercusYevick final : public EquationOfState {
   public:
    explicit PercusYevick(double kT) : EquationOfState(kT) {}

    [[nodiscard]] double excess_free_energy_per_particle(double density) const override {
      return hs_.excess_free_energy(packing_fraction(density));
    }

    [[nodiscard]] double d_excess_free_energy_per_particle(double density) const override {
      constexpr double PI6 = std::numbers::pi / 6.0;
      return PI6 * hs_.d_excess_free_energy(packing_fraction(density));
    }

    [[nodiscard]] double d2_excess_free_energy_per_particle(double density) const override {
      constexpr double PI6 = std::numbers::pi / 6.0;
      return PI6 * PI6 * hs_.d2_excess_free_energy(packing_fraction(density));
    }

    [[nodiscard]] double d3_excess_free_energy_per_particle(double density) const override {
      constexpr double PI6 = std::numbers::pi / 6.0;
      return PI6 * PI6 * PI6 * hs_.d3_excess_free_energy(packing_fraction(density));
    }

    [[nodiscard]] std::string name() const override { return "PercusYevick"; }

   private:
    dft::thermodynamics::PercusYevick hs_{
        dft::thermodynamics::PercusYevick::Route::Compressibility};
  };

  // ── Lennard-Jones: Johnson-Zollweg-Gubbins (JZG) ─────────────────────────

  /**
   * @brief Modified Benedict-Webb-Rubin EOS for the Lennard-Jones fluid.
   *
   * 32-parameter fit from Johnson, Zollweg, and Gubbins, Mol. Phys. 78, 591 (1993).
   * Optional long-range correction for cutoff at $r_c$ via the standard LJ tail integral.
   */
  class LennardJonesJZG final : public EquationOfState {
   public:
    /**
     * @param kT Reduced temperature $k_BT / \varepsilon$.
     * @param cutoff_radius Cutoff radius $r_c / \sigma$. Negative means no cutoff correction.
     * @param shifted If true, applies a shifted (not truncated-and-shifted) tail correction.
     */
    LennardJonesJZG(double kT, double cutoff_radius = -1.0, bool shifted = false);

    [[nodiscard]] double excess_free_energy_per_particle(double density) const override;
    [[nodiscard]] double d_excess_free_energy_per_particle(double density) const override;
    [[nodiscard]] double d2_excess_free_energy_per_particle(double density) const override;
    [[nodiscard]] double d3_excess_free_energy_per_particle(double density) const override;
    [[nodiscard]] std::string name() const override { return "LennardJonesJZG"; }

   private:
    [[nodiscard]] double a_coeff(int i) const;
    [[nodiscard]] double b_coeff(int i) const;
    [[nodiscard]] double G_integral(double rho, int i) const;

    static constexpr double GAMMA = 3.0;
    static constexpr std::array<double, 32> X = {
        0.8623085097507421,   2.976218765822098,    -8.402230115796038,   0.1054136629203555,   -0.8564583828174598,
        1.582759470107601,    0.7639421948305453,   1.753173414312048,    2.798291772190376e3,  -4.8394220260857657e-2,
        0.9963265197721935,   -3.698000291272493e1, 2.084012299434647e1,  8.305402124717285e1,  -9.574799715203068e2,
        -1.477746229234994e2, 6.398607852471505e1,  1.603993673294834e1,  6.805916615864377e1,  -2.791293578795945e3,
        -6.245128304568454,   -8.116836104958410e3, 1.488735559561229e1,  -1.059346754655084e4, -1.131607632802822e2,
        -8.867771540418822e3, -3.986982844450543e1, -4.689270299917261e3, 2.593535277438717e2,  -2.694523589434903e3,
        -7.218487631550215e2, 1.721802063863269e2,
    };
    double tail_correction_ = 0.0;
  };

  // ── Lennard-Jones: Mecke et al. ───────────────────────────────────────────

  /**
   * @brief EOS for the Lennard-Jones fluid from Mecke et al., Int. J. Thermophys. 17, 391 (1996).
   *
   * 32-term parametrisation with critical point $\rho_c = 0.3107$, $T_c = 1.328$.
   */
  class LennardJonesMecke final : public EquationOfState {
   public:
    LennardJonesMecke(double kT, double cutoff_radius = -1.0, bool shifted = false);

    [[nodiscard]] double excess_free_energy_per_particle(double density) const override;
    [[nodiscard]] double d_excess_free_energy_per_particle(double density) const override;
    [[nodiscard]] double d2_excess_free_energy_per_particle(double density) const override;
    [[nodiscard]] double d3_excess_free_energy_per_particle(double density) const override;
    [[nodiscard]] std::string name() const override { return "LennardJonesMecke"; }

   private:
    static constexpr double RHO_C = 0.3107;
    static constexpr double KT_C = 1.328;

    struct Term {
      double c;
      double m;
      int n;
      int p;
      int q;
    };

    static constexpr std::array<Term, 32> TERMS = {{
        {0.33619760720e-05, -2, 9, 0, 0},    {-0.14707220591e+01, -1, 1, 0, 0},   {-0.11972121043e+00, -1, 2, 0, 0},
        {-0.11350363539e-04, -1, 9, 0, 0},   {-0.26778688896e-04, -0.5, 8, 0, 0}, {0.12755936511e-05, -0.5, 10, 0, 0},
        {0.40088615477e-02, 0.5, 1, 0, 0},   {0.52305580273e-05, 0.5, 7, 0, 0},   {-0.10214454556e-07, 1, 10, 0, 0},
        {-0.14526799362e-01, -5, 1, -1, 1},  {0.64975356409e-01, -4, 1, -1, 1},   {-0.60304755494e-01, -2, 1, -1, 1},
        {-0.14925537332e+00, -2, 2, -1, 1},  {-0.31664355868e-03, -2, 8, -1, 1},  {0.28312781935e-01, -1, 1, -1, 1},
        {0.13039603845e-03, -1, 10, -1, 1},  {0.10121435381e-01, 0, 4, -1, 1},    {-0.15425936014e-04, 0, 9, -1, 1},
        {-0.61568007279e-01, -5, 2, -1, 2},  {0.76001994423e-02, -4, 5, -1, 2},   {-0.18906040708e+00, -3, 1, -1, 2},
        {0.33141311846e+00, -2, 2, -1, 2},   {-0.25229604842e+00, -2, 3, -1, 2},  {0.13145401812e+00, -2, 4, -1, 2},
        {-0.48672350917e-01, -1, 2, -1, 2},  {0.14756043863e-02, -10, 3, -1, 3},  {-0.85996667747e-02, -6, 4, -1, 3},
        {0.33880247915e-01, -4, 2, -1, 3},   {0.69427495094e-02, 0, 2, -1, 3},    {-0.22271531045e-07, -24, 5, -1, 4},
        {-0.22656880018e-03, -10, 2, -1, 4}, {0.24056013779e-02, -2, 10, -1, 4},
    }};

    double tail_correction_ = 0.0;
  };

}  // namespace dft::thermodynamics::eos

#endif  // CLASSICALDFT_THERMODYNAMICS_EOS_H
