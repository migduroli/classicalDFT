#ifndef DFT_FUNCTIONALS_FMT_MODELS_HPP
#define DFT_FUNCTIONALS_FMT_MODELS_HPP

#include "dft/functionals/fmt/measures.hpp"
#include "dft/math/autodiff.hpp"

#include <cmath>
#include <numbers>
#include <string_view>
#include <variant>

namespace dft::functionals::fmt {

  // Tensor-based phi3 and its partial derivatives, shared by White Bear models.

  namespace detail {

    // Tensor-based Phi3 for WhiteBear I/II = esFMT(A=1, B=-1).
    // Phi3 = (1/(24pi)) * [(n2^3 - 3n2*v1v1 + 3qf - T3) - (n2^3 - 3n2*T2 + 2T3)]
    //       = (1/(24pi)) * [3n2*T2 - 3n2*v1v1 + 3qf - 3T3]

    [[nodiscard]] inline auto tensor_phi3(const Measures& m) -> double {
      constexpr double INV_24PI = 1.0 / (24.0 * std::numbers::pi);
      return INV_24PI * 3.0 *
          (m.n2 * m.products.trace_T2 - m.n2 * m.products.dot_v1_v1 +
           m.products.quadratic_form - m.products.trace_T3);
    }

    [[nodiscard]] inline auto tensor_phi3_d_n2(const Measures& m) -> double {
      constexpr double INV_24PI = 1.0 / (24.0 * std::numbers::pi);
      return INV_24PI * 3.0 * (m.products.trace_T2 - m.products.dot_v1_v1);
    }

    [[nodiscard]] inline auto tensor_phi3_d_v1(const Measures& m) -> arma::rowvec3 {
      constexpr double INV_24PI = 1.0 / (24.0 * std::numbers::pi);
      return INV_24PI * (-6.0 * m.n2 * m.v1 + 3.0 * m.v1 * m.T + 3.0 * m.v1 * m.T.t());
    }

    [[nodiscard]] inline auto tensor_phi3_d_T(int i, int j, const Measures& m) -> double {
      constexpr double INV_8PI = 1.0 / (8.0 * std::numbers::pi);
      double TT_ji = 0.0;
      for (int k = 0; k < 3; ++k) TT_ji += m.T(j, k) * m.T(k, i);
      return INV_8PI * (m.v1(i) * m.v1(j) + 2.0 * m.n2 * m.T(j, i) - 3.0 * TT_ji);
    }

    // Common phi/d_phi/excess_chemical_potential for any FMT model type.

    template <typename Model>
    [[nodiscard]] auto compute_phi(const Model& func, const Measures& m) -> double {
      double e = m.eta;
      return -m.n0 * static_cast<double>(func.f1(e)) +
          (m.n1 * m.n2 - m.products.dot_v0_v1) * static_cast<double>(func.f2(e)) +
          func.phi3(m) * static_cast<double>(func.f3(e));
    }

    template <typename Model>
    [[nodiscard]] auto compute_d_phi(const Model& func, const Measures& m) -> MeasureDerivatives {
      double e = m.eta;
      auto [f1_val, df1_val] =
          math::derivatives_up_to_1([&func](math::dual x) -> math::dual { return func.f1(x); }, e);
      auto [f2_val, df2_val] =
          math::derivatives_up_to_1([&func](math::dual x) -> math::dual { return func.f2(x); }, e);
      auto [f3_val, df3_val] =
          math::derivatives_up_to_1([&func](math::dual x) -> math::dual { return func.f3(x); }, e);
      double p3 = func.phi3(m);

      MeasureDerivatives dm;
      dm.d_eta = -m.n0 * df1_val + (m.n1 * m.n2 - m.products.dot_v0_v1) * df2_val + p3 * df3_val;
      dm.d_n0 = -f1_val;
      dm.d_n1 = m.n2 * f2_val;
      dm.d_n2 = m.n1 * f2_val + func.d_phi3_d_n2(m) * f3_val;
      dm.d_v0 = -m.v1 * f2_val;
      dm.d_v1 = -m.v0 * f2_val + func.d_phi3_d_v1(m) * f3_val;

      if constexpr (Model::NEEDS_TENSOR) {
        for (int i = 0; i < 3; ++i)
          for (int j = 0; j < 3; ++j)
            dm.d_T(i, j) = func.d_phi3_d_T(i, j, m) * f3_val;
      } else {
        dm.d_T.zeros();
      }

      return dm;
    }

    template <typename Model>
    [[nodiscard]] auto compute_excess_chemical_potential(
        const Model& func, double density, double diameter
    ) -> double {
      auto m = make_uniform_measures(density, diameter);
      auto dm = compute_d_phi(func, m);
      double d = diameter;
      double r = 0.5 * d;

      double dn3_drho = (std::numbers::pi / 6.0) * d * d * d;
      double dn2_drho = std::numbers::pi * d * d;
      double dn1_drho = r;
      double dn0_drho = 1.0;

      double mu_ex =
          dm.d_eta * dn3_drho + dm.d_n2 * dn2_drho + dm.d_n1 * dn1_drho + dm.d_n0 * dn0_drho;

      if constexpr (Model::NEEDS_TENSOR) {
        double dt_drho = std::numbers::pi * d * d / 3.0;
        for (int j = 0; j < 3; ++j) {
          mu_ex += dm.d_T(j, j) * dt_drho;
        }
      }

      return mu_ex;
    }

  }  // namespace detail

  // CRTP base providing shared computed methods for all FMT model types.
  // Derived must provide: f1(T), f2(T), f3(T) (static templates),
  // phi3(Measures), d_phi3_d_n2(Measures), d_phi3_d_v1(Measures),
  // and optionally d_phi3_d_T(int, int, Measures) for tensor models.
  // Also: static constexpr bool NEEDS_TENSOR, std::string_view NAME.

  template <typename Derived>
  struct FMTModelBase {
    [[nodiscard]] auto d_f1(double eta) const -> double {
      const auto& self = static_cast<const Derived&>(*this);
      auto [f, df] = math::derivatives_up_to_1(
          [&self](math::dual x) -> math::dual { return self.f1(x); }, eta);
      return df;
    }

    [[nodiscard]] auto d_f2(double eta) const -> double {
      const auto& self = static_cast<const Derived&>(*this);
      auto [f, df] = math::derivatives_up_to_1(
          [&self](math::dual x) -> math::dual { return self.f2(x); }, eta);
      return df;
    }

    [[nodiscard]] auto d_f3(double eta) const -> double {
      const auto& self = static_cast<const Derived&>(*this);
      auto [f, df] = math::derivatives_up_to_1(
          [&self](math::dual x) -> math::dual { return self.f3(x); }, eta);
      return df;
    }

    [[nodiscard]] auto phi(const Measures& m) const -> double {
      return detail::compute_phi(static_cast<const Derived&>(*this), m);
    }

    [[nodiscard]] auto d_phi(const Measures& m) const -> MeasureDerivatives {
      return detail::compute_d_phi(static_cast<const Derived&>(*this), m);
    }

    [[nodiscard]] auto free_energy_density(double density, double diameter) const -> double {
      return phi(make_uniform_measures(density, diameter));
    }

    [[nodiscard]] auto excess_chemical_potential(double density, double diameter) const -> double {
      return detail::compute_excess_chemical_potential(static_cast<const Derived&>(*this), density, diameter);
    }
  };

  // Rosenfeld (1989)

  struct Rosenfeld : FMTModelBase<Rosenfeld> {
    static constexpr bool NEEDS_TENSOR = false;
    static constexpr std::string_view NAME = "Rosenfeld";

    template <typename T = double>
    [[nodiscard]] static auto f1(T eta) -> T {
      using std::log;
      return log(T(1.0) - eta);
    }

    template <typename T = double>
    [[nodiscard]] static auto f2(T eta) -> T {
      return T(1.0) / (T(1.0) - eta);
    }

    template <typename T = double>
    [[nodiscard]] static auto f3(T eta) -> T {
      T e = T(1.0) - eta;
      return T(1.0) / (e * e);
    }

    [[nodiscard]] static auto phi3(const Measures& m) -> double {
      constexpr double INV_24PI = 1.0 / (24.0 * std::numbers::pi);
      return INV_24PI * (m.n2 * m.n2 * m.n2 - 3.0 * m.n2 * m.products.dot_v1_v1);
    }

    [[nodiscard]] static auto d_phi3_d_n2(const Measures& m) -> double {
      constexpr double INV_8PI = 1.0 / (8.0 * std::numbers::pi);
      return INV_8PI * (m.n2 * m.n2 - m.products.dot_v1_v1);
    }

    [[nodiscard]] static auto d_phi3_d_v1(const Measures& m) -> arma::rowvec3 {
      constexpr double INV_4PI = 1.0 / (4.0 * std::numbers::pi);
      return -INV_4PI * m.n2 * m.v1;
    }
  };

  // RSLT (Rosenfeld-Schmidt-Lowen-Tarazona)

  struct RSLT : FMTModelBase<RSLT> {
    static constexpr bool NEEDS_TENSOR = false;
    static constexpr std::string_view NAME = "RSLT";

    template <typename T = double>
    [[nodiscard]] static auto f1(T eta) -> T {
      using std::log;
      return log(T(1.0) - eta);
    }

    template <typename T = double>
    [[nodiscard]] static auto f2(T eta) -> T {
      return T(1.0) / (T(1.0) - eta);
    }

    template <typename T = double>
    [[nodiscard]] static auto f3(T eta) -> T {
      using std::log;
      if (autodiff::val(eta) < 1e-6) {
        return T(1.0) + (T(2.0) / T(3.0)) * eta + T(0.5) * eta * eta;
      }
      T e = T(1.0) - eta;
      return T(1.0) / (eta * e * e) + log(T(1.0) - eta) / (eta * eta);
    }

    [[nodiscard]] static auto phi3(const Measures& m) -> double {
      if (m.n2 < 1e-30)
        return 0.0;
      double xi = std::min(m.products.dot_v1_v1 / (m.n2 * m.n2), 1.0);
      double q = 1.0 - xi;
      constexpr double INV_36PI = 1.0 / (36.0 * std::numbers::pi);
      return INV_36PI * m.n2 * m.n2 * m.n2 * q * q * q;
    }

    [[nodiscard]] static auto d_phi3_d_n2(const Measures& m) -> double {
      if (m.n2 < 1e-30)
        return 0.0;
      double xi = std::min(m.products.dot_v1_v1 / (m.n2 * m.n2), 1.0);
      double q = 1.0 - xi;
      constexpr double INV_36PI = 1.0 / (36.0 * std::numbers::pi);
      return INV_36PI * 3.0 * m.n2 * m.n2 * q * q * (1.0 + xi);
    }

    [[nodiscard]] static auto d_phi3_d_v1(const Measures& m) -> arma::rowvec3 {
      if (m.n2 < 1e-30)
        return arma::zeros<arma::rowvec>(3);
      double xi = std::min(m.products.dot_v1_v1 / (m.n2 * m.n2), 1.0);
      double q = 1.0 - xi;
      constexpr double INV_36PI = 1.0 / (36.0 * std::numbers::pi);
      return INV_36PI * (-6.0) * m.n2 * q * q * m.v1;
    }
  };

  // White Bear Mark I (Roth et al. 2002)

  struct WhiteBearI : FMTModelBase<WhiteBearI> {
    static constexpr bool NEEDS_TENSOR = true;
    static constexpr std::string_view NAME = "WhiteBearI";

    template <typename T = double>
    [[nodiscard]] static auto f1(T eta) -> T {
      using std::log;
      return log(T(1.0) - eta);
    }

    template <typename T = double>
    [[nodiscard]] static auto f2(T eta) -> T {
      return T(1.0) / (T(1.0) - eta);
    }

    template <typename T = double>
    [[nodiscard]] static auto f3(T eta) -> T {
      using std::log;
      if (autodiff::val(eta) < 1e-6) {
        // Taylor: f3(η) = 3/2 + 8η/3 + 15η²/4 + 24η³/5 + ...
        return T(1.5) + eta * (T(8.0 / 3.0) + eta * (T(15.0 / 4.0) + T(24.0 / 5.0) * eta));
      }
      T e = T(1.0) - eta;
      return T(1.0) / (eta * e * e) + log(T(1.0) - eta) / (eta * eta);
    }

    [[nodiscard]] static auto phi3(const Measures& m) -> double { return detail::tensor_phi3(m); }
    [[nodiscard]] static auto d_phi3_d_n2(const Measures& m) -> double { return detail::tensor_phi3_d_n2(m); }
    [[nodiscard]] static auto d_phi3_d_v1(const Measures& m) -> arma::rowvec3 { return detail::tensor_phi3_d_v1(m); }

    [[nodiscard]] static auto d_phi3_d_T(int i, int j, const Measures& m) -> double {
      return detail::tensor_phi3_d_T(i, j, m);
    }
  };

  // White Bear Mark II (Hansen-Goos & Roth 2006)

  struct WhiteBearII : FMTModelBase<WhiteBearII> {
    static constexpr bool NEEDS_TENSOR = true;
    static constexpr std::string_view NAME = "WhiteBearII";

    template <typename T = double>
    [[nodiscard]] static auto f1(T eta) -> T {
      using std::log;
      return log(T(1.0) - eta);
    }

    template <typename T = double>
    [[nodiscard]] static auto f2(T eta) -> T {
      using std::log;
      if (autodiff::val(eta) < 1e-6) {
        return T(1.0) + eta * (T(1.0) + eta * (T(10.0 / 9.0) + T(7.0 / 6.0) * eta));
      }
      return T(1.0 / 3.0) + T(4.0 / 3.0) / (T(1.0) - eta) +
             T(2.0) / (T(3.0) * eta) * log(T(1.0) - eta);
    }

    template <typename T = double>
    [[nodiscard]] static auto f3(T eta) -> T {
      using std::log;
      if (autodiff::val(eta) < 1e-6) {
        return T(1.5) + T(7.0 / 3.0) * eta + T(3.25) * eta * eta + T(4.2) * eta * eta * eta;
      }
      T e = T(1.0) - eta;
      return -((T(1.0) - T(3.0) * eta + eta * eta) / (eta * e * e)) -
             log(T(1.0) - eta) / (eta * eta);
    }

    [[nodiscard]] static auto phi3(const Measures& m) -> double { return detail::tensor_phi3(m); }
    [[nodiscard]] static auto d_phi3_d_n2(const Measures& m) -> double { return detail::tensor_phi3_d_n2(m); }
    [[nodiscard]] static auto d_phi3_d_v1(const Measures& m) -> arma::rowvec3 { return detail::tensor_phi3_d_v1(m); }

    [[nodiscard]] static auto d_phi3_d_T(int i, int j, const Measures& m) -> double {
      return detail::tensor_phi3_d_T(i, j, m);
    }
  };

  // Explicitly Stable FMT (Lutsko 2010)
  // esFMT(A, B) extends Rosenfeld with a tensor-based Phi3:
  //   Phi3 = (A/(24pi)) * (s2^3 - 3*s2*v2.v2 + 3*v.T.v - T3)
  //        + (B/(24pi)) * (s2^3 - 3*s2*Tr(T^2) + 2*T3)
  // Default A=1, B=0. Same f1, f2, f3 as Rosenfeld but tensor Phi3.

  struct EsFMT : FMTModelBase<EsFMT> {
    static constexpr bool NEEDS_TENSOR = true;
    static constexpr std::string_view NAME = "esFMT";
    double A{1.0};
    double B{0.0};

    template <typename T = double>
    [[nodiscard]] static auto f1(T eta) -> T { return Rosenfeld::f1(eta); }

    template <typename T = double>
    [[nodiscard]] static auto f2(T eta) -> T { return Rosenfeld::f2(eta); }

    template <typename T = double>
    [[nodiscard]] static auto f3(T eta) -> T { return Rosenfeld::f3(eta); }

    [[nodiscard]] auto phi3(const Measures& m) const -> double {
      constexpr double INV_24PI = 1.0 / (24.0 * std::numbers::pi);
      double phi_a = m.n2 * m.n2 * m.n2 - 3.0 * m.n2 * m.products.dot_v1_v1 +
                     3.0 * m.products.quadratic_form - m.products.trace_T3;
      double phi_b = m.n2 * m.n2 * m.n2 - 3.0 * m.n2 * m.products.trace_T2 +
                     2.0 * m.products.trace_T3;
      return INV_24PI * (A * phi_a + B * phi_b);
    }

    [[nodiscard]] auto d_phi3_d_n2(const Measures& m) const -> double {
      constexpr double INV_24PI = 1.0 / (24.0 * std::numbers::pi);
      double da = 3.0 * m.n2 * m.n2 - 3.0 * m.products.dot_v1_v1;
      double db = 3.0 * m.n2 * m.n2 - 3.0 * m.products.trace_T2;
      return INV_24PI * (A * da + B * db);
    }

    [[nodiscard]] auto d_phi3_d_v1(const Measures& m) const -> arma::rowvec3 {
      constexpr double INV_24PI = 1.0 / (24.0 * std::numbers::pi);
      // d/dv1_k of (-3*s2*v1.v1 + 3*v1.T.v1) = -6*s2*v1_k + 3*(T.v1)_k + 3*(v1.T)_k
      arma::rowvec3 tv = m.v1 * m.T + m.v1 * m.T.t();
      return INV_24PI * A * (-6.0 * m.n2 * m.v1 + 3.0 * tv);
    }

    [[nodiscard]] auto d_phi3_d_T(int i, int j, const Measures& m) const -> double {
      constexpr double INV_8PI = 1.0 / (8.0 * std::numbers::pi);
      // A-term: d/dT_ij of (3*v.T.v - T3)/(24pi) = (v_i*v_j - (T.T)_ji)/(8pi)
      // B-term: d/dT_ij of (-3*s2*T2 + 2*T3)/(24pi) = (-s2*T_ji + (T.T)_ji)/(4pi)
      double TT_ji = 0.0;
      for (int k = 0; k < 3; ++k) TT_ji += m.T(j, k) * m.T(k, i);
      return INV_8PI * A * (m.v1(i) * m.v1(j) - TT_ji) +
             (1.0 / (4.0 * std::numbers::pi)) * B * (-m.n2 * m.T(j, i) + TT_ji);
    }
  };

  // FMTModel wrapper — hides variant, exposes unified interface.
  // Constructible from any concrete FMT model type.

  class FMTModel {
   public:
    FMTModel() = default;

    template <typename T>
      requires(!std::is_same_v<std::decay_t<T>, FMTModel>)
    FMTModel(T concrete) : data_(std::move(concrete)) {}

    [[nodiscard]] auto phi(const Measures& m) const -> double {
      return std::visit([&m](const auto& f) { return f.phi(m); }, data_);
    }

    [[nodiscard]] auto d_phi(const Measures& m) const -> MeasureDerivatives {
      return std::visit([&m](const auto& f) { return f.d_phi(m); }, data_);
    }

    [[nodiscard]] auto free_energy_density(double density, double diameter) const -> double {
      return std::visit([density, diameter](const auto& f) { return f.free_energy_density(density, diameter); }, data_);
    }

    [[nodiscard]] auto excess_chemical_potential(double density, double diameter) const -> double {
      return std::visit(
          [density, diameter](const auto& f) { return f.excess_chemical_potential(density, diameter); }, data_
      );
    }

    [[nodiscard]] auto needs_tensor() const -> bool {
      return std::visit([](const auto& f) { return std::decay_t<decltype(f)>::NEEDS_TENSOR; }, data_);
    }

    [[nodiscard]] auto name() const -> std::string_view {
      return std::visit([](const auto& f) -> std::string_view { return f.NAME; }, data_);
    }

    [[nodiscard]] static auto from_name(std::string_view name) -> FMTModel {
      if (name == "Rosenfeld") return Rosenfeld{};
      if (name == "RSLT") return RSLT{};
      if (name == "WhiteBearI") return WhiteBearI{};
      if (name == "WhiteBearII") return WhiteBearII{};
      if (name == "esFMT") return EsFMT{};
      throw std::invalid_argument(std::string("Unknown FMT model: ") + std::string(name));
    }

    // Access underlying variant for rare type-specific inspection
    using VariantType = std::variant<Rosenfeld, RSLT, WhiteBearI, WhiteBearII, EsFMT>;
    [[nodiscard]] auto variant() const -> const VariantType& { return data_; }

   private:
    VariantType data_;
  };

}  // namespace dft::functionals::fmt

#endif  // DFT_FUNCTIONALS_FMT_MODELS_HPP
