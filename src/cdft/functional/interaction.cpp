#include "cdft/functional/interaction.hpp"

#include <cmath>
#include <gsl/gsl_integration.h>
#include <numbers>
#include <stdexcept>

namespace cdft::functional {

  Interaction::Interaction(
      Species& species_1, Species& species_2,
      const physics::PairPotential& potential,
      double kT, WeightScheme scheme, int gauss_order)
      : species_1_(species_1),
        species_2_(species_2),
        self_interaction_(&species_1 == &species_2),
        dV_(species_1.density().cell_volume()),
        convolution_field_(species_1.density().shape()),
        weight_(species_1.density().shape()) {
    if (kT <= 0.0) throw std::invalid_argument("Interaction: kT must be positive");
    if (species_1.density().spacing() != species_2.density().spacing())
      throw std::invalid_argument("Interaction: species must share the same grid spacing");
    if (species_1.density().shape() != species_2.density().shape())
      throw std::invalid_argument("Interaction: species must share the same grid shape");

    // Generate weights
    double dx = species_1.density().spacing();
    auto shape = species_1.density().shape();
    long nx = shape[0], ny = shape[1], nz = shape[2];
    long n = nx * ny * nz;

    double r_cut = physics::get_config(potential).r_cutoff;
    double r_cut_sq = r_cut * r_cut;

    std::vector<double> w_real(static_cast<std::size_t>(n), 0.0);
    double weight_sum = 0.0;

    for (long ix = 0; ix < nx; ++ix) {
      long sx = (ix <= nx / 2) ? ix : ix - nx;
      for (long iy = 0; iy < ny; ++iy) {
        long sy = (iy <= ny / 2) ? iy : iy - ny;
        for (long iz = 0; iz < nz; ++iz) {
          long sz = (iz <= nz / 2) ? iz : iz - nz;

          double dist2 = static_cast<double>(sx * sx + sy * sy + sz * sz) * dx * dx;
          if (r_cut > 0.0 && dist2 > r_cut_sq) continue;

          double w = 0.0;

          switch (scheme) {
            case WeightScheme::InterpolationZero: {
              double r2 = dist2;
              if (r2 < 1e-30) {
                w = physics::w_attractive(potential, 0.0) / kT;
              } else {
                w = physics::w_attractive_r2(potential, r2) / kT;
              }
              break;
            }
            case WeightScheme::InterpolationLinearE:
            case WeightScheme::InterpolationLinearF: {
              double sum = 0.0;
              for (int di = 0; di <= 1; ++di)
                for (int dj = 0; dj <= 1; ++dj)
                  for (int dk = 0; dk <= 1; ++dk) {
                    double x_p = (static_cast<double>(sx) - 0.5 + static_cast<double>(di)) * dx;
                    double y_p = (static_cast<double>(sy) - 0.5 + static_cast<double>(dj)) * dx;
                    double z_p = (static_cast<double>(sz) - 0.5 + static_cast<double>(dk)) * dx;
                    double r2 = x_p * x_p + y_p * y_p + z_p * z_p;
                    if (r2 < 1e-30) {
                      sum += physics::w_attractive(potential, 0.0) / kT;
                    } else {
                      sum += physics::w_attractive_r2(potential, r2) / kT;
                    }
                  }
              w = sum / 8.0;
              break;
            }
            case WeightScheme::GaussE:
            case WeightScheme::GaussF: {
              gsl_integration_glfixed_table* table =
                  gsl_integration_glfixed_table_alloc(static_cast<size_t>(gauss_order));
              double sum = 0.0;
              double total_weight = 0.0;

              for (int i = 0; i < gauss_order; ++i) {
                double xi = 0, wi = 0;
                gsl_integration_glfixed_point(-0.5, 0.5, static_cast<size_t>(i), &xi, &wi, table);
                double x_p = (static_cast<double>(sx) + xi) * dx;
                for (int j = 0; j < gauss_order; ++j) {
                  double xj = 0, wj = 0;
                  gsl_integration_glfixed_point(-0.5, 0.5, static_cast<size_t>(j), &xj, &wj, table);
                  double y_p = (static_cast<double>(sy) + xj) * dx;
                  for (int k = 0; k < gauss_order; ++k) {
                    double xk = 0, wk = 0;
                    gsl_integration_glfixed_point(-0.5, 0.5, static_cast<size_t>(k), &xk, &wk, table);
                    double z_p = (static_cast<double>(sz) + xk) * dx;
                    double r2 = x_p * x_p + y_p * y_p + z_p * z_p;
                    double w3 = wi * wj * wk;
                    total_weight += w3;
                    if (r2 < 1e-30) {
                      sum += w3 * physics::w_attractive(potential, 0.0) / kT;
                    } else {
                      sum += w3 * physics::w_attractive_r2(potential, r2) / kT;
                    }
                  }
                }
              }
              gsl_integration_glfixed_table_free(table);
              w = sum / total_weight;
              break;
            }
          }

          auto idx = static_cast<std::size_t>(iz + nz * (iy + ny * ix));
          w_real[idx] = w * dV_;
          weight_sum += w * dV_;
        }
      }
    }

    a_vdw_ = weight_sum;
    weight_.set_weight_from_real(std::span<const double>(w_real.data(), w_real.size()));
  }

  double Interaction::compute_free_energy() {
    species_2_.density().forward_fft();
    auto rho2_fourier = species_2_.density().fft().fourier();
    weight_.convolve(rho2_fourier);

    const arma::vec& rho1 = species_1_.density().values();
    auto w_field = weight_.field().real();

    double energy = 0.0;
    for (arma::uword i = 0; i < rho1.n_elem; ++i) {
      energy += rho1(i) * w_field[i];
    }
    return 0.5 * energy * dV_;
  }

  double Interaction::compute_forces() {
    species_2_.density().forward_fft();
    auto rho2_fourier = species_2_.density().fft().fourier();
    weight_.convolve(rho2_fourier);

    const arma::vec& rho1 = species_1_.density().values();
    auto w_field = weight_.field().real();

    double energy = 0.0;
    for (arma::uword i = 0; i < rho1.n_elem; ++i) {
      energy += rho1(i) * w_field[i];
    }
    energy *= 0.5 * dV_;

    if (self_interaction_) {
      for (arma::uword i = 0; i < rho1.n_elem; ++i) {
        species_1_.force()(i) += w_field[i] * dV_;
      }
    } else {
      for (arma::uword i = 0; i < rho1.n_elem; ++i) {
        species_1_.force()(i) += w_field[i] * (0.5 * dV_);
      }

      species_1_.density().forward_fft();
      auto rho1_fourier = species_1_.density().fft().fourier();
      weight_.convolve(rho1_fourier);
      auto field_from_rho1 = weight_.field().real();
      for (arma::uword i = 0; i < species_2_.density().size(); ++i) {
        species_2_.force()(i) += field_from_rho1[i] * (0.5 * dV_);
      }
    }

    return energy;
  }

  double Interaction::bulk_excess_free_energy(double density) const {
    return 0.5 * a_vdw_ * density * density;
  }

  double Interaction::bulk_excess_chemical_potential(double density) const {
    return a_vdw_ * density;
  }

}  // namespace cdft::functional
