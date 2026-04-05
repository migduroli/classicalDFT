#ifndef DFT_FUNCTIONALS_TYPES_HPP
#define DFT_FUNCTIONALS_TYPES_HPP

#include <armadillo>
#include <vector>

namespace dft::functionals {

  // A single functional contribution: free energy plus per-species forces.

  struct Contribution {
    double free_energy{ 0.0 };
    std::vector<arma::vec> forces;
  };

  // Complete result of a DFT functional evaluation.

  struct Result {
    double free_energy{ 0.0 };
    double grand_potential{ 0.0 };
    std::vector<arma::vec> forces;
  };

}  // namespace dft::functionals

#endif  // DFT_FUNCTIONALS_TYPES_HPP
