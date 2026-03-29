#include "classicaldft_bits/geometry/base/vertex.h"

namespace dft::geometry {

  // region Cttors:
  Vertex::Vertex(const vec_type& x) : dimension_(static_cast<int>(x.size())), coordinates_(x) {}
  Vertex::Vertex(vec_type&& x) : dimension_(static_cast<int>(x.size())), coordinates_(std::move(x)) {}
  Vertex::Vertex(const std::initializer_list<x_type>& x) : dimension_(static_cast<int>(x.size())), coordinates_(x) {}
  // endregion

  // region Methods:
  int Vertex::dimension() const {
    return dimension_;
  }
  const std::vector<x_type>& Vertex::coordinates() const {
    return coordinates_;
  }
  // endregion

  // region Overloads:
  const x_type& Vertex::operator[](int k) const {
    return coordinates_.at(k);
  }

  std::ostream& operator<<(std::ostream& os, const Vertex& vertex) {
    os << "(dimensions = " << vertex.dimension() << "): ";
    if (vertex.dimension() > 0) {
      os << "(";
      for (int k = 0; k < vertex.dimension() - 1; k++) {
        os << vertex.coordinates()[k] << ", ";
      }
      os << vertex.coordinates()[vertex.dimension() - 1] << ")";
    } else {
      os << "()";
    }
    os << " [front \u27fc " << std::addressof(vertex.coordinates_.front()) << "]";
    return os;
  }

  Vertex operator+(const Vertex& a, const Vertex& b) {
    if (a.dimension_ != b.dimension_) {
      throw std::runtime_error("The vertices you're trying to add don't have the same dimension");
    }

    vec_type x;
    x.reserve(a.coordinates().size());
    for (size_t i = 0; i < a.coordinates().size(); ++i) {
      x.push_back(a.coordinates()[i] + b.coordinates()[i]);
    }

    return Vertex(std::move(x));
  }

  Vertex operator-(const Vertex& a, const Vertex& b) {
    if (a.dimension_ != b.dimension_) {
      throw std::runtime_error("The vertices you're trying to add don't have the same dimension");
    }

    vec_type x;
    x.reserve(a.coordinates().size());
    for (size_t i = 0; i < a.coordinates().size(); ++i) {
      x.push_back(a.coordinates()[i] - b.coordinates()[i]);
    }

    return Vertex(std::move(x));
  }
  // endregion
}  // namespace dft::geometry