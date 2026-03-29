#include <cmath>
#include <armadillo>
#include "dft.h"
#include <filesystem>
#include <cstring>

int main(int argc, char **argv)
{
#ifdef EXAMPLE_SOURCE_DIR
  std::filesystem::current_path(EXAMPLE_SOURCE_DIR);
#endif
  std::filesystem::create_directories("exports");
  using namespace dft::geometry;
  using namespace dft;

  // region Vertex: Initializer cttor
  console::info("Vertex | Constructor");
  auto v1 = Vertex({0, 1, 2, 3});
  auto v2 = Vertex({3, 4, 5, 6});

  std::cout << "v1: " << v1 << std::endl;
  std::cout << "v2: " << v2 << std::endl << std::endl;
  // endregion

  // region Element: Cttor with pass-by-ref std::vector<Vertex> (copy)
  auto v_list = std::vector<Vertex>{ v1, v2 };

  console::info("Element | Copy constructor");
  auto e1 = Element(v_list);

  std::cout << "v_list[0]: " << v_list[0] << std::endl
            << "v_list[1]: " << v_list[1] << std::endl << std::endl;

  std::cout << "Element 1: " << std::endl
            << "\u2b91 raw[0] = " << e1.vertices_raw()[0] << std::endl
            << "\u2b91 map[0]  = " << e1.vertices().at(0).get() << std::endl
            << "\u2b91 raw[1] = " << e1.vertices_raw()[1] << std::endl
            << "\u2b91 map[1]  = " << e1.vertices().at(1).get() << std::endl << std::endl;

  // endregion

  // region Element: Cttor with std::move of std::vector<Vertex> (move)
  console::info("Element | Move semantics");

  std::cout << "v_list[0]: " << v_list[0] << std::endl
            << "v_list[1]: " << v_list[1] << std::endl << std::endl;

  auto e2 = Element(std::move(v_list));

  std::cout << "Element 2: " << std::endl
            << "\u2b91 raw[0] = " << e2.vertices_raw()[0] << std::endl
            << "\u2b91 map[0]  = " << e2[0] << std::endl
            << "\u2b91 raw[1] = " << e2.vertices_raw()[1] << std::endl
            << "\u2b91 map[1]  = " << e2[1] << std::endl << std::endl;

  std::cout << e2 << std::endl;
  // endregion

  // region 2D: SquareBox
  console::info("Square boxes 2D: default");
  auto default_box_2D = two_dimensional::SquareBox();
  std::cout << "Default square-box:" << std::endl
            << default_box_2D << std::endl;

  console::info("Square boxes 2D: customized");
  auto custom_box_2D = two_dimensional::SquareBox(0.25, {0,0});
  std::cout << "Customised square-box:" << std::endl
            << custom_box_2D << std::endl;

  auto v_list_b = std::vector<Vertex>{ {0,0}, {0,1}, {1,1}, {1,0} };
  std::cout << "Move-semantics square-box:" << std::endl
            << "v_list[3]: " << v_list_b[3] << std::endl
            << "v_list[2]: " << v_list_b[2] << std::endl
            << "v_list[1]: " << v_list_b[1] << std::endl
            << "v_list[0]: " << v_list_b[0] << std::endl << std::endl;

  auto move_box_2D = two_dimensional::SquareBox(std::move(v_list_b));
  std::cout << move_box_2D << std::endl;
  // endregion

  // region 2D::SUQMesh
  console::info("Mesh 2D: SUQMesh");

  auto origin_2D = std::vector<double>{0,0};
  auto lengths_2D = std::vector<double>{1.0,1.0};
  auto mesh_2D = two_dimensional::SUQMesh(0.25, lengths_2D, origin_2D);

  std::cout << mesh_2D << std::endl;

  std::cout << "Vertex[-1,-1]: " << mesh_2D[{-1,-1}] << std::endl;
  std::cout << "Vertex[2,3]: " << mesh_2D[{2,3}] << std::endl;

  std::cout << "Number of elements: " << mesh_2D.elements().size() << std::endl;
  std::cout << "Element volume: " << mesh_2D.element_volume() << std::endl;

  console::info("Mesh 2D: SUQMesh plot");

  // endregion

  // region 3D::SquareBox
  console::info("Square boxes 3D: default");
  auto default_box_3D = three_dimensional::SquareBox();
  std::cout << "Default square-box (3D):" << std::endl
            << default_box_3D << std::endl;

  console::info("Mesh 3D: SUQMesh");
  auto origin_3D = std::vector<double>{0,0,0};
  auto lengths_3D = std::vector<double>{1.0,1.0,1.0};
  auto mesh_3D = three_dimensional::SUQMesh(0.25, lengths_3D, origin_3D);

  std::cout << mesh_3D << std::endl;
  std::cout << "Vertex[-1,-1,-1]: " << mesh_3D[{-1,-1,-1}] << std::endl;
  std::cout << "Vertex[2,3,1]: " << mesh_3D[{2,3,1}] << std::endl;

  std::cout << "Number of elements: " << mesh_3D.elements().size() << std::endl;
  std::cout << "Element volume: " << mesh_3D.element_volume() << std::endl;
  // endregion

  // region 2D::UniformMesh
  console::info("UniformMesh 2D: periodic boundary conditions");

  auto um_origin_2D = std::vector<double>{0, 0};
  auto um_lengths_2D = std::vector<double>{4.0, 4.0};
  auto uniform_2D = two_dimensional::UniformMesh(1.0, um_lengths_2D, um_origin_2D);

  std::cout << uniform_2D << std::endl;
  std::cout << "Spacing: " << uniform_2D.spacing() << std::endl;
  std::cout << "Shape: [" << uniform_2D.shape()[0] << ", " << uniform_2D.shape()[1] << "]" << std::endl;
  std::cout << "Element volume: " << uniform_2D.element_volume() << std::endl;

  // PBC wrapping
  std::cout << std::endl;
  std::cout << "wrap({1.5, 2.5}):  " << uniform_2D.wrap(Vertex({1.5, 2.5})) << std::endl;
  std::cout << "wrap({5.5, 7.0}):  " << uniform_2D.wrap(Vertex({5.5, 7.0})) << std::endl;
  std::cout << "wrap({-1.0, -0.5}): " << uniform_2D.wrap(Vertex({-1.0, -0.5})) << std::endl;

  console::info("UniformMesh 2D: plot");
  // endregion

  // region 3D::UniformMesh
  console::info("UniformMesh 3D: periodic boundary conditions");

  auto um_origin_3D = std::vector<double>{0, 0, 0};
  auto um_lengths_3D = std::vector<double>{4.0, 4.0, 4.0};
  auto uniform_3D = three_dimensional::UniformMesh(1.0, um_lengths_3D, um_origin_3D);

  std::cout << uniform_3D << std::endl;
  std::cout << "Spacing: " << uniform_3D.spacing() << std::endl;
  std::cout << "Shape: [" << uniform_3D.shape()[0] << ", " << uniform_3D.shape()[1]
            << ", " << uniform_3D.shape()[2] << "]" << std::endl;
  std::cout << "Element volume: " << uniform_3D.element_volume() << std::endl;

  // PBC wrapping
  std::cout << std::endl;
  std::cout << "wrap({1.5, 2.5, 3.5}):     " << uniform_3D.wrap(Vertex({1.5, 2.5, 3.5})) << std::endl;
  std::cout << "wrap({5.5, 7.0, 12.5}):    " << uniform_3D.wrap(Vertex({5.5, 7.0, 12.5})) << std::endl;
  std::cout << "wrap({-1.0, -0.5, -5.0}):  " << uniform_3D.wrap(Vertex({-1.0, -0.5, -5.0})) << std::endl;
  // endregion

  // ── Plots ──────────────────────────────────────────────────────────────

  mesh_2D.plot("exports/mesh_2d.png");
  mesh_3D.plot("exports/mesh_3d_xy.png");
}
