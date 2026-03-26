#include <classicaldft>

#include <boost/range/combine.hpp>

int main() {
  using namespace dft_core;

  //region Default Cttor:

  auto config = config_parser::ConfigParser();

  auto x = config.tree().get<std::string>("default.StringValue");
  auto y = config.tree().get<double>("default.DoubleValue");

  console::info("This is x: " + x);
  console::info("This is 2*y: " + std::to_string(2*y));

  console::wait();

  //endregion

  //region Specific Cttor: All file formats

  std::vector<dft_core::config_parser::FileType> types {
      dft_core::config_parser::FileType::INI,
      dft_core::config_parser::FileType::JSON,
      dft_core::config_parser::FileType::XML,
      dft_core::config_parser::FileType::INFO
  };

  std::vector<std::string> files {
      "config.ini",
      "config.json",
      "config.xml",
      "config.info"
  };

  std::string f;
  dft_core::config_parser::FileType t;

  for (const auto& tuple : boost::combine(types, files)) {
    boost::tie(t, f) = tuple;
    auto c_obj = dft_core::config_parser::ConfigParser(f, t);

    std::string section = (c_obj.config_file_path().find("info") == std::string::npos) ? "default." : "";
    auto string_value = c_obj.tree().get<std::string>(section + "StringValue");
    auto double_value = c_obj.tree().get<double>(section + "DoubleValue");

    console::new_line();
    console::info("File name: " + f);
    console::info("String value: " + string_value);
    console::info("Double value: " + std::to_string(double_value));
  }
  console::wait();

  //endregion
}
