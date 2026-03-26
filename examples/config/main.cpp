#include <classicaldft>

int main() {
  using namespace dft_core;
  using namespace dft_core::io;

  //region Default Cttor:

  auto config = config_parser::ConfigParser();

  auto x = config.get<std::string>("default.StringValue");
  auto y = config.get<double>("default.DoubleValue");

  console::info("This is x: " + x);
  console::info("This is 2*y: " + std::to_string(2*y));

  console::wait();

  //endregion

  //region Specific Cttor: INI and JSON formats

  std::vector<config_parser::FileType> types {
      config_parser::FileType::INI,
      config_parser::FileType::JSON,
  };

  std::vector<std::string> files {
      "config.ini",
      "config.json",
  };

  for (size_t i = 0; i < types.size(); ++i) {
    auto c_obj = config_parser::ConfigParser(files[i], types[i]);

    auto string_value = c_obj.get<std::string>("default.StringValue");
    auto double_value = c_obj.get<double>("default.DoubleValue");

    console::new_line();
    console::info("File name: " + files[i]);
    console::info("String value: " + string_value);
    console::info("Double value: " + std::to_string(double_value));
  }
  console::wait();

  //endregion
}
