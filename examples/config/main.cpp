#include "dft.hpp"

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

using namespace dft;

int main() {
#ifdef EXAMPLE_SOURCE_DIR
  std::filesystem::current_path(EXAMPLE_SOURCE_DIR);
#endif

  std::vector<std::pair<std::string, config::FileType>> files = {
      {"config.ini", config::FileType::INI},
      {"config.json", config::FileType::JSON},
  };

  for (const auto& [path, type] : files) {
    auto data = config::parse_config(path, type);

    auto string_value = config::get<std::string>(data, "default.StringValue");
    auto double_value = config::get<double>(data, "default.DoubleValue");

    console::new_line();
    console::info("File: " + path);
    std::cout << "  StringValue = " << string_value << "\n";
    std::cout << "  DoubleValue = " << double_value << "\n";
  }
}
