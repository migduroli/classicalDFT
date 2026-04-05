#ifndef DFT_CONFIG_PARSER_HPP
#define DFT_CONFIG_PARSER_HPP

#include <cstdint>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <string>

namespace dft::config {

  enum class FileType : std::uint8_t {
    INI,
    JSON
  };

  [[nodiscard]] auto parse_config(const std::string& path, FileType type = FileType::INI) -> nlohmann::json;

  template <typename T> [[nodiscard]] auto get(const nlohmann::json& data, const std::string& dotted_path) -> T {
    const nlohmann::json* node = &data;
    std::string token;
    std::istringstream stream(dotted_path);
    while (std::getline(stream, token, '.')) {
      if (!node->contains(token)) {
        throw std::runtime_error("Key not found: " + dotted_path);
      }
      node = &(*node)[token];
    }
    return node->get<T>();
  }

} // namespace dft::config

#endif // DFT_CONFIG_PARSER_HPP
