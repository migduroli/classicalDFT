#pragma once

#include <fstream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <string>

namespace cdft::io {

  enum class ConfigFormat { INI, JSON };

  class ConfigParser {
   public:
    explicit ConfigParser(std::string path, ConfigFormat format = ConfigFormat::INI);

    [[nodiscard]] const std::string& path() const noexcept { return path_; }
    [[nodiscard]] ConfigFormat format() const noexcept { return format_; }
    [[nodiscard]] const nlohmann::json& data() const noexcept { return data_; }

    template <typename T>
    [[nodiscard]] T get(const std::string& dotted_path) const {
      const nlohmann::json* node = &data_;
      std::string token;
      std::istringstream stream(dotted_path);
      while (std::getline(stream, token, '.')) {
        if (!node->contains(token)) {
          throw std::invalid_argument("Key not found: " + dotted_path);
        }
        node = &(*node)[token];
      }
      return node->get<T>();
    }

   private:
    static nlohmann::json parse_ini(const std::string& path);

    std::string path_;
    ConfigFormat format_;
    nlohmann::json data_;
  };

}  // namespace cdft::io
