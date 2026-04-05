#include "dft/config/parser.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace dft::config;

static const std::string TEST_DIR = std::string(DFT_TEST_DATA_DIR) + "/unit/config/config_files/";

TEST_CASE("parse_config reads INI file", "[config]") {
  auto data = parse_config(TEST_DIR + "config.ini", FileType::INI);
  CHECK(data.contains("section"));
  CHECK(data["section"]["key"] == "value");
}

TEST_CASE("parse_config reads JSON file", "[config]") {
  auto data = parse_config(TEST_DIR + "config.json", FileType::JSON);
  CHECK(data.contains("section"));
  CHECK(data["section"]["key"] == "value");
}

TEST_CASE("parse_config INI parses numeric values", "[config]") {
  auto data = parse_config(TEST_DIR + "config.ini", FileType::INI);
  CHECK(data["section"]["number"] == 42.0);
  CHECK(data["section"]["pi"].get<double>() == Catch::Approx(3.14159));
}

TEST_CASE("parse_config INI parses global keys", "[config]") {
  auto data = parse_config(TEST_DIR + "config.ini", FileType::INI);
  CHECK(data["global_key"] == "global_value");
}

TEST_CASE("parse_config INI skips comments", "[config]") {
  auto data = parse_config(TEST_DIR + "config.ini", FileType::INI);
  // Comments should not appear as keys
  for (auto& [key, val] : data.items()) {
    CHECK(key.front() != ';');
  }
}

TEST_CASE("get retrieves values by dotted path", "[config]") {
  auto data = parse_config(TEST_DIR + "config.json", FileType::JSON);
  CHECK(get<std::string>(data, "section.key") == "value");
  CHECK(get<int>(data, "section.number") == 42);
  CHECK(get<double>(data, "section.pi") == Catch::Approx(3.14159));
}

TEST_CASE("get retrieves nested values", "[config]") {
  auto data = parse_config(TEST_DIR + "config.json", FileType::JSON);
  CHECK(get<std::string>(data, "database.host") == "localhost");
  CHECK(get<int>(data, "database.port") == 5432);
}

TEST_CASE("get throws for missing key", "[config]") {
  auto data = parse_config(TEST_DIR + "config.json", FileType::JSON);
  REQUIRE_THROWS_AS(get<std::string>(data, "nonexistent.key"), std::runtime_error);
}

TEST_CASE("parse_config throws for missing file", "[config]") {
  REQUIRE_THROWS_AS(parse_config("/nonexistent/path.ini", FileType::INI), std::runtime_error);
  REQUIRE_THROWS_AS(parse_config("/nonexistent/path.json", FileType::JSON), std::runtime_error);
}

TEST_CASE("parse_config INI throws for malformed section header", "[config]") {
  REQUIRE_THROWS_AS(parse_config(TEST_DIR + "malformed.ini", FileType::INI), std::runtime_error);
}
