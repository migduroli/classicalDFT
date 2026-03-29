#include "classicaldft_bits/io/config.h"

#include <fstream>
#include <gtest/gtest.h>

using namespace dft;

// ── Default constructor ─────────────────────────────────────────────────────

TEST(ConfigParser, DefaultConstructorUsesIniFile) {
  auto config = io::config::ConfigParser();
  ASSERT_EQ(config.config_file_path(), "config.ini");
  ASSERT_EQ(config.config_file_type(), io::config::FileType::INI);
}

// ── Specific constructors (parameterized) ───────────────────────────────────

struct ConfigFileTestCase {
  std::string name;
  std::string file;
  io::config::FileType type;
};

class ConfigParserFileTest : public testing::TestWithParam<ConfigFileTestCase> {};

TEST_P(ConfigParserFileTest, ConstructsAndReadsCorrectly) {
  auto [name, file, type] = GetParam();
  auto config = io::config::ConfigParser(file, type);
  ASSERT_EQ(config.config_file_path(), file);
  ASSERT_EQ(config.config_file_type(), type);

  auto actual_string = config.get<std::string>("default.StringValue");
  auto actual_double = config.get<double>("default.DoubleValue");

  ASSERT_EQ(actual_string, "a_text_string");
  ASSERT_DOUBLE_EQ(actual_double, 10.0);
}

INSTANTIATE_TEST_SUITE_P(
    AllFormats,
    ConfigParserFileTest,
    testing::Values(
        ConfigFileTestCase{"INI", "config.ini", io::config::FileType::INI},
        ConfigFileTestCase{"JSON", "config.json", io::config::FileType::JSON}
    ),
    [](const auto& info) { return info.param.name; }
);

// ── get() accessor ──────────────────────────────────────────────────────────

TEST(ConfigParser, GetReturnsCorrectValues) {
  auto config = io::config::ConfigParser("config.ini");
  ASSERT_DOUBLE_EQ(config.get<double>("default.DoubleValue"), 10.0);
  ASSERT_EQ(config.get<std::string>("default.StringValue"), "a_text_string");
}

TEST(ConfigParser, DataAccessorReturnsJson) {
  auto config = io::config::ConfigParser("config.json", io::config::FileType::JSON);
  EXPECT_FALSE(config.data().empty());
}

// ── Exception paths ─────────────────────────────────────────────────────────

TEST(ConfigParser, ThrowsOnMissingIniFile) {
  EXPECT_THROW(io::config::ConfigParser("nonexistent.ini", io::config::FileType::INI), std::runtime_error);
}

TEST(ConfigParser, ThrowsOnMissingJsonFile) {
  EXPECT_THROW(io::config::ConfigParser("nonexistent.json", io::config::FileType::JSON), std::runtime_error);
}

TEST(ConfigParser, ThrowsOnMissingKey) {
  auto config = io::config::ConfigParser("config.ini");
  EXPECT_THROW((void)config.get<std::string>("nonexistent.key"), std::runtime_error);
}

// ── INI parser edge cases ───────────────────────────────────────────────────

class IniEdgeCaseTest : public testing::Test {
 protected:
  void SetUp() override {
    // Create a temporary INI file with edge cases
    std::ofstream f("edge_case.ini");
    f << "; This is a comment\n";
    f << "# This is also a comment\n";
    f << "\n";
    f << "global_key = global_value\n";
    f << "\n";
    f << "[section]\n";
    f << "key = value\n";
    f << "number = 3.14\n";
    f.close();
  }

  void TearDown() override { std::remove("edge_case.ini"); }
};

TEST_F(IniEdgeCaseTest, CommentsAreSkipped) {
  auto config = io::config::ConfigParser("edge_case.ini", io::config::FileType::INI);
  EXPECT_EQ(config.get<std::string>("section.key"), "value");
}

TEST_F(IniEdgeCaseTest, GlobalKeysWork) {
  auto config = io::config::ConfigParser("edge_case.ini", io::config::FileType::INI);
  EXPECT_EQ(config.get<std::string>("global_key"), "global_value");
}

TEST_F(IniEdgeCaseTest, NumericValuesAreParsed) {
  auto config = io::config::ConfigParser("edge_case.ini", io::config::FileType::INI);
  EXPECT_DOUBLE_EQ(config.get<double>("section.number"), 3.14);
}

// ── INI malformed section ────────────────────────────────────────────────────

TEST(ConfigParser, MalformedIniSectionThrows) {
  std::ofstream f("malformed.ini");
  f << "[bad_section\n";
  f << "key = value\n";
  f.close();
  EXPECT_THROW(io::config::ConfigParser("malformed.ini", io::config::FileType::INI), std::runtime_error);
  std::remove("malformed.ini");
}

TEST(ConfigParser, IniLineWithoutEqualsIsSkipped) {
  std::ofstream f("bare_line.ini");
  f << "[section]\n";
  f << "bare_line_no_equals\n";
  f << "key = value\n";
  f.close();
  auto config = io::config::ConfigParser("bare_line.ini", io::config::FileType::INI);
  EXPECT_EQ(config.get<std::string>("section.key"), "value");
  std::remove("bare_line.ini");
}

// ── Setters ─────────────────────────────────────────────────────────────────

TEST(ConfigParser, SetConfigFilePath) {
  auto config = io::config::ConfigParser("config.ini");
  config.set_config_file_path("new_path.ini");
  EXPECT_EQ(config.config_file_path(), "new_path.ini");
}

TEST(ConfigParser, SetConfigFileType) {
  auto config = io::config::ConfigParser("config.ini");
  config.set_config_file_type(io::config::FileType::JSON);
  EXPECT_EQ(config.config_file_type(), io::config::FileType::JSON);
}

TEST(ConfigParser, ReadConfigFileReloads) {
  auto config = io::config::ConfigParser("config.ini");
  config.read_config_file("config.json", io::config::FileType::JSON);
  EXPECT_EQ(config.config_file_path(), "config.json");
  EXPECT_EQ(config.config_file_type(), io::config::FileType::JSON);
  EXPECT_DOUBLE_EQ(config.get<double>("default.DoubleValue"), 10.0);
}
