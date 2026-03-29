#include "cdft/config.hpp"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

namespace cdft::io {

  class ConfigParserTest : public ::testing::Test {
   protected:
    void SetUp() override {
      ini_path_ = std::filesystem::temp_directory_path() / "cdft_test_config.ini";
      json_path_ = std::filesystem::temp_directory_path() / "cdft_test_config.json";

      {
        std::ofstream f(ini_path_);
        f << "[section]\n"
          << "key_int = 42\n"
          << "key_double = 3.14\n"
          << "key_string = hello world\n";
      }

      {
        std::ofstream f(json_path_);
        f << R"({
  "section": {
    "key_int": 42,
    "key_double": 3.14,
    "key_string": "hello world",
    "key_bool": true
  }
})";
      }
    }

    void TearDown() override {
      std::filesystem::remove(ini_path_);
      std::filesystem::remove(json_path_);
    }

    std::filesystem::path ini_path_;
    std::filesystem::path json_path_;
  };

  TEST_F(ConfigParserTest, ParseINI) {
    ConfigParser parser(ini_path_.string());
    EXPECT_EQ(parser.get<int>("section.key_int"), 42);
    EXPECT_DOUBLE_EQ(parser.get<double>("section.key_double"), 3.14);
  }

  TEST_F(ConfigParserTest, ParseJSON) {
    ConfigParser parser(json_path_.string(), ConfigFormat::JSON);
    EXPECT_EQ(parser.get<int>("section.key_int"), 42);
    EXPECT_DOUBLE_EQ(parser.get<double>("section.key_double"), 3.14);
  }

  TEST_F(ConfigParserTest, GetStringINI) {
    ConfigParser parser(ini_path_.string());
    EXPECT_EQ(parser.get<std::string>("section.key_string"), "hello world");
  }

  TEST_F(ConfigParserTest, GetBoolJSON) {
    ConfigParser parser(json_path_.string(), ConfigFormat::JSON);
    EXPECT_TRUE(parser.get<bool>("section.key_bool"));
  }

  TEST_F(ConfigParserTest, MissingKeyThrows) {
    ConfigParser parser(ini_path_.string());
    EXPECT_THROW(parser.get<int>("section.missing"), std::invalid_argument);
  }

  TEST_F(ConfigParserTest, MissingFileThrows) {
    EXPECT_THROW(ConfigParser("/nonexistent/file.ini"), std::runtime_error);
  }

  TEST_F(ConfigParserTest, DataAccessible) {
    ConfigParser parser(ini_path_.string());
    EXPECT_TRUE(parser.data().contains("section"));
    EXPECT_FALSE(parser.data().contains("nonexistent"));
  }

}  // namespace cdft::io
