#include "dft/console.h"

#include <gtest/gtest.h>
#include <sstream>

using namespace dft;

// ── write / write_line / new_line ───────────────────────────────────────────

TEST(Console, WriteOutputsToStdout) {
  testing::internal::CaptureStdout();
  console::write("test");
  ASSERT_EQ(testing::internal::GetCapturedStdout(), "test");
}

TEST(Console, WriteLineOutputsWithNewline) {
  testing::internal::CaptureStdout();
  console::write_line("test");
  ASSERT_EQ(testing::internal::GetCapturedStdout(), "test\n");
}

TEST(Console, WriteLineInitializerList) {
  testing::internal::CaptureStdout();
  console::write_line({"test", "one", "two"});
  ASSERT_EQ(testing::internal::GetCapturedStdout(), "test\none\ntwo\n");
}

TEST(Console, NewLine) {
  testing::internal::CaptureStdout();
  console::new_line();
  ASSERT_EQ(testing::internal::GetCapturedStdout(), "\n");
}

// ── format::bold / format::blink ────────────────────────────────────────────

TEST(ConsoleFormat, BoldWrapsCorrectly) {
  auto result = console::format::bold("hello");
  EXPECT_EQ(result, "\x1b[1mhello\x1b[0m");
}

TEST(ConsoleFormat, BlinkWrapsCorrectly) {
  auto result = console::format::blink("hello");
  EXPECT_EQ(result, "\033[33;5;7mhello\033[0m");
}

// ── now_str ─────────────────────────────────────────────────────────────────

TEST(Console, NowStrReturnsFormattedTime) {
  auto result = console::now_str();
  // Format: YYYY-MM-DD HH:MM:SS (19 chars)
  EXPECT_EQ(result.size(), 19U);
  EXPECT_EQ(result[4], '-');
  EXPECT_EQ(result[7], '-');
  EXPECT_EQ(result[10], ' ');
  EXPECT_EQ(result[13], ':');
  EXPECT_EQ(result[16], ':');
}

// ── info / warning / error / debug ──────────────────────────────────────────

TEST(Console, InfoContainsMarker) {
  testing::internal::CaptureStdout();
  console::info("test message");
  auto output = testing::internal::GetCapturedStdout();
  EXPECT_NE(output.find("[i] Info:"), std::string::npos);
  EXPECT_NE(output.find("test message"), std::string::npos);
}

TEST(Console, WarningContainsMarker) {
  testing::internal::CaptureStdout();
  console::warning("test message");
  auto output = testing::internal::GetCapturedStdout();
  EXPECT_NE(output.find("[?] Warning:"), std::string::npos);
  EXPECT_NE(output.find("test message"), std::string::npos);
}

TEST(Console, ErrorContainsMarker) {
  testing::internal::CaptureStdout();
  console::error("test message");
  auto output = testing::internal::GetCapturedStdout();
  EXPECT_NE(output.find("[!] Error:"), std::string::npos);
  EXPECT_NE(output.find("test message"), std::string::npos);
}

TEST(Console, DebugContainsMarker) {
  testing::internal::CaptureStdout();
  console::debug("test message");
  auto output = testing::internal::GetCapturedStdout();
  EXPECT_NE(output.find("[+] Debug:"), std::string::npos);
  EXPECT_NE(output.find("test message"), std::string::npos);
}

// ── read_line (redirect stdin) ──────────────────────────────────────────────

TEST(Console, ReadLineFromRedirectedStdin) {
  std::istringstream input("hello_world");
  auto old_buf = std::cin.rdbuf(input.rdbuf());
  auto result = console::read_line();
  std::cin.rdbuf(old_buf);
  EXPECT_EQ(result, "hello_world");
}

// ── pause / wait (redirect stdin to avoid blocking) ─────────────────────────

TEST(Console, PauseReadsFromStdin) {
  std::istringstream input("\n");
  auto old_buf = std::cin.rdbuf(input.rdbuf());
  console::pause();
  std::cin.rdbuf(old_buf);
}

TEST(Console, WaitPrintsAndReadsFromStdin) {
  std::istringstream input("\n");
  auto old_buf = std::cin.rdbuf(input.rdbuf());
  testing::internal::CaptureStdout();
  console::wait();
  auto output = testing::internal::GetCapturedStdout();
  std::cin.rdbuf(old_buf);
  EXPECT_NE(output.find("Press enter to continue"), std::string::npos);
}
