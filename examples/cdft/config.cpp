// ── Config parser example (modern cdft API) ─────────────────────────────────
//
// Demonstrates:
//   - INI and JSON configuration parsing
//   - Typed value extraction with dotted paths
//   - Access to the underlying nlohmann::json data

#include <cdft.hpp>

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>

int main() {
  using namespace cdft::io;

  // ── Create sample INI file ──────────────────────────────────────────────

  {
    std::ofstream ini("sample.ini");
    ini << "[simulation]\n"
        << "kT = 1.2\n"
        << "density = 0.5\n"
        << "species = argon\n"
        << "\n"
        << "[grid]\n"
        << "spacing = 0.01\n"
        << "nx = 128\n"
        << "ny = 128\n"
        << "nz = 128\n";
  }

  // ── Create sample JSON file ─────────────────────────────────────────────

  {
    std::ofstream json("sample.json");
    json << R"({
  "simulation": {
    "kT": 1.2,
    "density": 0.5,
    "species": "argon"
  },
  "grid": {
    "spacing": 0.01,
    "nx": 128,
    "ny": 128,
    "nz": 128
  }
})";
  }

  std::cout << std::fixed << std::setprecision(4);

  // ── Parse INI file ──────────────────────────────────────────────────────

  std::cout << "INI configuration\n";
  std::cout << std::string(40, '-') << "\n";

  auto ini_config = ConfigParser("sample.ini", ConfigFormat::INI);

  auto kT = ini_config.get<double>("simulation.kT");
  auto rho = ini_config.get<double>("simulation.density");
  auto species = ini_config.get<std::string>("simulation.species");
  auto nx = ini_config.get<int>("grid.nx");
  auto dx = ini_config.get<double>("grid.spacing");

  std::cout << "  kT      = " << kT << "\n";
  std::cout << "  density = " << rho << "\n";
  std::cout << "  species = " << species << "\n";
  std::cout << "  grid    = " << nx << "^3 at dx = " << dx << "\n";

  // ── Parse JSON file ─────────────────────────────────────────────────────

  std::cout << "\nJSON configuration\n";
  std::cout << std::string(40, '-') << "\n";

  auto json_config = ConfigParser("sample.json", ConfigFormat::JSON);

  std::cout << "  kT      = " << json_config.get<double>("simulation.kT") << "\n";
  std::cout << "  species = " << json_config.get<std::string>("simulation.species") << "\n";
  std::cout << "  nx      = " << json_config.get<int>("grid.nx") << "\n";

  // ── Raw JSON access ─────────────────────────────────────────────────────

  std::cout << "\nRaw JSON tree\n";
  std::cout << std::string(40, '-') << "\n";
  std::cout << json_config.data().dump(2) << "\n";

  // ── Cleanup ─────────────────────────────────────────────────────────────

  std::filesystem::remove("sample.ini");
  std::filesystem::remove("sample.json");

  return 0;
}
