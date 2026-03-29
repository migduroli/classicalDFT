# Configuration file parser

## Overview

The `config::ConfigParser` class wraps Boost's `property_tree` to read
configuration parameters from external files. Four formats are supported:
**INI**, **JSON**, **XML** and **INFO**.

Delegating parameters to an external file avoids recompilation when only
inputs change, which is the standard practice in scientific computing.

## Supported formats

| Format | Extension | Boost parser             |
|--------|-----------|--------------------------|
| INI    | `.ini`    | `read_ini`               |
| JSON   | `.json`   | `read_json`              |
| XML    | `.xml`    | `read_xml`               |
| INFO   | `.info`   | `read_info`              |

Template files for each format are in [config_files/](config_files/).

## Usage

```cpp
#include <classicaldft>

int main() {
  using namespace dft;

  // Default constructor reads config.ini
  auto config = config::ConfigParser();

  auto x = config.tree().get<std::string>("default.StringValue");
  auto y = config.tree().get<double>("default.DoubleValue");

  console::info("This is x: " + x);
  console::info("This is 2*y: " + std::to_string(2 * y));

  // Explicit constructor with format selection
  auto json_config = config::ConfigParser("config.json", config::FileType::JSON);
  auto val = json_config.tree().get<double>("default.DoubleValue");
  console::info("JSON DoubleValue: " + std::to_string(val));
}
```

## Expected output

```text
2026-03-26 20:43: | [i] Info: This is x: a_text_string
2026-03-26 20:43: | [i] Info: This is 2*y: 20.000000

2026-03-26 20:43: | [i] Info: File name: config.ini
2026-03-26 20:43: | [i] Info: String value: a_text_string
2026-03-26 20:43: | [i] Info: Double value: 10.000000

2026-03-26 20:43: | [i] Info: File name: config.json
...
```

## Running

```bash
make run   # builds and runs inside Docker
```
