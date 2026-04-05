# Config: configuration file parsing

## Purpose

This example demonstrates the configuration file parser that reads INI and
JSON formats into a uniform key-value store. All physics docs in this
project use `config.ini` files to specify model parameters (temperature,
grid spacing, potential cutoffs, etc.), so the `config` module is the entry
point for every simulation workflow.

## Supported formats

### INI format

An INI file is organised into sections delimited by `[section]` headers.
Within each section, key-value pairs are written as `key = value`:

```ini
[default]
StringValue = hello
DoubleValue = 3.14
```

The parser converts this into a flat key-value map using dotted keys:
`"default.StringValue"` → `"hello"`, `"default.DoubleValue"` → `"3.14"`.

### JSON format

A JSON file uses nested objects to achieve the same logical structure:

```json
{
  "default": {
    "StringValue": "hello",
    "DoubleValue": 3.14
  }
}
```

The same dotted-key access pattern (`"default.DoubleValue"`) works for both
formats, so the physics code is agnostic to the file format.

---

## Step-by-step code walkthrough

```cpp
auto data = config::parse_config("config.ini", config::FileType::INI);
auto string_value = config::get<std::string>(data, "default.StringValue");
auto double_value = config::get<double>(data, "default.DoubleValue");
```

1. `config::parse_config(path, type)` reads the file and returns a
   `ConfigData` map (internally `std::unordered_map<std::string, std::string>`).
2. `config::get<T>(data, key)` extracts a value by dotted key and converts
   it to the requested type (`std::string`, `double`, `int`, `bool`, etc.)
   using `std::from_chars` or `std::stod` as appropriate.

The example parses both `config.ini` and `config.json` and prints the
extracted values to confirm they agree.

## Implementation notes

- The INI parser strips comments (lines starting with `#` or `;`).
- The JSON parser uses a lightweight recursive-descent parser (no external
  JSON dependency).
- Keys are case-sensitive; whitespace around `=` and values is trimmed.
- Missing keys throw `std::runtime_error` with a descriptive message.

## Build and run

```bash
make run-local
```

## Config files

- `config.ini` — INI format (section headers, `key = value` pairs)
- `config.json` — JSON format (nested objects, same logical structure)
