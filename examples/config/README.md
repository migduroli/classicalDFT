# Config: configuration file parsing

Demonstrates the configuration file parser that reads INI and JSON formats
into a uniform key-value store.

## What this example does

Parses `config.ini` (INI format) and `config.json` (JSON format) using
`config::parse_config()`, then extracts typed values with `config::get<T>()`.
Both formats use dotted keys (`"section.key"`) for access.

## Key API functions used

| Function | Purpose |
|----------|---------|
| `config::parse_config()` | parse INI or JSON file |
| `config::get<T>()` | extract typed value by dotted key |

## Build and run

```bash
make run
```

## Config files

- `config.ini` — INI format (section headers, key=value pairs)
- `config.json` — JSON format (nested objects)
