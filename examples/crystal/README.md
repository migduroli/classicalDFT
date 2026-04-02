# Crystal example

Builds BCC, FCC, and HCP lattices with various orientations using
`build_lattice()`, scales positions with `scaled_positions()`, and exports
to XYZ/CSV via `export_lattice()`.

With matplotlib enabled, produces xy-projection plots of replicated lattices.

## Build and run

```bash
make run
```

## Output

- `exports/fcc_4x4x4.xyz` — XYZ lattice file
- `exports/fcc_4x4x4.csv` — CSV lattice file
- `exports/fcc_001.png` — FCC [001] projection (matplotlib)
- `exports/bcc_110.png` — BCC [110] projection (matplotlib)
- `exports/hcp_001.png` — HCP [001] projection (matplotlib)
