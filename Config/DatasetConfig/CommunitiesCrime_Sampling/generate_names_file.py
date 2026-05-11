"""Generate the UCI Communities and Crime ``.names`` file from ucimlrepo.

Phase A.1 + Phase E post-shakedown follow-up. The original UCI
distribution shipped a ``communities and crime unnormalized.names``
ARFF-style schema file alongside the data. UCI 2.0 stopped shipping it
— the zip at ``archive.ics.uci.edu/static/public/211/...`` contains
only the data file. The canonical metadata source is now the
:mod:`ucimlrepo` Python package, which exposes the variable list
(name + type + role) via :func:`ucimlrepo.fetch_ucirepo`.

This script writes a ``.names`` file in the ARFF format our loader
(:func:`Config.DatasetConfig.CommunitiesCrime_Sampling.commCrimeDatasetLoad.parse_names_file`)
expects:

    @relation communities_and_crime_unnormalized
    @attribute communityname string
    @attribute State string
    @attribute countyCode numeric
    ...

Usage:
    python -m Config.DatasetConfig.CommunitiesCrime_Sampling.generate_names_file \\
        [--output PATH]

Default output: ``$HOME/datasets/CommunitiesCrime/communities_and_crime_unnormalized.names``.

Requires ``pip install ucimlrepo`` (one-time; not a runtime dependency
of the fusion-centers pipeline itself).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from Config.DatasetConfig.CommunitiesCrime_Sampling.commCrimeDatasetLoad import (
    DEFAULT_TARGET_DIR, RAW_NAMES_FILENAME,
)


def _arff_type_for(uci_type: str) -> str:
    """Map ucimlrepo type strings to ARFF type tokens that
    :func:`parse_names_file` will accept (our parser only cares about
    the attribute name, but a well-formed second token is good practice)."""
    if uci_type in ("Categorical",):
        return "string"
    return "numeric"  # Integer, Continuous, Binary, etc.


def generate(output_path: Path) -> Path:
    """Write the ``.names`` file. Returns the resolved path."""
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as exc:
        raise SystemExit(
            "ucimlrepo is required to regenerate the .names file. "
            "Install with: pip install ucimlrepo\n"
            f"(import error: {exc})"
        )

    print("Fetching UCI dataset 211 metadata via ucimlrepo...")
    ds = fetch_ucirepo(id=211)
    variables = ds.variables
    print(f"  -> {len(variables)} variables")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = ["@relation communities_and_crime_unnormalized", ""]
    for _, row in variables.iterrows():
        name = row["name"]
        arff_type = _arff_type_for(str(row.get("type", "")))
        lines.append(f"@attribute {name} {arff_type}")
    lines.append("")
    lines.append("@data")
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"=== Wrote {output_path} ({len(variables)} attributes) ===")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", default=None,
        help=f"Output path for the .names file "
             f"(default: {DEFAULT_TARGET_DIR / RAW_NAMES_FILENAME})",
    )
    args = parser.parse_args()
    output_path = Path(args.output) if args.output else DEFAULT_TARGET_DIR / RAW_NAMES_FILENAME
    generate(output_path)


if __name__ == "__main__":
    main()
