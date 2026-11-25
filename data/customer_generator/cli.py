"""Interfaz mínima para generar, reparar y preparar una referencia local."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .config import load_config
from .generator import CustomerGenerator
from .reference import read_dummy_workbook


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Clientes sintéticos académicos; políticas de demostración, sin internet")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in [("generate", "Generar clientes nuevos"), ("repair", "Reparar una copia de un CSV")]:
        command = sub.add_parser(name, help=help_text)
        command.add_argument("--config", type=Path, help="JSON; predeterminado: data/config.demo.json")
        command.add_argument("--seed", type=int, default=42)
        command.add_argument("--reference", type=Path, help="CSV de referencia independiente del CSV a reparar; también XLSX del caso")
        command.add_argument("--output", type=Path, required=True, help="Carpeta NUEVA para los seis artefactos")
        if name == "generate":
            command.add_argument("--rows", type=int, default=10000)
        else:
            command.add_argument("--input", type=Path, required=True, help="CSV a reparar; nunca se modifica")
    extract = sub.add_parser("extract-reference", help="Exportar valores de la hoja dummy a un CSV nuevo, sin dependencias Excel")
    extract.add_argument("--input", type=Path, required=True)
    extract.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "extract-reference":
            frame = read_dummy_workbook(args.input)
            # Modo exclusivo: no sobrescribe, incluso si entrada y salida coinciden.
            with args.output.open("x", encoding="utf-8", newline="") as f:
                frame.to_csv(f, index=False)
            print(f"Referencia local: {len(frame)} filas -> {args.output}")
            return 0
        config = load_config(args.config)
        if args.output.exists():
            raise ValueError("La carpeta de salida ya existe: elija una carpeta nueva")
        if args.command == "repair" and args.reference and args.reference.resolve() == args.input.resolve():
            raise ValueError("El CSV a reparar y la referencia deben ser entradas distintas; omita --reference o proporcione otra fuente")
        generator = CustomerGenerator(config, args.seed, args.reference)
        result = generator.generate(args.rows) if args.command == "generate" else generator.repair(args.input)
        protected = tuple(p for p in [args.reference, getattr(args, "input", None), args.config] if p)
        path = result.export(args.output, protected)
        print(f"{len(result.customers)} clientes; {len(result.audit)} modificaciones; "
              f"{result.report['violations_after']['total']} infracciones finales -> {path}")
        return 0
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
