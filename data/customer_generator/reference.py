"""Referencia marginal empírica y puntos de entrada para futuras fuentes ponderadas."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pandas as pd

from .config import COLUMNS, ConfigurationError, parameters
from .policies import valid_number


def file_sha256(path: str | Path) -> str:
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read_csv(path: str | Path) -> pd.DataFrame:
    """Preserva ceros iniciales de identificadores y textos como 'NA'."""
    return pd.read_csv(path, dtype={"customer_id": str}, keep_default_na=False, na_values=[""])


def read_dummy_workbook(path: str | Path, sheet: str = "dataset_dummy_tarjetas_santande") -> pd.DataFrame:
    """Lee valores almacenados del XLSX del caso con stdlib, sin motor Excel extra.

    No evalúa fórmulas, macros ni enlaces. Es un lector acotado para este insumo,
    no un reemplazo general de Excel; para otras fuentes se recomienda CSV.
    """
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    rel_ns = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    with zipfile.ZipFile(path) as z:
        book = ET.fromstring(z.read("xl/workbook.xml"))
        sheets = book.findall("m:sheets/m:sheet", ns)
        chosen = next((s for s in sheets if s.get("name") == sheet), None)
        if chosen is None:
            raise ValueError(f"No se encontró la hoja {sheet}")
        rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        target = next(r.get("Target") for r in rels if r.get("Id") == chosen.get(f"{{{rel_ns}}}id"))
        member = target.lstrip("/") if target.startswith("/") else "xl/" + target
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            shared = ["".join(x.itertext()) for x in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", ns)]
        records = []
        for row in ET.fromstring(z.read(member)).findall("m:sheetData/m:row", ns):
            values: dict[int, Any] = {}
            for cell in row:
                letters = "".join(x for x in cell.get("r", "") if x.isalpha())
                index = 0
                for char in letters:
                    index = index * 26 + ord(char.upper()) - ord("A") + 1
                value = cell.find("m:v", ns)
                if cell.find("m:f", ns) is not None:
                    raise ValueError("Referencia con fórmulas: exportar valores a CSV antes de usar")
                if cell.get("t") == "s":
                    item = shared[int(value.text)]
                elif cell.get("t") == "inlineStr":
                    item = "".join(cell.find("m:is", ns).itertext())
                else:
                    item = value.text if value is not None else None
                values[index - 1] = item
            records.append(values)
    if not records:
        raise ValueError("Hoja de referencia vacía")
    width = max(records[0]) + 1
    headers = [records[0].get(i) for i in range(width)]
    return pd.DataFrame([[r.get(i) for i in range(width)] for r in records[1:]], columns=headers)


@dataclass
class ReferenceProfile:
    """Soportes completos, no aproximaciones normales ni intervalos min/max."""

    samples: dict[str, np.ndarray]
    categories: dict[str, dict[str, int]]
    metadata: dict[str, Any]

    @classmethod
    def from_path(cls, path: str | Path, config: dict[str, Any]) -> ReferenceProfile:
        p = Path(path)
        frame = read_dummy_workbook(p) if p.suffix.lower() == ".xlsx" else read_csv(p)
        return cls.from_frame(frame, config, {"path": str(p), "sha256": file_sha256(p),
                                            "sheet": "dataset_dummy_tarjetas_santande" if p.suffix.lower() == ".xlsx" else None})

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, config: dict[str, Any],
                   provenance: dict[str, Any] | None = None) -> ReferenceProfile:
        if frame.empty:
            raise ValueError("La referencia está vacía")
        if not set(frame.columns) & (set(COLUMNS) - {"customer_id"}):
            raise ValueError("La referencia no contiene variables útiles del esquema")
        domains = parameters(config, "DOM-01")["numeric"]
        samples, summaries = {}, {}
        for field, spec in domains.items():
            if field not in frame:
                continue
            mask = frame[field].map(lambda x: valid_number(x, spec))
            values = pd.to_numeric(frame.loc[mask, field]).to_numpy(dtype=float)
            if not len(values):
                summaries[field] = {"valid": 0, "excluded": len(frame)}
                continue
            # Repeticiones retienen las frecuencias, incluidos ceros y extremos.
            samples[field] = values
            summaries[field] = {"valid": len(values), "excluded": int((~mask).sum()),
                                "zeros": int((values == 0).sum()),
                                "quantiles": dict(zip(["min", "p01", "p25", "p50", "p75", "p99", "max"],
                                                      np.quantile(values, [0, .01, .25, .5, .75, .99, 1]).tolist()))}
        categories = {}
        for field in ["segmento", "tarjeta_santander"]:
            if field in frame:
                categories[field] = {str(k): int(v) for k, v in frame[field].dropna().value_counts(sort=False).items()}
        return cls(samples, categories, {
            "kind": "referencia_aportada_no_validada_como_poblacion_real", "rows": len(frame),
            "provenance": provenance or {"source": "DataFrame proporcionado; sin archivo"},
            "numeric_distributions": summaries, "categorical_frequencies": categories,
            "used_for_sampling": [f for f in ["edad", "ingreso_mensual", "antiguedad_meses"] if f in samples],
            "dependent_fields": "Se resumen todas las marginales; consumo, digital y experiencia se generan con políticas condicionales DEMO",
            "future_adapters": {
                "ENIGH": "Entrada futura explícita: ingreso INDIVIDUAL mensual, persona, concepto de ingreso y factor de expansión; no sustituir por ingreso del hogar",
                "ENIF": "Entrada futura separada: perfiles financieros de personas, año, universos, definiciones y factores de expansión; no es una distribución de ingresos",
                "weighted_csv": "Ver sample_weighted_individual_income: persona_id, ingreso_mensual_individual, factor_expansion; requiere metadatos de unidad, periodo, concepto y fuente"
            }
        })

    def priors(self, field: str, names: list[str]) -> dict[str, float] | None:
        counts = self.categories.get(field, {})
        known = {name: float(counts.get(name, 0)) for name in names}
        if counts and not any(known.values()):
            raise ConfigurationError(f"Referencia de {field}: ninguna categoría pertenece al catálogo; revisar nombres")
        return known if any(known.values()) else None


def sample_weighted_individual_income(frame: pd.DataFrame, rng: np.random.Generator,
                                      size: int, metadata: dict[str, str]) -> np.ndarray:
    """Contrato futuro explícito; no descarga, interpreta ni integra ENIGH automáticamente."""
    from .config import normalize_weights

    required = {"persona_id", "ingreso_mensual_individual", "factor_expansion"}
    if not required <= set(frame.columns):
        raise ValueError(f"Faltan columnas individuales: {sorted(required - set(frame.columns))}")
    if metadata.get("unit") != "persona" or metadata.get("period") != "mensual" or not metadata.get("source") or not metadata.get("income_concept"):
        raise ValueError("Se requieren unidad persona, periodo mensual, concepto y fuente identificada")
    if frame["persona_id"].isna().any() or frame["persona_id"].duplicated().any():
        raise ValueError("Se requiere una observación por persona antes de ponderar")
    values = pd.to_numeric(frame["ingreso_mensual_individual"], errors="raise").to_numpy(dtype=float)
    weights = pd.to_numeric(frame["factor_expansion"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Ingresos individuales inválidos")
    if not isinstance(size, int) or size < 0:
        raise ConfigurationError("size debe ser entero no negativo")
    return rng.choice(values, size=size, p=normalize_weights(weights))
