"""Funciones de políticas explícitas y validaciones obligatorias reutilizables."""

from __future__ import annotations

import math
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd

from .config import COLUMNS, ConfigurationError, normalize_weights, parameters

ISSUE_COLUMNS = ["row_key", "field", "policy_id", "reason", "kind"]


def missing(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip()) or (
        pd.api.types.is_scalar(value) and bool(pd.isna(value))
    )


def numeric(value: Any) -> float | None:
    if isinstance(value, (bool, np.bool_)) or missing(value):
        return None
    if not isinstance(value, (str, Real, np.integer, np.floating)):
        return None
    try:
        result = float(value)
    except (ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def valid_number(value: Any, spec: dict[str, Any]) -> bool:
    x = numeric(value)
    return x is not None and spec["min"] <= x <= spec["max"] and (
        not spec["integer"] or x.is_integer()
    )


def eligible_products(age: float, income: float, config: dict[str, Any]) -> list[dict[str, Any]]:
    return [p for p in parameters(config, "TAR-01")["products"]
            if age >= p["min_age"] and income >= p["min_income"]]


def card_probabilities(age: float, income: float, preferences: dict[str, float],
                       config: dict[str, Any], priors: dict[str, float] | None = None
                       ) -> tuple[list[str], np.ndarray]:
    """El filtro obligatorio precede a la afinidad y a la normalización."""
    options = eligible_products(age, income, config)
    if not options:
        raise ConfigurationError(f"TAR-01: ningún producto elegible para edad={age}, ingreso={income}")
    p = parameters(config, "TAR-01")
    weights = [(priors.get(x["name"], 0) if priors is not None else x["weight"])
               * (1 + p["affinity_strength"] * preferences[x["benefit"]]) for x in options]
    return [x["name"] for x in options], normalize_weights(weights)


def segment_probabilities(income: float, tenure: float, config: dict[str, Any],
                          priors: dict[str, float] | None = None
                          ) -> tuple[list[str], np.ndarray, int]:
    p = parameters(config, "SEG-01")
    names = parameters(config, "DOM-01")["segments"]
    tier = int(np.searchsorted(p["income_thresholds"], income, side="right"))
    base = np.array([priors.get(n, 0) for n in names] if priors is not None else p["prior_weights"])
    relation = 1 + np.array(p["tenure_boost"]) * tenure / p["tenure_scale"]
    return names, normalize_weights(base * p["tier_factors"][tier] * relation), tier


def create_profile(rng: np.random.Generator, row: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Latentes compartidas; en reparación se anclan a indicadores válidos disponibles."""
    p = parameters(config, "AUX-01")
    result: dict[str, Any] = {
        name: float(rng.beta(*p[f"{name}_beta"]))
        for name in ["digital", "activity", "experience", "payment"]
    }
    domains = parameters(config, "DOM-01")["numeric"]
    anchors = {"digital": [("ecommerce_pct", 100, 0)],
               "experience": [("satisfaccion", 4, 1), ("nps", 10, 0)],
               "payment": [("pagos_puntuales_pct", 100, 0)]}
    for latent, indicators in anchors.items():
        observed = [(float(row[name]) - offset) / scale for name, scale, offset in indicators
                    if valid_number(row[name], domains[name])]
        if observed:
            weight = p["observed_anchor_weight"]
            result[latent] = weight * float(np.mean(observed)) + (1 - weight) * result[latent]
    result["preferences"] = dict(zip(p["preferences"], rng.dirichlet(p["preference_alpha"]).tolist()))
    return result


def consumption(rng: np.random.Generator, income: float, card: str, profile: dict[str, Any],
                config: dict[str, Any]) -> tuple[int, float, dict[str, Any]]:
    """Decisión de actividad, frecuencia y ticket medio dentro de una misma rama."""
    p = parameters(config, "CON-01")
    product = next(x for x in parameters(config, "TAR-01")["products"] if x["name"] == card)
    inactive_p = p["inactive_probability"] * (1 - p["activity_protection"] * profile["activity"])
    inactive = bool(rng.random() < inactive_p)
    details = {"branch": "inactivo" if inactive else "activo", "inactive_probability": inactive_p}
    if inactive:
        return 0, 0.0, details
    mean = (p["frequency_base"] + p["frequency_income_weight"] * math.log1p(income / p["frequency_income_scale"])
            + p["frequency_activity_weight"] * profile["activity"]) * product["consumption_factor"]
    maximum = parameters(config, "DOM-01")["numeric"]["transacciones_mes"]["max"]
    # Saturación explícita y finita; nunca rechazo iterativo.
    count = min(maximum, max(1, int(rng.poisson(min(mean, maximum)))))
    ticket = average_ticket(rng, income, card, profile, config)
    details.update({"frequency_mean": mean, "average_ticket": ticket})
    return count, ticket, details


def average_ticket(rng: np.random.Generator, income: float, card: str,
                   profile: dict[str, Any], config: dict[str, Any]) -> float:
    p = parameters(config, "CON-01")
    product = next(x for x in parameters(config, "TAR-01")["products"] if x["name"] == card)
    median = (p["ticket_base"] + p["ticket_income_weight"] * income) * (
        1 + p["ticket_activity_weight"] * profile["activity"]
    ) * product["consumption_factor"]
    maximum = parameters(config, "DOM-01")["numeric"]["gasto_mensual"]["max"]
    log_ticket = float(rng.normal(math.log(median), p["ticket_sigma"]))
    return round(math.exp(min(log_ticket, math.log(maximum))), 2)


def digital_value(field: str, rng: np.random.Generator, profile: dict[str, Any],
                  config: dict[str, Any]) -> float | int:
    p = parameters(config, "DIG-01")
    if field == "ecommerce_pct":
        return round(100 * float(np.clip(profile["digital"] + rng.normal(0, p["ecommerce_noise"]), 0, 1)), 2)
    if rng.random() < p["app_zero_probability"]:
        return 0
    cap = parameters(config, "DOM-01")["numeric"][field]["max"]
    return min(cap, int(rng.poisson(min(cap, p["app_base"] + p["app_digital_weight"] * profile["digital"]))))


def experience_value(field: str, rng: np.random.Generator, profile: dict[str, Any],
                     card: str, config: dict[str, Any]) -> float | int:
    p = parameters(config, "EXP-01")
    if field == "satisfaccion":
        return round(1 + 4 * float(np.clip(profile["experience"] + rng.normal(0, p["satisfaction_noise"]), 0, 1)), 2)
    if field == "nps":
        return int(round(10 * float(np.clip(profile["experience"] + rng.normal(0, p["nps_noise"]), 0, 1))))
    if field == "pagos_puntuales_pct":
        return round(100 * float(np.clip(profile["payment"] + rng.normal(0, p["payment_noise"]), 0, 1)), 2)
    product = next(x for x in parameters(config, "TAR-01")["products"] if x["name"] == card)
    intensity = p["promo_intercept"] + p["promo_digital_weight"] * profile["digital"]
    intensity += p["promo_benefit_weight"] * profile["preferences"][product["benefit"]]
    return round(100 * float(np.clip(intensity + rng.normal(0, p["promo_noise"]), 0, 1)), 2)


def usage_value(count: int, rng: np.random.Generator, profile: dict[str, Any], config: dict[str, Any]) -> float:
    p = parameters(config, "USO-01")
    intensity = (1 - math.exp(-count / p["frequency_scale"])) + p["activity_weight"] * profile["activity"]
    return round(100 * float(np.clip(intensity + rng.normal(0, p["noise"]), 0, 1)), 2)


def risk_value(row: dict[str, Any], rng: np.random.Generator, config: dict[str, Any]) -> float:
    p = parameters(config, "RIE-01")
    score = p["intercept"] + p["satisfaction_weight"] * (float(row["satisfaccion"]) - 1) / 4
    score += p["nps_weight"] * float(row["nps"]) / 10
    score += p["payment_weight"] * float(row["pagos_puntuales_pct"]) / 100 + rng.normal(0, p["noise"])
    # Logística estable en ambos signos.
    e = math.exp(-abs(float(score)))
    return round(1 / (1 + e) if score >= 0 else e / (1 + e), 6)


def validate_frame(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Una pasada, sin exigir que relaciones probabilísticas se cumplan individualmente."""
    d = parameters(config, "DOM-01")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for i, record in enumerate(frame.reindex(columns=COLUMNS).to_dict("records")):
        key = f"row-{i:08d}"

        def add(field: str, pid: str, reason: str, kind: str = "infraccion") -> None:
            rows.append(dict(row_key=key, field=field, policy_id=pid, reason=reason, kind=kind))

        for field, value in record.items():
            if missing(value):
                add(field, "DOM-01", "faltante", "faltante")
            elif field in d["numeric"] and not valid_number(value, d["numeric"][field]):
                add(field, "DOM-01", "fuera de dominio o tipo no numérico/entero")
        ident = record["customer_id"]
        if not missing(ident):
            if not isinstance(ident, str) or not ident.strip():
                add("customer_id", "DOM-01", "identificador debe ser texto no vacío")
            elif ident in seen:
                add("customer_id", "DOM-01", "identificador duplicado")
            else:
                seen.add(ident)
        if not missing(record["segmento"]) and record["segmento"] not in d["segments"]:
            add("segmento", "DOM-01", "segmento desconocido")
        card_names = [p["name"] for p in parameters(config, "TAR-01")["products"]]
        if not missing(record["tarjeta_santander"]) and record["tarjeta_santander"] not in card_names:
            add("tarjeta_santander", "DOM-01", "producto desconocido")
        if valid_number(record["edad"], d["numeric"]["edad"]):
            age = float(record["edad"])
            tenure = numeric(record["antiguedad_meses"])
            start = parameters(config, "CRO-01")["relationship_start_age"]
            if tenure is not None and tenure > (age - start) * 12:
                add("antiguedad_meses", "CRO-01", "antigüedad anterior al inicio permitido de relación")
            if valid_number(record["ingreso_mensual"], d["numeric"]["ingreso_mensual"]):
                allowed = [p["name"] for p in eligible_products(age, float(record["ingreso_mensual"]), config)]
                if record["tarjeta_santander"] in card_names and record["tarjeta_santander"] not in allowed:
                    add("tarjeta_santander", "TAR-01", "incumple mínimo DEMO de edad/ingreso")
        if numeric(record["transacciones_mes"]) == 0:
            for field in parameters(config, "CON-01")["zero_when_inactive"]:
                value = numeric(record[field])
                if value is not None and value != 0:
                    add(field, "CON-01", "cero compras exige cero según convención del periodo")
    return pd.DataFrame(rows, columns=ISSUE_COLUMNS)
