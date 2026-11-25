"""Carga y validación estricta de parámetros, sin ejecutar expresiones."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

COLUMNS = (
    "customer_id", "edad", "segmento", "antiguedad_meses", "ingreso_mensual",
    "tarjeta_santander", "uso_tarjeta_pct", "transacciones_mes", "gasto_mensual",
    "ecommerce_pct", "pagos_puntuales_pct", "app_logins_mes", "nps", "satisfaccion",
    "abandono_riesgo", "activacion_promos_pct",
)
ORDER = (
    "customer_id", "edad", "ingreso_mensual", "antiguedad_meses", "segmento",
    "tarjeta_santander", "transacciones_mes", "gasto_mensual", "ecommerce_pct",
    "app_logins_mes", "satisfaccion", "nps", "pagos_puntuales_pct",
    "activacion_promos_pct", "uso_tarjeta_pct", "abandono_riesgo",
)
POLICY_IDS = (
    "DOM-01", "INI-01", "CRO-01", "AUX-01", "TAR-01", "SEG-01", "CON-01",
    "DIG-01", "EXP-01", "USO-01", "RIE-01",
)
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config.demo.json"


class ConfigurationError(ValueError):
    """Parámetros incompletos, incompatibles o sin opciones de probabilidad."""


def normalize_weights(weights: list[float] | np.ndarray) -> np.ndarray:
    """Normaliza pesos finitos no negativos; todos cero es un error."""
    try:
        w = np.asarray(weights, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError("Los pesos deben ser números") from exc
    if w.ndim != 1 or not w.size or not np.isfinite(w).all() or (w < 0).any():
        raise ConfigurationError("Pesos vacíos, negativos o no finitos")
    scale = float(w.max())
    if scale == 0:
        raise ConfigurationError("Pesos todos cero: no hay distribución de probabilidad")
    w = w / scale
    return w / w.sum()


def parameters(config: dict[str, Any], policy_id: str) -> dict[str, Any]:
    return config["policies"][policy_id]["parameters"]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    with Path(path or DEFAULT_CONFIG).open(encoding="utf-8") as f:
        config = json.load(f)
    validate_config(config)
    return config


def checked_config(config: dict[str, Any] | None) -> dict[str, Any]:
    result = copy.deepcopy(config) if config is not None else load_config()
    validate_config(result)
    return result


def validate_config(config: dict[str, Any]) -> None:
    """Verifica dominios, coberturas de elegibilidad, pesos y grafo antes del sorteo."""
    try:
        _validate(config)
    except (KeyError, TypeError, IndexError, OverflowError) as exc:
        raise ConfigurationError(f"Configuración incompleta o mal formada: {exc}") from exc


def _validate(c: dict[str, Any]) -> None:
    def require(test: bool, reason: str) -> None:
        if not test:
            raise ConfigurationError(reason)

    def number(x: Any, name: str, low: float | None = None,
               high: float | None = None, positive: bool = False) -> None:
        require(isinstance(x, (int, float)) and not isinstance(x, bool)
                and math.isfinite(x), f"{name}: se requiere número finito")
        require(low is None or x >= low, f"{name}: menor que {low}")
        require(high is None or x <= high, f"{name}: mayor que {high}")
        require(not positive or x > 0, f"{name}: debe ser positivo")

    require(bool(c["version"]) and bool(c["scenario"]), "Falta versión o escenario")
    require(set(c["policies"]) == set(POLICY_IDS), "IDs de política desconocidos o faltantes")
    for pid in POLICY_IDS:
        p = c["policies"][pid]
        require(bool(p["description"]) and bool(p["origin"]), f"{pid}: falta procedencia")
        require(p["type"] in {"obligatorio", "probabilistico"}, f"{pid}: tipo inválido")
        require(bool(p["variables"]) and set(p["variables"]) <= set(COLUMNS),
                f"{pid}: variables desconocidas")
    d, ini, cro, aux, card, seg, con, dig, exp, uso, risk = [parameters(c, x) for x in POLICY_IDS]
    numeric = d["numeric"]
    require(set(numeric) == set(COLUMNS) - {"customer_id", "segmento", "tarjeta_santander"},
            "Dominios numéricos incompletos")
    for name, spec in numeric.items():
        number(spec["min"], name)
        number(spec["max"], name)
        require(spec["min"] <= spec["max"], f"{name}: límites invertidos")
        require(isinstance(spec["integer"], bool), f"{name}: integer debe ser booleano")
        require(not spec["integer"] or (float(spec["min"]).is_integer()
                and float(spec["max"]).is_integer()), f"{name}: límites no enteros")
    required_ranges = {x: (0, 100) for x in COLUMNS if x.endswith("_pct")}
    required_ranges.update({"nps": (0, 10), "satisfaccion": (1, 5), "abandono_riesgo": (0, 1)})
    for name, bounds in required_ranges.items():
        require((numeric[name]["min"], numeric[name]["max"]) == bounds,
                f"{name}: escala contractual incompatible")
    for name in ["edad", "antiguedad_meses", "transacciones_mes", "app_logins_mes", "nps"]:
        require(numeric[name]["integer"], f"{name}: esta versión usa enteros")
    for name, spec in numeric.items():
        require(spec["min"] >= 0, f"{name}: no admite negativos")
        if name not in {"edad", "satisfaccion"}:
            require(spec["min"] == 0, f"{name}: esta versión requiere mínimo cero")
    require(numeric["transacciones_mes"]["max"] >= 1, "No se pueden generar compras activas")
    require(numeric["ingreso_mensual"]["max"] > 0 and numeric["gasto_mensual"]["max"] > 0,
            "Los máximos monetarios deben ser positivos")
    require(isinstance(d["id_prefix"], str) and bool(d["id_prefix"].strip()), "Prefijo de ID vacío")
    require(len(d["segments"]) == len(set(d["segments"])) == 3
            and set(d["segments"]) == {"Mass", "Select", "Premier"}, "Catálogo de segmentos inválido")
    number(cro["relationship_start_age"], "edad de inicio", 0, numeric["edad"]["min"])
    require(len(cro["beta"]) == 2, "Beta de antigüedad necesita dos parámetros")
    for x in cro["beta"]:
        number(x, "beta antigüedad", positive=True)
    normalize_weights(ini["age_weights"])
    require(len(ini["age_bins"]) == len(ini["age_weights"]), "Pesos de edad incompatibles")
    for low, high in ini["age_bins"]:
        number(low, "edad inicial", numeric["edad"]["min"], numeric["edad"]["max"])
        number(high, "edad inicial", low, numeric["edad"]["max"])
        require(float(low).is_integer() and float(high).is_integer(), "Edades deben ser enteras")
    normalize_weights(ini["income_weights"])
    require(len(ini["income_medians"]) == len(ini["income_sigmas"])
            == len(ini["income_weights"]), "Mezcla de ingresos incompatible")
    for x in ini["income_medians"]:
        number(x, "mediana ingreso", positive=True)
    for x in ini["income_sigmas"]:
        number(x, "sigma ingreso", 0)
    number(ini["income_zero_probability"], "masa cero ingreso", 0, 1)
    require(len(aux["preferences"]) == len(set(aux["preferences"]))
            == len(aux["preference_alpha"]), "Preferencias incompatibles")
    for key in ["digital_beta", "activity_beta", "experience_beta", "payment_beta"]:
        require(len(aux[key]) == 2, f"{key}: faltan parámetros")
        for x in aux[key]:
            number(x, key, positive=True)
    for x in aux["preference_alpha"]:
        number(x, "alpha preferencias", positive=True)
    number(aux["observed_anchor_weight"], "anclaje observado", 0, 1)
    products = card["products"]
    require(bool(products) and len({p["name"] for p in products}) == len(products),
            "Catálogo de productos vacío o duplicado")
    number(card["affinity_strength"], "fuerza afinidad", 0)
    for p in products:
        require(isinstance(p["name"], str) and bool(p["name"].strip()), "Nombre de producto vacío")
        number(p["min_income"], p["name"], 0, numeric["ingreso_mensual"]["max"])
        number(p["min_age"], p["name"], 0, numeric["edad"]["max"])
        number(p["weight"], p["name"], 0)
        number(p["consumption_factor"], p["name"], positive=True)
        require(p["benefit"] in aux["preferences"], f"Beneficio desconocido: {p['benefit']}")
    # Con mínimos monotónicos basta comprobar el extremo inferior del dominio.
    fallback = [p["weight"] for p in products if p["min_income"] <= numeric["ingreso_mensual"]["min"]
                and p["min_age"] <= numeric["edad"]["min"]]
    require(bool(fallback), "Sin producto elegible en el extremo inferior de edad/ingreso")
    normalize_weights(fallback)
    normalize_weights([p["weight"] for p in products])
    normalize_weights(seg["prior_weights"])
    thresholds = seg["income_thresholds"]
    for x in thresholds:
        number(x, "umbral de segmento", 0)
    require(thresholds == sorted(set(thresholds)), "Umbrales de segmento desordenados o duplicados")
    require(len(seg["tier_factors"]) == len(thresholds) + 1, "Ramas de segmento incompletas")
    for factors in seg["tier_factors"]:
        require(len(factors) == len(seg["prior_weights"]) == len(d["segments"]), "Pesos de segmento incompatibles")
        normalize_weights(np.asarray(factors) * np.asarray(seg["prior_weights"]))
    require(len(seg["tenure_boost"]) == len(d["segments"]), "Bonos de relación incompatibles")
    for x in seg["tenure_boost"]:
        number(x, "bono relación", 0)
    number(seg["tenure_scale"], "escala relación", positive=True)
    probability_keys = [(con, "inactive_probability"), (con, "activity_protection"),
                        (dig, "app_zero_probability")]
    for obj, key in probability_keys:
        number(obj[key], key, 0, 1)
    for obj, keys in [
        (con, ["frequency_base", "frequency_income_weight", "frequency_activity_weight", "ticket_base", "ticket_income_weight", "ticket_activity_weight", "ticket_sigma"]),
        (dig, ["ecommerce_noise", "app_base", "app_digital_weight"]),
        (exp, ["satisfaction_noise", "nps_noise", "payment_noise", "promo_intercept", "promo_digital_weight", "promo_benefit_weight", "promo_noise"]),
        (uso, ["activity_weight", "noise"]), (risk, ["noise"]),
    ]:
        for key in keys:
            number(obj[key], key, 0)
    for obj, key in [(con, "frequency_income_scale"), (uso, "frequency_scale")]:
        number(obj[key], key, positive=True)
    require(con["ticket_base"] > 0 and con["frequency_base"] > 0, "Consumo requiere base positiva")
    require(set(con["zero_when_inactive"]) == {"gasto_mensual", "ecommerce_pct", "uso_tarjeta_pct"},
            "Convención de inactividad incompatible con esta versión")
    for key in ["intercept", "satisfaction_weight", "nps_weight", "payment_weight"]:
        number(risk[key], key)
    deps = c["dependencies"]
    require(set(deps) == set(COLUMNS), "Grafo de dependencias incompleto")
    for child, parents in deps.items():
        require(isinstance(parents, list) and set(parents) <= set(COLUMNS), "Dependencia desconocida")
        require(all(ORDER.index(parent) < ORDER.index(child) for parent in parents),
                f"Ciclo u orden incompatible de dependencias en {child}")
    essential = {
        "antiguedad_meses": ["edad"], "tarjeta_santander": ["edad", "ingreso_mensual"],
        "segmento": ["ingreso_mensual", "antiguedad_meses"],
        "transacciones_mes": ["ingreso_mensual", "tarjeta_santander"],
        "gasto_mensual": ["transacciones_mes", "ingreso_mensual", "tarjeta_santander"],
        "ecommerce_pct": ["transacciones_mes"], "uso_tarjeta_pct": ["transacciones_mes"],
        "activacion_promos_pct": ["tarjeta_santander"],
        "abandono_riesgo": ["satisfaccion", "nps", "pagos_puntuales_pct"],
    }
    for field, parents in essential.items():
        require(set(parents) <= set(deps[field]), f"Faltan dependencias de {field}")
    supported = {"gasto_mensual": {"transacciones_mes"}, "uso_tarjeta_pct": {"transacciones_mes"},
                 "abandono_riesgo": {"satisfaccion", "nps", "pagos_puntuales_pct"}}
    require(set(c["stale_dependencies"]) == set(supported), "Derivados obsoletos incompletos")
    for field, parents in c["stale_dependencies"].items():
        require(set(parents) == supported[field] and set(parents) <= set(deps[field]),
                f"Invalidación de derivados incompatible en {field}")
