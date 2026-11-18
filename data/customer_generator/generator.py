"""Coordinación de borradores, árbol de políticas, reparación y trazabilidad."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
from typing import Any

import numpy as np
import pandas as pd

from .config import COLUMNS, ConfigurationError, checked_config, normalize_weights, parameters
from .policies import (
    average_ticket, card_probabilities, consumption, create_profile, digital_value,
    eligible_products, experience_value, missing, numeric, risk_value,
    segment_probabilities, usage_value, valid_number, validate_frame,
)
from .reference import ReferenceProfile, file_sha256, read_csv

AUDIT_COLUMNS = ["row_key", "original_customer_id", "original_index", "field", "previous_value",
                 "new_value", "policy_id", "action", "reason"]
DECISION_COLUMNS = ["row_key", "policy_id", "branch", "details"]


def json_value(value: Any) -> Any:
    """Convierte escalares pandas/NumPy y faltantes a JSON estricto."""
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, np.generic):
        return json_value(value.item())
    if missing(value):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


@dataclass
class GenerationResult:
    """Las auxiliares, decisiones y claves de fila quedan fuera de las 16 columnas."""

    draft: pd.DataFrame
    customers: pd.DataFrame
    audit: pd.DataFrame
    decisions: pd.DataFrame
    auxiliary: pd.DataFrame
    report: dict[str, Any]

    def export(self, directory: str | Path, protected_paths: tuple[Path, ...] = ()) -> Path:
        """Crea una carpeta nueva; jamás reutiliza archivos ni el CSV original."""
        destination = Path(directory)
        names = ["borradores.csv", "clientes.csv", "auditoria.csv", "decisiones.csv", "auxiliares.csv", "reporte.json"]
        protected = {p.resolve() for p in protected_paths}
        if any((destination / name).resolve() in protected for name in names):
            raise ValueError("La salida coincidiría con un insumo protegido")
        destination.mkdir(parents=True, exist_ok=False)
        for name, frame in zip(names, [self.draft, self.customers, self.audit, self.decisions, self.auxiliary]):
            frame.to_csv(destination / name, index=False)
        report = dict(self.report)
        report["artifacts"] = {name: {"sha256": file_sha256(destination / name)} for name in names[:-1]}
        (destination / "reporte.json").write_text(
            json.dumps(json_value(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        return destination


class CustomerGenerator:
    """Una instancia, un único Generator local; una instancia nueva reinicia la semilla.

    Generar define relaciones desde el perfil. Reparar conserva valores observados
    válidos aunque sean inusuales; sólo los derivados declarados se invalidan al
    cambiar sus entradas. No se fuerza gasto <= ingreso ni acuerdo NPS/satisfacción.
    """

    def __init__(self, config: dict[str, Any] | None = None, seed: int = 42,
                 reference: str | Path | pd.DataFrame | ReferenceProfile | None = None) -> None:
        self.config = checked_config(config)
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("La semilla debe ser un entero no negativo")
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.reference = (ReferenceProfile.from_path(reference, self.config) if isinstance(reference, (str, Path))
                          else ReferenceProfile.from_frame(reference, self.config) if isinstance(reference, pd.DataFrame)
                          else reference)
        self.domains = parameters(self.config, "DOM-01")["numeric"]
        products = parameters(self.config, "TAR-01")["products"]
        segments = parameters(self.config, "DOM-01")["segments"]
        self.card_priors = self.reference.priors("tarjeta_santander", [p["name"] for p in products]) if self.reference else None
        self.segment_priors = self.reference.priors("segmento", segments) if self.reference else None
        # Valida también la compatibilidad de los insumos con los pesos efectivos.
        prefs = {name: 1 / len(parameters(self.config, "AUX-01")["preferences"])
                 for name in parameters(self.config, "AUX-01")["preferences"]}
        try:
            card_probabilities(self.domains["edad"]["min"], self.domains["ingreso_mensual"]["min"],
                               prefs, self.config, self.card_priors)
            for income in [0, *parameters(self.config, "SEG-01")["income_thresholds"]]:
                segment_probabilities(income, 0, self.config, self.segment_priors)
        except ConfigurationError as exc:
            raise ConfigurationError(f"Referencia/configuración sin probabilidades compatibles: {exc}") from exc
        self.operation_index = 0

    def _sample_initial(self, field: str) -> float | int:
        if self.reference and field in self.reference.samples:
            value = float(self.rng.choice(self.reference.samples[field]))
            return int(value) if self.domains[field]["integer"] else value
        else:
            p = parameters(self.config, "INI-01")
            if field == "edad":
                k = int(self.rng.choice(len(p["age_bins"]), p=normalize_weights(p["age_weights"])))
                low, high = p["age_bins"][k]
                return int(self.rng.integers(low, high + 1))
            if self.rng.random() < p["income_zero_probability"]:
                return 0.0
            k = int(self.rng.choice(len(p["income_weights"]), p=normalize_weights(p["income_weights"])))
            exponent = self.rng.normal(math.log(p["income_medians"][k]), p["income_sigmas"][k])
            value = math.exp(min(float(exponent), math.log(self.domains[field]["max"])))
        return int(value) if self.domains[field]["integer"] else min(self.domains[field]["max"], round(value, 2))

    def generate(self, n: int = 10000, candidates: pd.DataFrame | None = None) -> GenerationResult:
        """Genera un borrador; candidatos opcionales conservan sus datos provisionales."""
        if isinstance(n, bool) or not isinstance(n, int) or n < 0:
            raise ValueError("n debe ser entero no negativo")
        if candidates is not None:
            if len(candidates) != n:
                raise ValueError("n debe coincidir con el número de candidatos")
            return self._process(candidates, "generar_candidatos", {"kind": "DataFrame de candidatos"})
        prefix = parameters(self.config, "DOM-01")["id_prefix"]
        records = [{"customer_id": f"{prefix}{i:08d}", "edad": self._sample_initial("edad"),
                    "ingreso_mensual": self._sample_initial("ingreso_mensual")} for i in range(n)]
        draft = pd.DataFrame(records, columns=COLUMNS)
        return self._process(draft, "generar", {"kind": "borrador nuevo"})

    def repair(self, source: pd.DataFrame | str | Path) -> GenerationResult:
        """Recibe una copia lógica; nunca escribe ni modifica el objeto de entrada."""
        if isinstance(source, (str, Path)):
            frame = read_csv(source)
            provenance = {"kind": "CSV a reparar; no referencia", "path": str(source), "sha256": file_sha256(source)}
        elif isinstance(source, pd.DataFrame):
            frame = source.copy(deep=True)
            provenance = {"kind": "DataFrame a reparar; no referencia"}
        else:
            raise TypeError("Se requiere DataFrame o ruta de CSV")
        return self._process(frame, "reparar", provenance)

    def validate(self, frame: pd.DataFrame) -> pd.DataFrame:
        return validate_frame(frame, self.config)

    def _process(self, source: pd.DataFrame, operation: str, provenance: dict[str, Any]) -> GenerationResult:
        if source.columns.duplicated().any():
            raise ValueError("Hay nombres de columna duplicados")
        original_draft = source.copy(deep=True)
        draft = source.reindex(columns=COLUMNS).copy(deep=True)
        if not all(pd.api.types.is_scalar(v) for v in draft.to_numpy().flat):
            raise ValueError("Cada celda debe contener un escalar")
        before = self.validate(draft)
        audit: list[dict[str, Any]] = []
        decisions: list[dict[str, Any]] = []
        auxiliaries: list[dict[str, Any]] = []
        final: list[dict[str, Any]] = []
        seen: set[str] = set()
        reserved = {x for x in draft["customer_id"] if isinstance(x, str) and x.strip()}
        prefix = parameters(self.config, "DOM-01")["id_prefix"]
        available = iter(f"{prefix}{i:08d}" for i in range(len(reserved) + len(draft) + 1)
                         if f"{prefix}{i:08d}" not in reserved)
        for pos, (original_index, row) in enumerate(zip(draft.index, draft.to_dict("records"))):
            original = row.copy()
            key = f"row-{pos:08d}"
            changed: set[str] = set()
            profile: dict[str, Any] | None = None
            aux: dict[str, Any] = {"row_key": key, "original_index": str(original_index),
                                    "original_customer_id": json_value(original["customer_id"])}

            def decision(pid: str, branch: str, details: dict[str, Any]) -> None:
                decisions.append(dict(row_key=key, policy_id=pid, branch=branch,
                                      details=json.dumps(json_value(details), ensure_ascii=False, sort_keys=True, allow_nan=False)))

            def set_value(field: str, value: Any, pid: str, reason: str) -> None:
                old = row[field]
                old_is_valid = field not in self.domains or valid_number(old, self.domains[field])
                if not missing(old) and old_is_valid and old == value:
                    return
                action = "completar" if missing(old) else "corregir"
                audit.append(dict(row_key=key, original_customer_id=json_value(original["customer_id"]),
                                  original_index=str(original_index), field=field, previous_value=json_value(old),
                                  new_value=json_value(value), policy_id=pid, action=action, reason=reason))
                row[field] = value
                changed.add(field)

            def valid(field: str) -> bool:
                return valid_number(row[field], self.domains[field])

            def latent() -> dict[str, Any]:
                nonlocal profile
                if profile is None:
                    profile = create_profile(self.rng, row, self.config)
                    aux.update({k: v for k, v in profile.items() if k != "preferences"})
                    aux.update({f"preferencia_{k}": v for k, v in profile["preferences"].items()})
                    decision("AUX-01", "perfil_compartido", profile)
                return profile

            def stale(field: str) -> bool:
                return bool(changed & set(self.config["stale_dependencies"].get(field, [])))

            ident = row["customer_id"]
            if not isinstance(ident, str) or not ident.strip() or ident in seen:
                set_value("customer_id", next(available), "DOM-01", "ID faltante, no textual o duplicado; clave de fila conserva identidad original")
            seen.add(row["customer_id"])
            for field in ["edad", "ingreso_mensual"]:
                if not valid(field):
                    set_value(field, self._sample_initial(field), "INI-01", "faltante o fuera de dominio; marginal empírica o DEMO")
            if operation == "generar":
                decision("INI-01", "borrador_inicial", {k: row[k] for k in ["customer_id", "edad", "ingreso_mensual"]})
            cro = parameters(self.config, "CRO-01")
            maximum = min(self.domains["antiguedad_meses"]["max"], int((float(row["edad"]) - cro["relationship_start_age"]) * 12))
            if not valid("antiguedad_meses") or float(row["antiguedad_meses"]) > maximum:
                old_tenure = numeric(row["antiguedad_meses"])
                if old_tenure is not None:
                    tenure = int(np.clip(round(old_tenure), 0, maximum))
                    branch = "acotar_obligacion"
                elif self.reference and "antiguedad_meses" in self.reference.samples:
                    tenure = min(maximum, int(self.rng.choice(self.reference.samples["antiguedad_meses"])))
                    branch = "empirica_con_cota"
                else:
                    tenure = int(round(maximum * self.rng.beta(*cro["beta"])))
                    branch = "beta_demo_con_cota"
                set_value("antiguedad_meses", tenure, "CRO-01", "completar o acotar a cronología bancaria")
                decision("CRO-01", branch, {"maximum_months": maximum})
            if missing(row["segmento"]) or row["segmento"] not in parameters(self.config, "DOM-01")["segments"]:
                options, probabilities, tier = segment_probabilities(float(row["ingreso_mensual"]), float(row["antiguedad_meses"]), self.config, self.segment_priors)
                choice = str(self.rng.choice(options, p=probabilities))
                set_value("segmento", choice, "SEG-01", "elección por capacidad y relación, sin restricción por tarjeta")
                decision("SEG-01", f"nivel_ingreso_{tier}", {"options": options, "probabilities": probabilities.tolist(), "selected": choice})
            allowed = [p["name"] for p in eligible_products(float(row["edad"]), float(row["ingreso_mensual"]), self.config)]
            if missing(row["tarjeta_santander"]) or row["tarjeta_santander"] not in allowed:
                options, probabilities = card_probabilities(float(row["edad"]), float(row["ingreso_mensual"]), latent()["preferences"], self.config, self.card_priors)
                choice = str(self.rng.choice(options, p=probabilities))
                set_value("tarjeta_santander", choice, "TAR-01", "producto faltante/desconocido/inelegible; se conserva ingreso válido")
                decision("TAR-01", "elegir_elegible", {"eligible": options, "probabilities": probabilities.tolist(), "selected": choice})
            else:
                decision("TAR-01", "conservar_elegible", {"eligible": allowed, "selected": row["tarjeta_santander"]})
            ticket: float | None = None
            if not valid("transacciones_mes"):
                count, ticket, details = consumption(self.rng, float(row["ingreso_mensual"]), row["tarjeta_santander"], latent(), self.config)
                set_value("transacciones_mes", count, "CON-01", "frecuencia condicionada al perfil y rama de actividad")
                decision("CON-01", details["branch"], details)
            count = int(float(row["transacciones_mes"]))
            if count == 0:
                for field in parameters(self.config, "CON-01")["zero_when_inactive"]:
                    set_value(field, 0.0, "CON-01", "cero compras en el mismo periodo; convención de inactividad")
                aux["average_ticket"] = 0.0
            elif not valid("gasto_mensual") or stale("gasto_mensual"):
                original_count = numeric(original["transacciones_mes"])
                if original_count is not None and original_count > 0 and valid_number(original["gasto_mensual"], self.domains["gasto_mensual"]):
                    ticket = float(original["gasto_mensual"]) / original_count
                    ticket_source = "ticket implícito conservado de las compras originales"
                else:
                    ticket = ticket if ticket is not None else average_ticket(self.rng, float(row["ingreso_mensual"]), row["tarjeta_santander"], latent(), self.config)
                    ticket_source = "ticket condicionado al perfil DEMO"
                amount = min(self.domains["gasto_mensual"]["max"], round(count * ticket, 2))
                set_value("gasto_mensual", amount, "CON-01", "derivado obsoleto o inválido; " + ticket_source)
                aux["average_ticket"] = amount / count
                decision("CON-01", "gasto_conjunto", {"count": count, "average_ticket": ticket, "ticket_source": ticket_source, "spend": amount})
            else:
                aux["average_ticket"] = float(row["gasto_mensual"]) / count
            for field in ["ecommerce_pct", "app_logins_mes"]:
                if not valid(field):
                    set_value(field, digital_value(field, self.rng, latent(), self.config), "DIG-01", "afinidad digital compartida y variación individual")
                    decision("DIG-01", field, {"selected": row[field], "digital": latent()["digital"]})
            for field in ["satisfaccion", "nps", "pagos_puntuales_pct", "activacion_promos_pct"]:
                if not valid(field):
                    set_value(field, experience_value(field, self.rng, latent(), row["tarjeta_santander"], self.config), "EXP-01", "experiencia, comportamiento de pago o preferencias con ruido individual")
                    decision("EXP-01", field, {"selected": row[field]})
            if count and (not valid("uso_tarjeta_pct") or stale("uso_tarjeta_pct")):
                set_value("uso_tarjeta_pct", usage_value(count, self.rng, latent(), self.config), "USO-01", "intensidad sintética faltante/inválida u obsoleta al cambiar compras")
            if not valid("abandono_riesgo") or stale("abandono_riesgo"):
                set_value("abandono_riesgo", risk_value(row, self.rng, self.config), "RIE-01", "score faltante/inválido u obsoleto al cambiar satisfacción, NPS o puntualidad")
            auxiliaries.append(aux)
            final.append(row)
        customers = pd.DataFrame(final, columns=COLUMNS)
        for field, spec in self.domains.items():
            customers[field] = pd.to_numeric(customers[field]).astype("int64" if spec["integer"] else "float64")
        after = self.validate(customers)
        if not after.empty:
            raise RuntimeError(f"Error interno: quedaron infracciones obligatorias: {after.head().to_dict('records')}")
        audit_frame = pd.DataFrame(audit, columns=AUDIT_COLUMNS)
        decision_frame = pd.DataFrame(decisions, columns=DECISION_COLUMNS)
        aux_frame = pd.DataFrame(auxiliaries) if auxiliaries else pd.DataFrame(columns=["row_key", "original_index", "original_customer_id"])
        report = self._report(operation, provenance, draft, customers, before, after, audit_frame)
        report["input_columns"] = source.columns.tolist()
        report["extra_columns_preserved_only_in_draft"] = [str(f) for f in source.columns if f not in COLUMNS]
        self.operation_index += 1
        return GenerationResult(original_draft, customers, audit_frame, decision_frame, aux_frame, report)

    def _report(self, operation: str, provenance: dict[str, Any], draft: pd.DataFrame,
                customers: pd.DataFrame, before: pd.DataFrame, after: pd.DataFrame,
                audit: pd.DataFrame) -> dict[str, Any]:
        def issues(frame: pd.DataFrame) -> dict[str, Any]:
            return {"total": len(frame), "by_kind": frame["kind"].value_counts().to_dict(),
                    "by_policy": frame["policy_id"].value_counts().to_dict(),
                    "by_field": frame["field"].value_counts().to_dict()}

        card_summary = {}
        for card, group in customers.groupby("tarjeta_santander", sort=True):
            s = group["ingreso_mensual"]
            card_summary[card] = {"count": len(s), "min": s.min(), "p25": s.quantile(.25),
                                  "median": s.median(), "mean": s.mean(), "p75": s.quantile(.75), "max": s.max()}
        serialized = json.dumps(self.config, ensure_ascii=False, sort_keys=True, allow_nan=False)
        code_hash = hashlib.sha256()
        for p in sorted(Path(__file__).parent.glob("*.py")):
            code_hash.update(p.name.encode())
            code_hash.update(p.read_bytes())
        return json_value({
            "operation": operation, "rows": len(customers), "seed": self.seed,
            "operation_index": self.operation_index, "rng": type(self.rng.bit_generator).__name__,
            "versions": {"policies": self.config["version"], "python": platform.python_version(),
                         "numpy": np.__version__, "pandas": pd.__version__},
            "code_sha256": code_hash.hexdigest(), "configuration_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
            "configuration": self.config, "input": provenance,
            "reference": self.reference.metadata if self.reference else {
                "kind": "sin_referencia_distribuciones_DEMO", "source": "config.demo.json / supuestos del equipo",
                "external_sources": "Sin INEGI ni fuentes externas descargadas; adaptadores ENIGH/ENIF documentados en README"
            },
            "parameter_origins": {pid: p["origin"] for pid, p in self.config["policies"].items()},
            "effective_sampling_sources": {
                **{field: "soporte empírico de referencia" if self.reference and field in self.reference.samples
                   else "configuración DEMO" for field in ["edad", "ingreso_mensual", "antiguedad_meses"]},
                "card_base_weights": "frecuencias de referencia" if self.card_priors else "configuración DEMO",
                "segment_base_weights": "frecuencias de referencia" if self.segment_priors else "configuración DEMO",
                "dependent_fields": "políticas condicionales y coeficientes DEMO; no se conserva exactamente cada marginal de referencia"
            },
            "missing_before": {f: int(draft[f].map(missing).sum()) for f in COLUMNS},
            "missing_after": {f: int(customers[f].map(missing).sum()) for f in COLUMNS},
            "violations_before": issues(before), "violations_after": issues(after),
            "changes": {"total": len(audit), "by_action": audit["action"].value_counts().to_dict(),
                        "by_policy": audit["policy_id"].value_counts().to_dict(),
                        "by_field": audit["field"].value_counts().to_dict()},
            "card_frequencies": customers["tarjeta_santander"].value_counts().to_dict(),
            "income_by_card": card_summary,
            "limitations": ["Las relaciones son supuestos de simulación; no hallazgos sobre clientes reales",
                            "Correlaciones distintas no prueban realismo", "Los mínimos de ingreso son DEMO y se aplican al ingreso actual",
                            "Reparar datos válidos no les introduce necesariamente nuevas relaciones estadísticas",
                            "Mismo código, versiones, orden, insumos, configuración y semilla en instancia nueva reproducen el resultado"],
        })
