"""Pruebas de invariantes; no dependen de correlaciones ni de sorteos afortunados."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from data.customer_generator import CustomerGenerator
from data.customer_generator.cli import main
from data.customer_generator.config import (
    COLUMNS, ConfigurationError, load_config, normalize_weights, parameters,
)
from data.customer_generator.policies import card_probabilities, consumption, eligible_products, segment_probabilities
from data.customer_generator.reference import ReferenceProfile, read_csv, sample_weighted_individual_income

DATA = Path(__file__).resolve().parents[1]


def valid_record(**changes: object) -> pd.DataFrame:
    """Perfil deliberadamente poco frecuente, pero válido bajo todas las reglas."""
    row = dict(zip(COLUMNS, ["original", 40, "Mass", 12, 150000, "Free", 60, 10,
                            200000, 100, 0, 0, 10, 1.0, .95, 0]))
    row.update(changes)
    return pd.DataFrame([row], columns=COLUMNS)


class EligibilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config()
        self.preferences = {p: .25 for p in parameters(self.config, "AUX-01")["preferences"]}

    def test_every_income_and_age_boundary(self) -> None:
        for p in parameters(self.config, "TAR-01")["products"]:
            with self.subTest(product=p["name"]):
                names = lambda age, income: [x["name"] for x in eligible_products(age, income, self.config)]
                self.assertIn(p["name"], names(p["min_age"], p["min_income"]))
                self.assertNotIn(p["name"], names(p["min_age"] - 1, p["min_income"]))
                self.assertNotIn(p["name"], names(p["min_age"], p["min_income"] - .01))

    def test_basic_cards_still_have_positive_probability_at_high_income(self) -> None:
        options, probs = card_probabilities(50, 1000000, self.preferences, self.config)
        self.assertGreater(probs[options.index("Free")], 0)
        self.assertGreater(probs[options.index("LikeU")], 0)
        self.assertEqual(len(options), 9)
        self.assertAlmostEqual(probs.sum(), 1)
        self.assertTrue((probs > 0).all())

    def test_filter_then_normalize(self) -> None:
        options, probs = card_probabilities(18, 9999.99, self.preferences, self.config)
        self.assertEqual(options, ["LikeU", "Free"])
        np.testing.assert_allclose(probs, np.array([.25, .20]) / .45)
        self.assertAlmostEqual(normalize_weights([1e308, 1e308]).sum(), 1)

    def test_segment_changes_probabilities_not_hard_card_rules(self) -> None:
        a, low, _ = segment_probabilities(0, 0, self.config)
        b, high, _ = segment_probabilities(1000000, 240, self.config)
        self.assertEqual(a, b)
        self.assertTrue((low > 0).all() and (high > 0).all())
        self.assertFalse(np.array_equal(low, high))

    def test_impossible_weights_and_eligibility(self) -> None:
        for weights in [[0, 0], [-1, 2], [float("nan")], [], [float("inf")]]:
            with self.subTest(weights=weights), self.assertRaises(ConfigurationError):
                normalize_weights(weights)
        with self.assertRaisesRegex(ConfigurationError, "ningún producto"):
            card_probabilities(10, 0, self.preferences, self.config)


class ConfigurationTests(unittest.TestCase):
    def test_impossible_configs_are_rejected_before_generation(self) -> None:
        def all_premium(c: dict) -> None:
            for p in parameters(c, "TAR-01")["products"]:
                p["min_income"] = 50000

        def zero_weights(c: dict) -> None:
            for p in parameters(c, "TAR-01")["products"]:
                p["weight"] = 0

        def cycle(c: dict) -> None:
            c["dependencies"]["edad"] = ["antiguedad_meses"]

        def backwards_thresholds(c: dict) -> None:
            parameters(c, "SEG-01")["income_thresholds"] = [60000, 20000]

        def invalid_probability(c: dict) -> None:
            parameters(c, "DIG-01")["app_zero_probability"] = 1.1

        def missing_dependency(c: dict) -> None:
            c["dependencies"]["gasto_mensual"] = []

        def zero_tier(c: dict) -> None:
            parameters(c, "SEG-01")["tier_factors"][0] = [0, 0, 0]

        def negative_noise(c: dict) -> None:
            parameters(c, "RIE-01")["noise"] = -1

        for mutate in [all_premium, zero_weights, cycle, backwards_thresholds,
                       invalid_probability, missing_dependency, zero_tier, negative_noise]:
            c = load_config()
            mutate(c)
            with self.subTest(case=mutate.__name__), self.assertRaises(ConfigurationError):
                CustomerGenerator(c)

    def test_input_configuration_is_copied(self) -> None:
        c = load_config()
        g = CustomerGenerator(c)
        c["version"] = "changed"
        self.assertNotEqual(g.config["version"], c["version"])


class RepairTests(unittest.TestCase):
    def test_conserve_valid_values_including_rare_combinations(self) -> None:
        original = valid_record()
        snapshot = original.copy(deep=True)
        result = CustomerGenerator(seed=7).repair(original)
        assert_frame_equal(original, snapshot)
        assert_frame_equal(result.customers, original, check_dtype=False)
        self.assertTrue(result.audit.empty)
        self.assertTrue(CustomerGenerator().validate(result.customers).empty)

    def test_income_is_not_increased_to_preserve_premium_card(self) -> None:
        row = valid_record(ingreso_mensual=9999.99, tarjeta_santander="World Elite")
        result = CustomerGenerator(seed=7).repair(row)
        self.assertEqual(result.customers.at[0, "ingreso_mensual"], 9999.99)
        self.assertIn(result.customers.at[0, "tarjeta_santander"], ["LikeU", "Free"])
        self.assertEqual(result.audit["field"].tolist(), ["tarjeta_santander"])

    def test_second_repair_is_value_identical_and_has_empty_audit(self) -> None:
        first = CustomerGenerator(seed=7).repair(DATA / "examples/clientes_inconsistentes.csv")
        second = CustomerGenerator(seed=8).repair(first.customers)
        assert_frame_equal(first.customers, second.customers)
        self.assertTrue(second.audit.empty)
        self.assertEqual(set(first.audit["action"]), {"corregir", "completar"})
        self.assertEqual(first.customers.at[1, "customer_id"].startswith("SYN-"), True)
        self.assertEqual(first.customers.at[0, "customer_id"], "DUP-01")
        self.assertTrue(first.customers.customer_id.is_unique)

    def test_revalidate_chronology_after_age_change(self) -> None:
        c = load_config()
        parameters(c, "INI-01")["age_bins"] = [[18, 18]]
        parameters(c, "INI-01")["age_weights"] = [1]
        result = CustomerGenerator(c).repair(valid_record(edad=-1, antiguedad_meses=240))
        self.assertEqual(result.customers.at[0, "edad"], 18)
        self.assertEqual(result.customers.at[0, "antiguedad_meses"], 0)
        self.assertEqual(set(result.audit.field), {"edad", "antiguedad_meses"})

    def test_revalidate_card_after_income_change(self) -> None:
        c = load_config()
        parameters(c, "INI-01")["income_zero_probability"] = 1
        result = CustomerGenerator(c).repair(valid_record(ingreso_mensual=-1, tarjeta_santander="World Elite"))
        self.assertEqual(result.customers.at[0, "ingreso_mensual"], 0)
        self.assertIn(result.customers.at[0, "tarjeta_santander"], ["Free", "LikeU"])
        self.assertEqual(result.customers.at[0, "gasto_mensual"], 200000)
        self.assertEqual(result.customers.at[0, "segmento"], "Mass")

    def test_stale_spend_and_usage_after_count_changes(self) -> None:
        c = load_config()
        parameters(c, "CON-01")["inactive_probability"] = 1
        parameters(c, "CON-01")["activity_protection"] = 0
        result = CustomerGenerator(c).repair(valid_record(transacciones_mes=-1))
        for field in ["transacciones_mes", "gasto_mensual", "uso_tarjeta_pct", "ecommerce_pct"]:
            self.assertEqual(result.customers.at[0, field], 0)
        self.assertEqual(result.customers.at[0, "app_logins_mes"], 0)
        self.assertEqual(result.customers.at[0, "abandono_riesgo"], .95)

    def test_stale_risk_after_experience_changes(self) -> None:
        result = CustomerGenerator(seed=7).repair(valid_record(satisfaccion=9))
        self.assertEqual(set(result.audit.field), {"satisfaccion", "abandono_riesgo"})
        self.assertIn("obsoleto", result.audit.iloc[-1].reason)
        self.assertEqual(result.customers.at[0, "nps"], 10)

    def test_bool_is_invalid_numeric_and_nullable_fields_work(self) -> None:
        result = CustomerGenerator(seed=7).repair(valid_record(transacciones_mes=0, ecommerce_pct=False,
                                                             segmento=pd.NA, tarjeta_santander=pd.NA))
        self.assertEqual(result.customers.at[0, "ecommerce_pct"], 0)
        self.assertTrue(CustomerGenerator().validate(result.customers).empty)
        self.assertIn("ecommerce_pct", result.audit.field.tolist())

    def test_duplicate_original_ids_and_indices_use_distinct_audit_keys(self) -> None:
        rows = pd.concat([valid_record(nps=None), valid_record(nps=None)], ignore_index=False)
        result = CustomerGenerator(seed=7).repair(rows)
        nps_audit = result.audit[result.audit.field == "nps"]
        self.assertEqual(nps_audit.original_customer_id.tolist(), ["original", "original"])
        self.assertEqual(nps_audit.row_key.nunique(), 2)
        self.assertEqual(nps_audit.original_index.tolist(), ["0", "0"])

    def test_future_original_id_is_reserved(self) -> None:
        rows = pd.concat([valid_record(customer_id=None), valid_record(customer_id="SYN-00000000")], ignore_index=True)
        result = CustomerGenerator().repair(rows)
        self.assertEqual(result.customers.at[1, "customer_id"], "SYN-00000000")
        self.assertTrue(result.customers.customer_id.is_unique)

    def test_draft_preserves_extra_columns_and_missing_schema_fields(self) -> None:
        source = pd.DataFrame({"customer_id": ["only-id"], "nota_interna": ["original"]})
        result = CustomerGenerator().repair(source)
        assert_frame_equal(result.draft, source)
        self.assertEqual(result.customers.columns.tolist(), list(COLUMNS))
        self.assertEqual(result.report["extra_columns_preserved_only_in_draft"], ["nota_interna"])


class ReferenceTests(unittest.TestCase):
    def test_empirical_support_zeros_extremes_categories_and_provenance(self) -> None:
        frame = pd.DataFrame({"edad": [18, 30, 100], "ingreso_mensual": [0, 100, 1e9],
                              "tarjeta_santander": ["Free", "Free", "LikeU"], "segmento": ["Mass"] * 3})
        profile = ReferenceProfile.from_frame(frame, load_config(), {"source": "fixture"})
        np.testing.assert_equal(profile.samples["ingreso_mensual"], [0, 100, 1e9])
        summary = profile.metadata["numeric_distributions"]["ingreso_mensual"]
        self.assertEqual(summary["zeros"], 1)
        self.assertEqual(summary["quantiles"]["max"], 1e9)
        self.assertEqual(profile.categories["tarjeta_santander"]["Free"], 2)
        self.assertEqual(profile.metadata["provenance"]["source"], "fixture")
        generator = CustomerGenerator(reference=frame)
        values = [generator._sample_initial("ingreso_mensual") for _ in range(20)]
        self.assertTrue(set(values) <= {0, 100, 1e9})

    def test_reference_cannot_silently_create_zero_eligible_weights(self) -> None:
        with self.assertRaisesRegex(ConfigurationError, "Referencia/configuración"):
            CustomerGenerator(reference=pd.DataFrame({"tarjeta_santander": ["World Elite"]}))

    def test_unknown_reference_catalog_is_rejected(self) -> None:
        with self.assertRaisesRegex(ConfigurationError, "catálogo"):
            CustomerGenerator(reference=pd.DataFrame({"tarjeta_santander": ["otro producto"]}))

    def test_reference_extreme_is_not_rounded_away_and_runs_reproduce(self) -> None:
        reference = pd.DataFrame({"ingreso_mensual": [0.000001], "edad": [18]})
        a = CustomerGenerator(seed=8, reference=reference).generate(12)
        b = CustomerGenerator(seed=8, reference=reference).generate(12)
        self.assertEqual(a.draft.ingreso_mensual.tolist(), [0.000001] * 12)
        for name in ["draft", "customers", "audit", "decisions", "auxiliary"]:
            assert_frame_equal(getattr(a, name), getattr(b, name))

    def test_repository_workbook_catalog_matches_configuration(self) -> None:
        profile = ReferenceProfile.from_path(DATA.parent / "dataset_dummy.xlsx", load_config())
        self.assertEqual(profile.metadata["rows"], 10000)
        configured = {p["name"] for p in parameters(load_config(), "TAR-01")["products"]}
        self.assertEqual(set(profile.categories["tarjeta_santander"]), configured)
        self.assertEqual(sum(profile.categories["tarjeta_santander"].values()), 10000)

    def test_weighted_individual_adapter_checks_units_and_weights(self) -> None:
        frame = pd.DataFrame({"persona_id": [1, 2], "ingreso_mensual_individual": [0, 100], "factor_expansion": [1, 0]})
        metadata = {"unit": "persona", "period": "mensual", "source": "fixture", "income_concept": "neto"}
        values = sample_weighted_individual_income(frame, np.random.default_rng(2), 5, metadata)
        np.testing.assert_equal(values, [0] * 5)
        with self.assertRaises(ValueError):
            sample_weighted_individual_income(frame, np.random.default_rng(2), 5, {**metadata, "unit": "hogar"})
        with self.assertRaises(ConfigurationError):
            sample_weighted_individual_income(frame.assign(factor_expansion=0), np.random.default_rng(2), 5, metadata)


class IntegrationTests(unittest.TestCase):
    def test_reproducibility_and_global_random_state_is_untouched(self) -> None:
        np.random.seed(100)
        state = np.random.get_state()
        a = CustomerGenerator(seed=456).generate(80)
        b = CustomerGenerator(seed=456).generate(80)
        for name in ["draft", "customers", "audit", "decisions", "auxiliary"]:
            assert_frame_equal(getattr(a, name), getattr(b, name))
        self.assertEqual(a.report, b.report)
        current = np.random.get_state()
        self.assertEqual(state[0], current[0])
        np.testing.assert_equal(state[1], current[1])
        self.assertEqual(state[2:], current[2:])

    def test_zero_rows_and_inconsistent_candidates(self) -> None:
        result = CustomerGenerator().generate(0)
        self.assertEqual(result.customers.columns.tolist(), list(COLUMNS))
        self.assertTrue(result.customers.empty)
        result = CustomerGenerator().generate(1, valid_record(tarjeta_santander="desconocida"))
        self.assertEqual(result.draft.at[0, "tarjeta_santander"], "desconocida")
        self.assertTrue(CustomerGenerator().validate(result.customers).empty)

    def test_inactive_and_active_branches_without_random_flakiness(self) -> None:
        c = load_config()
        profile = {"activity": .5}
        p = parameters(c, "CON-01")
        p["inactive_probability"], p["activity_protection"] = 1, 0
        count, ticket, branch = consumption(np.random.default_rng(1), 20000, "Free", profile, c)
        self.assertEqual((count, ticket, branch["branch"]), (0, 0, "inactivo"))
        p["inactive_probability"] = 0
        count, ticket, branch = consumption(np.random.default_rng(1), 20000, "Free", profile, c)
        self.assertGreater(count, 0)
        self.assertGreater(ticket, 0)
        self.assertEqual(branch["branch"], "activo")

    def test_cli_exports_new_files_and_csv_roundtrip_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "generated"
            self.assertEqual(main(["generate", "--rows", "25", "--seed", "13", "--output", str(out)]), 0)
            self.assertEqual(len(list(out.iterdir())), 6)
            report = json.loads((out / "reporte.json").read_text())
            self.assertEqual(report["rows"], 25)
            self.assertEqual(report["violations_after"]["total"], 0)
            repaired = CustomerGenerator().repair(out / "clientes.csv")
            self.assertTrue(repaired.audit.empty)
            assert_frame_equal(read_csv(out / "clientes.csv"), repaired.customers)
            before = (out / "clientes.csv").read_bytes()
            with self.assertRaises(FileExistsError):
                repaired.export(out)
            self.assertEqual((out / "clientes.csv").read_bytes(), before)

    def test_cli_repair_keeps_original_bytes(self) -> None:
        source = DATA / "examples/clientes_inconsistentes.csv"
        before = source.read_bytes()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "repair"
            self.assertEqual(main(["repair", "--input", str(source), "--output", str(out)]), 0)
            self.assertTrue(CustomerGenerator().validate(read_csv(out / "clientes.csv")).empty)
        self.assertEqual(source.read_bytes(), before)

    def test_full_10000_has_exact_schema_and_no_mandatory_violations(self) -> None:
        result = CustomerGenerator(seed=2026).generate(10000)
        self.assertEqual(result.customers.shape, (10000, 16))
        self.assertEqual(result.customers.columns.tolist(), list(COLUMNS))
        self.assertTrue(result.customers.customer_id.is_unique)
        self.assertFalse(result.customers.isna().any().any())
        self.assertEqual(result.report["violations_after"]["total"], 0)
        self.assertTrue(CustomerGenerator().validate(result.customers).empty)


if __name__ == "__main__":
    unittest.main()
