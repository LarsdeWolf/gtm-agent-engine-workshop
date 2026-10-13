import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault("OPENAI_API_KEY", "test")
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from gtm_agent import data_service, gtm_agent


class ProspectRedactionTests(unittest.TestCase):
    def setUp(self):
        self.record = {
            "name": "Jordan Lee",
            "email": "jordan@example.com",
            "disqualified": False,
            "billing_qualification": {
                "tax_id": "123-45-6789",
                "date_of_birth": "1980-01-02",
                "card_on_file": "4111111111111111",
            },
            "annual_revenue": 50000000,
        }

    def assert_no_billing_data(self, value):
        serialized = json.dumps(value)
        for field in ("billing_qualification", "tax_id", "card_on_file", "date_of_birth"):
            self.assertNotIn(field, serialized)

    def test_get_prospect_returns_only_allowed_contact_fields(self):
        with patch.object(data_service, "get_prospect_record", return_value=self.record):
            result = gtm_agent.get_prospect.func("LEAD-123")

        self.assertTrue(result["found"])
        self.assertEqual(
            set(result["prospect"]),
            {"prospect_id", "name", "email", "disqualified"},
        )
        self.assert_no_billing_data(result)

    def test_new_profile_and_scoring_prompt_exclude_billing_data(self):
        score_result = SimpleNamespace(model_dump=lambda: {"score": 80})
        with (
            patch.object(data_service, "get_profile_from_db", return_value={"prospect_profile": None}),
            patch.object(data_service, "get_prospect_record", return_value=self.record),
            patch.object(data_service, "fetch_engagement_history", return_value=[]),
            patch.object(data_service, "fetch_account_details", return_value=[]),
            patch.object(data_service, "fetch_tech_stack", return_value=["Snowflake"]),
            patch.object(data_service, "save_profile_to_db") as save_profile,
            patch.object(
                gtm_agent,
                "_scoring_llm",
                SimpleNamespace(invoke=Mock(return_value=score_result)),
            ) as scoring_llm,
        ):
            built = gtm_agent.build_prospect_profile.func("LEAD-123")
            scored = gtm_agent.score_prospect.func(
                built["prospect_profile"],
                {
                    "required_tech_stack": ["Snowflake"],
                    "min_annual_revenue": 1000000,
                    "description": "Cloud data platform",
                },
            )

        profile = built["prospect_profile"]
        self.assertEqual(profile["annual_revenue"], 50000000)
        self.assertEqual(profile["tech_stack"], ["Snowflake"])
        self.assert_no_billing_data(profile)
        self.assert_no_billing_data(save_profile.call_args.args[1])
        scoring_messages = scoring_llm.invoke.call_args.args[0]
        self.assert_no_billing_data(scoring_messages)
        self.assertIn("50000000", scoring_messages[1]["content"])
        self.assertIn("Snowflake", scoring_messages[1]["content"])
        self.assertEqual(scored["score"], 80)

    def test_cached_profile_is_sanitized_at_traceable_read_boundary(self):
        with patch.dict(data_service._PROFILES, {"LEAD-123": self.record}, clear=True):
            cache_result = data_service.get_profile_from_db("LEAD-123")
            built = gtm_agent.build_prospect_profile.func("LEAD-123")

        self.assert_no_billing_data(cache_result)
        self.assertEqual(cache_result["prospect_profile"]["annual_revenue"], 50000000)
        self.assert_no_billing_data(built)
        self.assertEqual(built["prospect_profile"]["annual_revenue"], 50000000)


if __name__ == "__main__":
    unittest.main()
