import unittest
from types import SimpleNamespace
from unittest.mock import patch

from gtm_agent import data_service, gtm_records
from gtm_agent.gtm_agent import SYSTEM_PROMPT, send_prospect_email


class SendProspectEmailTests(unittest.TestCase):
    def setUp(self):
        self.runtime = SimpleNamespace(config={"metadata": {"user_id": "rep_amills"}})
        self.prospect = {
            "prospect_id": "LEAD-50001",
            "name": "LLM-provided name",
            "email": "llm-provided@example.com",
            "disqualified": False,
        }

    def test_disqualified_records_block_email_using_authoritative_status(self):
        for prospect_id in (
            "LEAD-50001",
            "LEAD-50002",
            "LEAD-50003",
            "LEAD-50004",
            "LEAD-50005",
        ):
            with self.subTest(prospect_id=prospect_id):
                prospect = {**self.prospect, "prospect_id": prospect_id}
                with (
                    patch.object(
                        data_service,
                        "get_prospect_record",
                        wraps=data_service.get_prospect_record,
                    ) as get_record,
                    patch("gtm_agent.gtm_agent.uuid.uuid4") as generate_id,
                ):
                    result = send_prospect_email.func(
                        prospect, "Subject", "Body", self.runtime
                    )
                get_record.assert_called_once_with(prospect_id)
                generate_id.assert_not_called()
                self.assertEqual(
                    result,
                    {
                        "status": "blocked",
                        "error": "Prospect is disqualified; email not sent.",
                        "prospect_id": prospect_id,
                    },
                )
                self.assertNotIn("message_id", result)

    def test_non_disqualified_prospect_is_sent(self):
        with patch(
            "gtm_agent.gtm_agent.data_service.get_prospect_record",
            return_value={"disqualified": False},
        ) as get_record:
            result = send_prospect_email.func(
                self.prospect, "Subject", "Body", self.runtime
            )

        get_record.assert_called_once_with("LEAD-50001")
        self.assertEqual(result["status"], "sent")
        self.assertEqual(result["to"], self.prospect["email"])

    def test_prospect_without_id_keeps_existing_send_behavior(self):
        prospect = {"name": "Prospect", "email": "prospect@example.com"}
        with patch(
            "gtm_agent.gtm_agent.data_service.get_prospect_record"
        ) as get_record:
            result = send_prospect_email.func(
                prospect, "Subject", "Body", self.runtime
            )

        get_record.assert_not_called()
        self.assertEqual(result["status"], "sent")
        self.assertIn("message_id", result)

    def test_prompt_explains_blocked_disqualified_sends(self):
        self.assertIn("never email a prospect whose record is disqualified", SYSTEM_PROMPT)
        self.assertIn("no email was sent", SYSTEM_PROMPT)
        self.assertIn("send tool returns a blocked result", SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
