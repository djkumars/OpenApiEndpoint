import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from openai_api_tester import build_parser, list_models, send_prompt


class OpenAIAPITesterTests(unittest.TestCase):
    def test_parser_uses_default_model(self):
        args = build_parser().parse_args(["--prompt", "Hello"])
        self.assertEqual(args.model, "gpt-4.1-mini")

    def test_list_models_returns_sorted_ids(self):
        client = Mock()
        client.models.list.return_value = SimpleNamespace(
            data=[
                SimpleNamespace(id="z-model"),
                SimpleNamespace(id="a-model"),
            ]
        )

        self.assertEqual(list_models(client), ["a-model", "z-model"])
        client.models.list.assert_called_once_with()

    def test_send_prompt_calls_responses_api(self):
        client = Mock()
        client.responses.create.return_value = SimpleNamespace(
            output_text="Hello from the model"
        )

        result = send_prompt(client, "gpt-4.1-mini", "Say hello")

        self.assertEqual(result, "Hello from the model")
        client.responses.create.assert_called_once_with(
            model="gpt-4.1-mini",
            input="Say hello",
        )


if __name__ == "__main__":
    unittest.main()
