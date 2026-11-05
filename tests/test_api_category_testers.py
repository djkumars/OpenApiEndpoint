import base64
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import Mock

import agents_tool_calling
import embeddings_rag
import evaluation_monitoring
import model_customization
import safety_governance
import speech_audio
import text_generation
import vision_image_generation


class CategoryTesterTests(unittest.TestCase):
    def test_text_generation_uses_responses_api(self):
        client = Mock()
        client.responses.create.return_value.output_text = "Generated"

        result = text_generation.generate_text(client, "test-model", "Hello")

        self.assertEqual(result, "Generated")
        client.responses.create.assert_called_once_with(
            model="test-model", input="Hello"
        )

    def test_image_description_uses_responses_api(self):
        client = Mock()
        client.responses.create.return_value.output_text = "A landscape"

        result = vision_image_generation.describe_image(
            client, "vision-model", "https://example.test/image.png", "Describe it"
        )

        self.assertEqual(result, "A landscape")
        request = client.responses.create.call_args.kwargs
        self.assertEqual(
            request["input"][0]["content"][1]["image_url"],
            "https://example.test/image.png",
        )

    def test_image_generation_saves_returned_image(self):
        client = Mock()
        client.images.generate.return_value.data = [
            SimpleNamespace(b64_json=base64.b64encode(b"png data").decode("ascii"))
        ]

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "image.png"
            result = vision_image_generation.generate_image(
                client, "A test image", output, "image-model"
            )
            self.assertEqual(result.read_bytes(), b"png data")
        client.images.generate.assert_called_once_with(
            model="image-model", prompt="A test image"
        )

    def test_audio_transcription_and_speech(self):
        client = Mock()
        client.audio.transcriptions.create.return_value.text = "Transcript"
        with tempfile.TemporaryDirectory() as directory:
            audio_path = Path(directory) / "input.mp3"
            audio_path.write_bytes(b"audio")
            self.assertEqual(
                speech_audio.transcribe_audio(client, audio_path, "audio-model"),
                "Transcript",
            )

            output = Path(directory) / "output.mp3"
            self.assertEqual(
                speech_audio.synthesize_speech(
                    client, "Hello", output, "speech-model", "alloy"
                ),
                output,
            )
            client.audio.speech.create.return_value.stream_to_file.assert_called_once_with(
                output
            )

    def test_realtime_secret_uses_session_configuration(self):
        client = Mock()
        speech_audio.create_realtime_secret(client, "realtime-model")
        client.realtime.client_secrets.create.assert_called_once_with(
            session={"type": "realtime", "model": "realtime-model"}
        )

    def test_embedding_and_vector_store_search(self):
        client = Mock()
        client.embeddings.create.return_value.data = [
            SimpleNamespace(embedding=[0.1, 0.2])
        ]
        self.assertEqual(
            embeddings_rag.create_embedding(client, "embedding-model", "text"),
            [0.1, 0.2],
        )
        client.responses.create.return_value.output_text = "Found"
        self.assertEqual(
            embeddings_rag.search_vector_store(
                client, "response-model", "vs_test", "question"
            ),
            "Found",
        )
        request = client.responses.create.call_args.kwargs
        self.assertEqual(request["tools"][0]["vector_store_ids"], ["vs_test"])

    def test_function_tool_round_trip(self):
        client = Mock()
        client.responses.create.side_effect = [
            SimpleNamespace(
                id="resp_1",
                output=[
                    SimpleNamespace(
                        type="function_call",
                        name="get_current_time",
                        arguments="{}",
                        call_id="call_1",
                    )
                ],
            ),
            SimpleNamespace(output_text="It is now."),
        ]

        self.assertEqual(
            agents_tool_calling.run_function_tool(client, "test-model"), "It is now."
        )
        self.assertEqual(client.responses.create.call_count, 2)
        self.assertEqual(
            client.responses.create.call_args.kwargs["input"][0]["call_id"], "call_1"
        )

    def test_agent_session_creation(self):
        client = Mock()
        client.post.return_value = {"id": "session_test"}

        result = agents_tool_calling.run_agent(client, "agent-model", "Say hello.")

        self.assertEqual(result["id"], "session_test")
        client.post.assert_called_once_with(
            "/agents/sessions",
            cast_to=Dict[str, Any],
            options={"headers": {"OpenAI-Beta": "agents=v1"}},
            body={
                "agent": {
                    "model": "agent-model",
                    "instructions": "Answer the user's request concisely and safely.",
                },
                "environment": {"type": "none"},
                "input": "Say hello.",
            },
        )

    def test_fine_tuning_job_creation(self):
        client = Mock()
        model_customization.create_fine_tuning_job(
            client, "base-model", "file_training"
        )
        client.fine_tuning.jobs.create.assert_called_once_with(
            model="base-model", training_file="file_training"
        )

    def test_moderation_and_eval_read_methods(self):
        client = Mock()
        safety_governance.moderate_text(client, "sample")
        client.moderations.create.assert_called_once_with(
            model="omni-moderation-latest", input="sample"
        )
        evaluation_monitoring.get_eval(client, "eval_test")
        client.evals.retrieve.assert_called_once_with("eval_test")


if __name__ == "__main__":
    unittest.main()
