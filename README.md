# OpenAI API Tester

A Python command-line project for testing OpenAI model access and exercising
several API categories with standalone scripts.

## Setup

Use Python 3.9 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set `OPENAI_API_KEY` to your API key. Keep `.env` private; it
is excluded from Git. You can optionally change `OPENAI_MODEL`, or provide a
model with `--model`.

## Usage

List models available to your API key:

```bash
python openai_api_tester.py --list-models
```

Send a test prompt:

```bash
python openai_api_tester.py --prompt "Reply with a short greeting."
```

Choose a model explicitly:

```bash
python openai_api_tester.py --model gpt-4.1-mini --prompt "Explain APIs in one sentence."
```

## Test text generation

1. Set `OPENAI_API_KEY` in `.env`. If you have not created the file yet, copy
   `.env.example` to `.env`.
2. Activate your virtual environment and install dependencies if needed:

   ```bash
   source .venv/bin/activate
   python -m pip install -r requirements.txt
   ```

3. Send a test prompt with the Responses API:

   ```bash
   python text_generation.py --prompt "Explain what an API is in one sentence."
   ```

To select a model, include `--model MODEL_ID`. This sends a live API request
and uses your API quota.

## API category testers

Each script below exercises a separate API category. Run `python <script>
--help` for all options. These live API calls can use quota or create remote
resources; use test data and review the command before running it.

| Category | Script | Example |
| --- | --- | --- |
| Text Generation (Responses API) | `text_generation.py` | `python text_generation.py --prompt "Say hello."` |
| Vision/Image Generation (Images API) | `vision_image_generation.py` | `python vision_image_generation.py --image-url "https://example.com/image.jpg"` |
| Speech & Audio (Audio, Realtime) | `speech_audio.py` | `python speech_audio.py --realtime-secret` |
| Embeddings & RAG (Embeddings, Vector Stores) | `embeddings_rag.py` | `python embeddings_rag.py --embed-text "Sample text"` |
| Agents & Tool Calling (Agents API) | `agents_tool_calling.py` | `python agents_tool_calling.py --tool-call` |
| Model Customization (Fine-Tuning) | `model_customization.py` | `python model_customization.py --list-jobs` |
| Safety & Governance (Moderation, Administration) | `safety_governance.py` | `python safety_governance.py --text "Sample text to classify"` |
| Evaluation & Monitoring (Evals) | `evaluation_monitoring.py` | `python evaluation_monitoring.py` |

Image generation writes a PNG; speech generation writes an MP3. Vector store
uploads create a remote vector store, fine-tuning job creation can incur
training costs, and Agents API session creation and tool calls use model
quota. Administration audit logs require a separate `OPENAI_ADMIN_KEY` with
an Admin API key. The Evals API is read-only for existing users as of October
31, 2026 and is scheduled to shut down on November 30, 2026; this tester only
lists or retrieves existing evals.

Example additional operations:

```bash
python vision_image_generation.py --generate "A simple blue geometric pattern" --output sample.png
python speech_audio.py --transcribe recording.mp3
python speech_audio.py --speak "Hello from the speech API." --output greeting.mp3
python embeddings_rag.py --upload-file ./notes.txt
python embeddings_rag.py --query "What is in the notes?" --vector-store-id vs_...
python agents_tool_calling.py --input "Answer 2 + 2."
python model_customization.py --create-job file-... --model gpt-4.1-mini
OPENAI_ADMIN_KEY=... python safety_governance.py --audit-logs
python evaluation_monitoring.py --eval-id eval_...
```

After installing the project, the `openai-api-tester` command is also
available.

## Tests

The unit tests use mocked API responses and do not require an API key or make
network requests:

```bash
python -m unittest discover -s tests -v
```
