# MHF Safeguard Classifier API Server

This folder contains the Python/Flask classifier API used by the **MHF Safeguard** XenForo plugin.

The XenForo plugin sends **one full cleaned message per submission** to this server. The server classifies the message using a trained model and returns a structured JSON response that the plugin can use to allow, log, moderate, or ask the user to revise the post.

## What changed from the original test server

The earlier `server.py` was a quick proof-of-concept. It loaded the model inside each request and passed the full JSON payload directly to the predictor.

The improved version now:

- loads the model once at startup;
- extracts the actual forum message from `payload["message"]`;
- supports a `/health` endpoint;
- supports optional Bearer-token authentication;
- returns JSON in the format expected by the XenForo plugin;
- normalises confidence scores to a 0--100 scale;
- maps model labels to plugin labels such as `method_or_action`, `ideation`, and `not_harmful`;
- recommends an action: `allow`, `review`, `moderate`, or `revise`.

## Expected request from XenForo

The plugin should send a POST request to:

```text
/api/classify
```

Example payload:

```json
{
  "platform": "xenforo",
  "source": "mhf_safeguard_plugin",
  "site_url": "https://www.mentalhealthforum.net",
  "context": {
    "content_type": "post",
    "content_id": 12345,
    "thread_id": 456,
    "node_id": 12,
    "user_id": 99,
    "username": "example_user",
    "title": "Thread title",
    "is_first_post": false
  },
  "message": "Cleaned full message text goes here.",
  "message_hash": "sha256_hash_here",
  "return_spans": true,
  "return_sentences": true,
  "sent_at": 1710000000
}
```

The most important field is:

```json
"message": "Cleaned full message text goes here."
```

The server extracts this field and sends only the message text to the model.

## Expected response to XenForo

The server returns JSON like this:

```json
{
  "risk_level": "high",
  "recommended_action": "moderate",
  "highest_label": "method_or_action",
  "highest_score": 94,
  "flagged_parts": [
    {
      "text": "flagged sentence or phrase",
      "label": "method_or_action",
      "score": 94,
      "start_offset": 0,
      "end_offset": 31
    }
  ]
}
```

The plugin expects these fields:

| Field | Meaning |
|---|---|
| `risk_level` | Overall risk level, e.g. `none`, `medium`, `high`, `critical` |
| `recommended_action` | Suggested action, e.g. `allow`, `review`, `moderate`, `revise` |
| `highest_label` | Highest-risk predicted label |
| `highest_score` | Confidence score from 0 to 100 |
| `flagged_parts` | Risky span, sentence, or phrase returned for moderation/revision |

## Label mapping

The server maps model labels into plugin-friendly labels.

| Model output example | Normalised plugin label |
|---|---|
| `method`, `suicide method`, `self-harm method`, `harmful method` | `method_or_action` |
| labels containing `method` or `action` | `method_or_action` |
| labels containing `ideation` | `ideation` |
| labels containing `not`, `none`, `safe`, or `non` | `not_harmful` |

You should adjust the `normalise_label()` function in `server.py` to match the exact labels produced by your final trained model.

## Action decision logic

The server recommends actions based on the label and confidence score.

| Condition | Risk level | Recommended action |
|---|---|---|
| `not_harmful` | `none` | `allow` |
| score >= revision threshold | `critical` | `revise` |
| score >= moderation threshold | `high` | `moderate` |
| risky label but lower score | `medium` | `review` |
| otherwise | `low` | `allow` |

Default thresholds:

```text
MHFS_MODERATE_THRESHOLD=85
MHFS_REVISE_THRESHOLD=95
```

## Running the server

From the `server` folder:

```bash
python server.py
```

By default, the server runs at:

```text
http://127.0.0.1:8000
```

The main classifier endpoint is:

```text
http://127.0.0.1:8000/api/classify
```

The health-check endpoint is:

```text
http://127.0.0.1:8000/health
```

## Environment variables

You can configure the server using environment variables.

| Variable | Purpose | Default |
|---|---|---|
| `MHFS_MODEL_PATH` | Path to saved ktrain predictor | `../content/bert_model_Suicide` |
| `MHFS_API_KEY` | Optional Bearer-token API key | empty |
| `MHFS_HOST` | Host/IP to bind server | `127.0.0.1` |
| `MHFS_PORT` | Port number | `8000` |
| `MHFS_MODERATE_THRESHOLD` | Score for moderation | `85` |
| `MHFS_REVISE_THRESHOLD` | Score for revision warning | `95` |

Example on Windows PowerShell:

```powershell
$env:MHFS_MODEL_PATH="../content/bert_model_Suicide"
$env:MHFS_API_KEY="your-secret-key"
$env:MHFS_HOST="127.0.0.1"
$env:MHFS_PORT="8000"
python server.py
```

Example on Linux/macOS:

```bash
export MHFS_MODEL_PATH="../content/bert_model_Suicide"
export MHFS_API_KEY="your-secret-key"
export MHFS_HOST="127.0.0.1"
export MHFS_PORT="8000"
python server.py
```

## XenForo plugin configuration

In the XenForo Admin Control Panel, set:

```text
Classifier API URL:
http://127.0.0.1:8000/api/classify

API key:
your-secret-key
```

For first testing, use:

```text
Action mode = log
Fail open = on
Store raw API response = on
Store cleaned message text = off
```

After the results look correct, switch to:

```text
Action mode = moderate
```

Only later test:

```text
Action mode = revise
```

## Testing with curl

Without API key:

```bash
curl -X POST http://127.0.0.1:8000/api/classify \
  -H "Content-Type: application/json" \
  -d '{"message":"I feel overwhelmed and need support."}'
```

With API key:

```bash
curl -X POST http://127.0.0.1:8000/api/classify \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secret-key" \
  -d '{"message":"I feel overwhelmed and need support."}'
```

## Current limitation

The current `build_flagged_parts()` function returns the whole message as the flagged span when the model predicts a risky label. This is acceptable for initial testing, but should be improved later.

Future improvements:

- split the message into sentences on the server;
- classify each sentence internally;
- return only the risky sentence or phrase;
- add keyword/span extraction;
- add model-attention or explainability-based span detection;
- add logging and audit trails for API-level requests.

## Production deployment notes

For production, do not use Flask's built-in development server directly. Use a proper WSGI deployment such as Gunicorn, uWSGI, Waitress, or a reverse proxy setup with Nginx/Apache.

Recommended production practices:

- use HTTPS;
- set `MHFS_API_KEY`;
- keep the model path outside public web folders;
- log errors, not full user messages unless strictly required;
- restrict server access by firewall or reverse proxy;
- monitor response time because XenForo waits for the classification result during post submission.
