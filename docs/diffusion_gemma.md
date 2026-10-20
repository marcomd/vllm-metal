# DiffusionGemma structured decisions

DiffusionGemma supports Jev-like System 1 decisions: yes/no probabilities,
choices, and scores over ordered levels. A structured read can score answer
slots in a supplied template after one denoising step.

The upstream example server provides `/v1/systemone` on port 8011 and sends
structured reads to vllm-metal on port 8000. It handles the template, token IDs,
and conversion from logprobs to answers.

## Start the servers

Follow [Installation](installation.md), then start the model:

```bash
VLLM_METAL_BACKEND=mlx vllm serve mlx-community/diffusiongemma-26B-A4B-it-4bit \
  --served-model-name dgemma --host 127.0.0.1 --port 8000 \
  --diffusion-config '{"canvas_length": 32}' --max-logprobs 32
```

In another terminal, activate the same Python environment and run the upstream
example from vLLM 0.31.0:

```bash
curl -fL \
  https://raw.githubusercontent.com/vllm-project/vllm/v0.31.0/examples/features/structured_diffusion/structured_server.py \
  -o structured_server.py
python structured_server.py \
  --upstream http://127.0.0.1:8000 --model dgemma \
  --tokenizer mlx-community/diffusiongemma-26B-A4B-it-4bit \
  --canvas 32 --canvas-step 32 --host 127.0.0.1 --port 8011
```

Keep `--canvas` and `--canvas-step` equal to the model server's `canvas_length`.
Metal uses synchronous scheduling, which requires full-width reads with
vLLM 0.31.0. Increase all three values together for larger answer templates.

## Make a decision

This request asks one yes/no question using one read and one denoising step:

```bash
curl -sS http://127.0.0.1:8011/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "dgemma",
    "state": {"ticket": "Everything is down and we have a demo at noon."},
    "questions": {
      "urgent": {
        "type": "noul",
        "instructions": "Does the customer need a reply within the hour?"
      }
    },
    "samples": 1,
    "steps": 1
  }'
```

Read `answers.urgent.noul` for the probability of "yes". The example server
constrains reads to the allowed labels by default. Setting `"samples": "auto"`
allows additional reads when the first result is uncertain.

For `choice` and `score` schemas, multiple questions, and the low-level request
options, see the [upstream structured-read guide](https://github.com/vllm-project/vllm/blob/v0.31.0/examples/features/structured_diffusion/README.md).

## Metal support

Structured reads use the MLX backend and accept text input only. Sample logprobs
and explicit label-token scoring are supported; prompt logprobs, full-vocabulary
logprobs (`logprobs=-1`), and LoRA are unsupported.
