# LM Studio + Gemma-4 setup (remote backend)

How to serve **`google/gemma-4-12b-qat`** from **LM Studio** so the offline-agent
`remote` backend gets reliable tool calling. Without this setup, tool calls
degrade or fail outright. Everything here is external to the agent (LM Studio
owns the model + its chat template); it's captured here for reproducibility.

## Files

- `custom_pub_chat_template_gemma4.jinja` — the working chat template (a public
  fork of Gemma-4's template, plus two fixes for LM Studio's Jinja engine; see
  below). Source of truth.
- `apply-template.py` — installs the template (and the required custom fields)
  into an LM Studio `model.yaml`, with a backup and a byte-exact round-trip check.

## Why a custom template

Gemma-4's **stock** chat template mis-renders tool calls for agentic use: after
2–3 turns, `tool_call.arguments` collapses to `{}` (it also wraps JSON-string
arguments in extra braces and renders JSON `null` as bare `None`). The public
fork fixes those. It also keeps prior-turn reasoning visible across tool calls
(`preserve_thinking`), which the model card recommends for coding agents.

## Two fixes layered on top (for LM Studio's Jinja engine)

LM Studio's bundled llama.cpp Jinja engine is stricter than standard Jinja2, so
the upstream fork crashes on real-world tool schemas. Both are already applied in
the `.jinja` here:

1. **`is sequence` → `is iterable`** (4 sites). The engine has no `sequence`
   test, so any tool whose schema contains an array — e.g. a `cell_type`
   **enum** — failed the whole request with `Unknown test: sequence`. Each site
   is guarded by `is string`/`is mapping`, so `is iterable` is equivalent.
2. **`value['type'] | upper` → `value['type'] | default('string') | upper`**
   (plus `response['type'] | default('object')`). A JSON-schema **property with
   no `type`** left the value undefined, and the engine errors on
   `upper` of an undefined value (`Cannot apply filter "upper" to type:
   UndefinedValue`). Defaulting the type avoids the crash.

Symptom of either bug: a 400 "Error rendering prompt with jinja template …" and,
to the user, roughly half the tool calls failing (the half that offer an
affected tool).

## Apply it

```bash
# 1. Install the template into the model's LM Studio config (backs up model.yaml)
python docs/lmstudio-gemma4/apply-template.py
#    (defaults to ~/.lmstudio/hub/models/google/gemma-4-12b-qat/model.yaml;
#     pass a path to target a different one)

# 2. Reload so LM Studio re-reads model.yaml. Use a generous context so the
#    (growing) prompt + max_tokens always fits; set it as the model's default in
#    the GUI too, since a JIT/idle reload otherwise falls back to a small context.
lms unload --all
lms load google/gemma-4-12b-qat -c 16384 --ttl 86400
```

`apply-template.py` also adds the custom fields the template reads:
`enableThinking` → `enable_thinking` and `preserveThinking` → `preserve_thinking`
(both default **true**). The model's existing `llm.prediction.reasoning.parsing`
(start `"<|channel>thought"`, end `"<channel|>"`) is what surfaces reasoning as
`reasoning_content`; leave it in place.

## Matching agent config (`offline_agent.toml`)

```toml
[remote]
model = "google/gemma-4-12b-qat"
reasoning_effort = "none"   # see note

[sampling]
max_tokens = 4096           # ample with thinking off; leaves context for the prompt
```

**`reasoning_effort = "none"`**: with thinking on, Gemma-4-qat reasons 2k+ tokens
on many turns at `temperature = 0` and truncates tool calls. `"low"`/`"medium"`
are **not honored** by this model (≈ full), so `"none"` is the only effective
dial-down and the reliable choice. With thinking off, output is short, so
`max_tokens = 4096` is plenty and keeps more of the context window for the prompt.
The agent still carries `reasoning_content` back on tool-call turns, so if you
later set `reasoning_effort = ""` (thinking on), `preserve_thinking` works
end-to-end — but expect 2k+ extra tokens per turn and raise `max_tokens` (and the
loaded context) to match.

## Verify

```bash
# Should render (no 400) and return a tool call:
curl -s http://localhost:1234/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "google/gemma-4-12b-qat",
  "messages": [{"role":"user","content":"add a markdown cell to test.ipynb"}],
  "tools": [{"type":"function","function":{"name":"add_cell","description":"add",
    "parameters":{"type":"object","properties":{
      "file_path":{"type":"string"},
      "cell_type":{"type":"string","enum":["code","markdown","raw"]}},
    "required":["file_path"]}}}],
  "max_tokens": 64, "temperature": 0
}'
```
