#!/usr/bin/env python3
"""Install the working Gemma-4 chat template into an LM Studio model.yaml.

LM Studio reads per-model config (including a prompt-template override) from
``model.yaml`` under its hub. This injects the sibling
``custom_pub_chat_template_gemma4.jinja`` plus the custom fields the template
needs (enable_thinking / preserve_thinking), then verifies the template
round-trips byte-exact. Re-run after pulling a fresh model.yaml.

Usage:
    python apply-template.py [/path/to/model.yaml]

Default path:
    ~/.lmstudio/hub/models/google/gemma-4-12b-qat/model.yaml

After running, reload the model so LM Studio re-reads it, with enough context
for the agent's output budget (prompt + max_tokens):
    lms unload --all
    lms load google/gemma-4-12b-qat -c 16384
"""
from __future__ import annotations

import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "custom_pub_chat_template_gemma4.jinja")
DEFAULT_MODEL_YAML = os.path.expanduser(
    "~/.lmstudio/hub/models/google/gemma-4-12b-qat/model.yaml"
)


class _Literal(str):
    pass


def _literal_repr(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:str", str(data), style="|")


yaml.add_representer(_Literal, _literal_repr)


def main() -> int:
    model_yaml = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL_YAML
    if not os.path.exists(model_yaml):
        print(f"model.yaml not found: {model_yaml}", file=sys.stderr)
        return 1

    template_text = open(TEMPLATE).read()
    # Guard against re-introducing the LM Studio incompatibility (see README).
    assert " is sequence" not in template_text, "template still uses unsupported 'is sequence'"

    backup = model_yaml + ".bak"
    if not os.path.exists(backup):
        open(backup, "w").write(open(model_yaml).read())
        print(f"backed up original -> {backup}")

    data = yaml.safe_load(open(model_yaml))
    fields = [
        f for f in data["config"]["operation"]["fields"]
        if f.get("key") != "llm.prediction.promptTemplate"
    ]
    fields.append({
        "key": "llm.prediction.promptTemplate",
        "value": {
            "type": "jinja",
            "jinjaPromptTemplate": {
                "template": _Literal(template_text),
                "bosToken": "<bos>",
                "eosToken": "<eos>",
            },
            "stopStrings": [],
        },
    })
    data["config"]["operation"]["fields"] = fields

    cf = data.get("customFields") or []
    have = {c.get("key") for c in cf}
    if "enableThinking" not in have:
        cf.append({
            "key": "enableThinking", "displayName": "Enable Thinking",
            "description": "Controls whether the model will think before replying",
            "type": "boolean", "defaultValue": True,
            "effects": [{"type": "setJinjaVariable", "variable": "enable_thinking"}],
        })
    if "preserveThinking" not in have:
        cf.append({
            "key": "preserveThinking", "displayName": "Preserve Thinking",
            "description": "Preserve reasoning content in all prior assistant turns instead of only the most recent one",
            "type": "boolean", "defaultValue": True,
            "effects": [{"type": "setJinjaVariable", "variable": "preserve_thinking"}],
        })
    data["customFields"] = cf

    open(model_yaml, "w").write(
        yaml.dump(data, sort_keys=False, allow_unicode=True, width=10**9, default_flow_style=False)
    )

    reloaded = yaml.safe_load(open(model_yaml))
    rt = next(
        f["value"]["jinjaPromptTemplate"]["template"]
        for f in reloaded["config"]["operation"]["fields"]
        if f["key"] == "llm.prediction.promptTemplate"
    )
    ok = rt == template_text
    print(f"template round-trip byte-exact: {ok}")
    print(f"customFields: {[c['key'] for c in reloaded['customFields']]}")
    print("Done. Now: lms unload --all && lms load <model> -c 16384")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
