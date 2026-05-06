# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""
Tests for the OpenAI /v1/chat/completions client-side tool pass-through.

Covers:
- ChatCompletionRequest accepts standard OpenAI `tools` / `tool_choice` / `stop`.
- ChatMessage accepts role="tool" with `tool_call_id` and role="assistant"
  with `content: None` + `tool_calls`.
- ChatCompletionRequest carries unknown fields via `extra="allow"`.
- anthropic_tool_choice_to_openai() covers all four Anthropic shapes.
- _build_passthrough_payload() honors a caller-supplied tool_choice and
  defaults to "auto" when unset.
- _friendly_error() maps httpx transport errors to a "Lost connection"
  message so passthrough failures are legible instead of bare 500s.

No running server or GPU required.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

_backend = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _backend)

import httpx
import pytest
from pydantic import ValidationError
import routes.inference as inference_routes

from models.inference import (
    ChatCompletionRequest,
    ChatMessage,
)
from core.inference.anthropic_compat import (
    anthropic_tool_choice_to_openai,
)
from routes.inference import (
    _build_passthrough_payload,
    _build_openai_upstream_body,
    _openai_upstream_tool_loop_events,
    _friendly_error,
    _llm_upstream_completions_fallback_enabled,
    _llm_upstream_embeddings_fallback_enabled,
    _looks_like_history_intent,
    _normalize_openai_base_url,
    _resolve_llm_upstream_model,
    _upstream_empty_assistant_fallback,
    _upstream_builtin_tools_for_payload,
    _upstream_server_tools_enabled,
    _upstream_tool_use_nudge,
    _route_wiki_llm_stub,
    _wants_wiki_structured_json,
    _wiki_llm_available,
)


# =====================================================================
# ChatMessage — tool role, tool_calls, optional content
# =====================================================================


class TestChatMessageToolRoles:
    def test_tool_role_with_tool_call_id(self):
        msg = ChatMessage(
            role = "tool",
            tool_call_id = "call_abc123",
            content = '{"temperature": 72}',
        )
        assert msg.role == "tool"
        assert msg.tool_call_id == "call_abc123"
        assert msg.content == '{"temperature": 72}'

    def test_tool_role_with_name(self):
        msg = ChatMessage(
            role = "tool",
            tool_call_id = "call_abc123",
            name = "get_weather",
            content = '{"temperature": 72}',
        )
        assert msg.name == "get_weather"

    def test_assistant_with_tool_calls_no_content(self):
        msg = ChatMessage(
            role = "assistant",
            content = None,
            tool_calls = [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "Paris"}',
                    },
                }
            ],
        )
        assert msg.role == "assistant"
        assert msg.content is None
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0]["function"]["name"] == "get_weather"

    def test_assistant_with_content_and_tool_calls(self):
        msg = ChatMessage(
            role = "assistant",
            content = "Let me check the weather.",
            tool_calls = [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{}"},
                }
            ],
        )
        assert msg.content == "Let me check the weather."
        assert msg.tool_calls[0]["id"] == "call_1"

    def test_plain_user_message_still_works(self):
        msg = ChatMessage(role = "user", content = "Hello")
        assert msg.role == "user"
        assert msg.tool_call_id is None
        assert msg.tool_calls is None
        assert msg.name is None

    def test_invalid_role_rejected(self):
        with pytest.raises(ValidationError):
            ChatMessage(role = "function", content = "x")

    def test_content_absent_on_assistant_tool_call_defaults_to_none(self):
        # Assistant messages that carry only tool_calls are the one
        # documented case where `content=None` is permitted.
        msg = ChatMessage(
            role = "assistant",
            tool_calls = [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "f", "arguments": "{}"},
                }
            ],
        )
        assert msg.content is None

    def test_tool_role_missing_tool_call_id_rejected(self):
        # Per OpenAI spec, role="tool" messages must carry tool_call_id so
        # upstream backends can associate the result with its prior call.
        # Pin the boundary-level rejection so a malformed tool-result
        # message never reaches the passthrough path.
        with pytest.raises(ValidationError) as exc_info:
            ChatMessage(role = "tool", content = '{"temperature": 72}')
        assert "tool_call_id" in str(exc_info.value)

    def test_tool_role_empty_tool_call_id_rejected(self):
        with pytest.raises(ValidationError):
            ChatMessage(
                role = "tool",
                tool_call_id = "",
                content = '{"temperature": 72}',
            )

    # ── Role-aware content requirements ────────────────────────────

    def test_user_empty_content_rejected(self):
        with pytest.raises(ValidationError):
            ChatMessage(role = "user", content = "")

    def test_system_empty_content_rejected(self):
        with pytest.raises(ValidationError):
            ChatMessage(role = "system", content = "")

    def test_user_empty_list_content_rejected(self):
        with pytest.raises(ValidationError):
            ChatMessage(role = "user", content = [])

    def test_tool_empty_content_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            ChatMessage(role = "tool", tool_call_id = "call_1", content = "")
        assert "content" in str(exc_info.value)

    def test_assistant_without_content_or_tool_calls_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            ChatMessage(role = "assistant")
        assert "content" in str(exc_info.value) or "tool_calls" in str(exc_info.value)

    # ── Role-constrained tool-call metadata ────────────────────────

    def test_tool_calls_on_user_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            ChatMessage(
                role = "user",
                content = "Hi",
                tool_calls = [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "f", "arguments": "{}"},
                    }
                ],
            )
        assert "tool_calls" in str(exc_info.value)

    def test_tool_call_id_on_user_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            ChatMessage(role = "user", content = "Hi", tool_call_id = "call_1")
        assert "tool_call_id" in str(exc_info.value)

    def test_name_on_user_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            ChatMessage(role = "user", content = "Hi", name = "get_weather")
        assert "name" in str(exc_info.value)


# =====================================================================
# ChatCompletionRequest — standard OpenAI tool fields
# =====================================================================


class TestChatCompletionRequestToolFields:
    def _make(self, **kwargs):
        base = {"messages": [{"role": "user", "content": "Hi"}]}
        base.update(kwargs)
        return ChatCompletionRequest(**base)

    def test_tools_parses(self):
        req = self._make(
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Return the weather in a city",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }
            ],
        )
        assert req.tools is not None
        assert len(req.tools) == 1
        assert req.tools[0]["function"]["name"] == "get_weather"

    def test_tool_choice_string_auto(self):
        assert self._make(tool_choice = "auto").tool_choice == "auto"

    def test_tool_choice_string_required(self):
        assert self._make(tool_choice = "required").tool_choice == "required"

    def test_tool_choice_string_none(self):
        assert self._make(tool_choice = "none").tool_choice == "none"

    def test_tool_choice_named_function(self):
        tc = {"type": "function", "function": {"name": "get_weather"}}
        assert self._make(tool_choice = tc).tool_choice == tc

    def test_stop_string(self):
        assert self._make(stop = "\nUser:").stop == "\nUser:"

    def test_stop_list(self):
        assert self._make(stop = ["\nUser:", "\nAssistant:"]).stop == [
            "\nUser:",
            "\nAssistant:",
        ]

    def test_tools_default_none(self):
        req = self._make()
        assert req.tools is None
        assert req.tool_choice is None
        assert req.stop is None

    def test_extra_fields_accepted(self):
        # `frequency_penalty`, `seed`, `response_format` are not yet
        # explicitly declared but must survive Pydantic parsing now that
        # extra="allow" is set.
        req = self._make(
            frequency_penalty = 0.5,
            seed = 42,
            response_format = {"type": "json_object"},
        )
        # Extras land in model_extra
        assert req.model_extra is not None
        assert req.model_extra.get("frequency_penalty") == 0.5
        assert req.model_extra.get("seed") == 42
        assert req.model_extra.get("response_format") == {"type": "json_object"}

    def test_unsloth_extensions_still_work(self):
        req = self._make(
            enable_tools = True,
            enabled_tools = ["web_search", "python"],
            session_id = "abc",
        )
        assert req.enable_tools is True
        assert req.enabled_tools == ["web_search", "python"]
        assert req.session_id == "abc"

    def test_stream_defaults_false_matching_openai_spec(self):
        # OpenAI's /v1/chat/completions spec defaults `stream` to false.
        # Studio previously defaulted to true, which broke naive curl
        # clients that omit `stream` (they expect a JSON blob, got SSE).
        # Pin the corrected default so it can't silently regress.
        req = self._make()
        assert req.stream is False

    def test_multiturn_tool_loop_messages(self):
        req = ChatCompletionRequest(
            messages = [
                {"role": "user", "content": "What's the weather in Paris?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "Paris"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": '{"temperature": 14, "unit": "celsius"}',
                },
            ],
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        )
        assert len(req.messages) == 3
        assert req.messages[1].role == "assistant"
        assert req.messages[1].content is None
        assert req.messages[1].tool_calls[0]["id"] == "call_1"
        assert req.messages[2].role == "tool"
        assert req.messages[2].tool_call_id == "call_1"


# =====================================================================
# anthropic_tool_choice_to_openai — pure translation helper
# =====================================================================


class TestAnthropicToolChoiceToOpenAI:
    def test_auto(self):
        assert anthropic_tool_choice_to_openai({"type": "auto"}) == "auto"

    def test_any_becomes_required(self):
        assert anthropic_tool_choice_to_openai({"type": "any"}) == "required"

    def test_none(self):
        assert anthropic_tool_choice_to_openai({"type": "none"}) == "none"

    def test_tool_named(self):
        result = anthropic_tool_choice_to_openai(
            {"type": "tool", "name": "get_weather"}
        )
        assert result == {
            "type": "function",
            "function": {"name": "get_weather"},
        }

    def test_tool_missing_name_returns_none(self):
        assert anthropic_tool_choice_to_openai({"type": "tool"}) is None

    def test_none_input_returns_none(self):
        assert anthropic_tool_choice_to_openai(None) is None

    def test_unrecognized_shape_returns_none(self):
        assert anthropic_tool_choice_to_openai({"type": "wibble"}) is None
        assert anthropic_tool_choice_to_openai("auto") is None
        assert anthropic_tool_choice_to_openai(42) is None


# =====================================================================
# _build_passthrough_payload — tool_choice propagation
# =====================================================================


class TestBuildPassthroughPayloadToolChoice:
    def _args(self):
        return dict(
            openai_messages = [{"role": "user", "content": "Hi"}],
            openai_tools = [
                {
                    "type": "function",
                    "function": {"name": "f", "parameters": {"type": "object"}},
                }
            ],
            temperature = 0.6,
            top_p = 0.95,
            top_k = 20,
            max_tokens = 128,
            stream = False,
        )

    def test_default_tool_choice_is_auto(self):
        body = _build_passthrough_payload(**self._args())
        assert body["tool_choice"] == "auto"

    def test_override_tool_choice_required(self):
        body = _build_passthrough_payload(**self._args(), tool_choice = "required")
        assert body["tool_choice"] == "required"

    def test_override_tool_choice_none(self):
        body = _build_passthrough_payload(**self._args(), tool_choice = "none")
        assert body["tool_choice"] == "none"

    def test_override_tool_choice_named_function(self):
        tc = {"type": "function", "function": {"name": "f"}}
        body = _build_passthrough_payload(**self._args(), tool_choice = tc)
        assert body["tool_choice"] == tc

    def test_stream_adds_include_usage(self):
        args = self._args()
        args["stream"] = True
        body = _build_passthrough_payload(**args)
        assert body.get("stream_options") == {"include_usage": True}

    def test_repetition_penalty_renamed(self):
        body = _build_passthrough_payload(**self._args(), repetition_penalty = 1.1)
        assert body.get("repeat_penalty") == 1.1
        assert "repetition_penalty" not in body


# =====================================================================
# OpenAI upstream helpers
# =====================================================================


class TestOpenAIUpstreamHelpers:
    def test_normalize_openai_base_url_appends_v1(self):
        assert (
            _normalize_openai_base_url("https://integrate.api.nvidia.com")
            == "https://integrate.api.nvidia.com/v1"
        )

    def test_normalize_openai_base_url_keeps_existing_v1(self):
        assert (
            _normalize_openai_base_url("https://integrate.api.nvidia.com/v1/")
            == "https://integrate.api.nvidia.com/v1"
        )

    def test_resolve_upstream_model_uses_env_default_for_aliases(self, monkeypatch):
        monkeypatch.setattr(
            inference_routes,
            "_LLM_UPSTREAM_MODEL",
            "meta/llama-3.1-8b-instruct",
        )
        assert _resolve_llm_upstream_model("default") == "meta/llama-3.1-8b-instruct"
        assert _resolve_llm_upstream_model("current") == "meta/llama-3.1-8b-instruct"
        assert _resolve_llm_upstream_model("custom/model") == "custom/model"

    def test_build_openai_upstream_body_strips_unsloth_only_fields(self):
        req = ChatCompletionRequest(
            model = "default",
            messages = [{"role": "user", "content": "hello"}],
            stream = False,
            use_upstream = True,
            upstream_auto_stream_fallback = True,
            enable_tools = True,
            enabled_tools = ["python"],
            session_id = "abc123",
            top_k = 40,
            min_p = 0.05,
            repetition_penalty = 1.1,
            frequency_penalty = 0.4,
        )

        body = _build_openai_upstream_body(
            req,
            "meta/llama-3.1-8b-instruct",
        )

        assert body["model"] == "meta/llama-3.1-8b-instruct"
        assert body["messages"] == [{"role": "user", "content": "hello"}]
        assert "enable_tools" not in body
        assert "enabled_tools" not in body
        assert "session_id" not in body
        assert "use_upstream" not in body
        assert "upstream_auto_stream_fallback" not in body
        assert "top_k" not in body
        assert "min_p" not in body
        assert "repetition_penalty" not in body
        assert body.get("frequency_penalty") == 0.4

    def test_build_openai_upstream_body_forwards_thinking_for_nim_auto(self, monkeypatch):
        monkeypatch.setattr(inference_routes, "_LLM_UPSTREAM_FORWARD_THINKING", "auto")
        monkeypatch.setattr(
            inference_routes,
            "_LLM_UPSTREAM_BASE_URL",
            "https://integrate.api.nvidia.com/v1",
        )

        req = ChatCompletionRequest(
            model = "default",
            messages = [{"role": "user", "content": "hello"}],
            stream = False,
            enable_thinking = True,
            chat_template_kwargs = {"existing": "ok"},
        )

        body = _build_openai_upstream_body(req, "google/gemma-4-31b-it")

        assert body.get("chat_template_kwargs") == {
            "existing": "ok",
            "enable_thinking": True,
        }

    def test_build_openai_upstream_body_merges_extra_body_and_reasoning_effort(self):
        req = ChatCompletionRequest(
            model = "default",
            messages = [{"role": "user", "content": "hello"}],
            stream = False,
            enable_thinking = True,
            reasoning_effort = "high",
            extra_body = {
                "thinking": {"type": "enabled"},
                "chat_template_kwargs": {"foo": "bar"},
            },
        )

        body = _build_openai_upstream_body(req, "deepseek/deepseek-chat")

        assert "extra_body" not in body
        assert body.get("reasoning_effort") == "high"
        assert body.get("thinking") == {"type": "enabled"}
        assert body.get("chat_template_kwargs", {}).get("foo") == "bar"
        assert body.get("chat_template_kwargs", {}).get("enable_thinking") is True

    def test_build_openai_upstream_body_uses_deepseek_thinking_toggle_format(
        self,
        monkeypatch,
    ):
        monkeypatch.setattr(
            inference_routes,
            "_LLM_UPSTREAM_BASE_URL",
            "https://api.deepseek.com/v1",
        )

        req = ChatCompletionRequest(
            model = "default",
            messages = [{"role": "user", "content": "hello"}],
            stream = False,
            enable_thinking = False,
            chat_template_kwargs = {"foo": "bar"},
        )

        body = _build_openai_upstream_body(req, "deepseek-v4-pro")

        assert body.get("thinking") == {"type": "disabled"}
        assert body.get("chat_template_kwargs", {}).get("foo") == "bar"
        assert body.get("chat_template_kwargs", {}).get("enable_thinking") is None

    def test_build_openai_upstream_body_reasoning_content_provider_guard(self, monkeypatch):
        base_req = ChatCompletionRequest(
            model = "default",
            messages = [
                {
                    "role": "assistant",
                    "content": "Final answer",
                    "reasoning_content": "Internal reasoning",
                },
                {"role": "user", "content": "Follow-up"},
            ],
            stream = False,
        )

        monkeypatch.setattr(
            inference_routes,
            "_LLM_UPSTREAM_BASE_URL",
            "https://integrate.api.nvidia.com/v1",
        )
        non_deepseek_body = _build_openai_upstream_body(base_req, "provider/model")
        assert "reasoning_content" not in non_deepseek_body["messages"][0]

        monkeypatch.setattr(
            inference_routes,
            "_LLM_UPSTREAM_BASE_URL",
            "https://api.deepseek.com/v1",
        )
        deepseek_body = _build_openai_upstream_body(base_req, "deepseek-v4-pro")
        assert deepseek_body["messages"][0].get("reasoning_content") == "Internal reasoning"

    def test_upstream_tool_nudge_skips_tools_for_greetings(self):
        nudge = _upstream_tool_use_nudge(
            "meta/llama-3.3-70b-instruct",
            [
                {
                    "type": "function",
                    "function": {"name": "web_search"},
                }
            ],
        )
        assert "greetings" in nudge.lower()
        assert "respond directly without tools" in nudge.lower()

    def test_upstream_empty_assistant_fallback_includes_last_tool_output(self):
        fallback = _upstream_empty_assistant_fallback(
            "web_search",
            "Title: Example\nSnippet: hello world",
        )
        assert "calling web_search" in fallback.lower()
        assert "hello world" in fallback

    def test_upstream_empty_assistant_fallback_generic_message(self):
        fallback = _upstream_empty_assistant_fallback("", "")
        assert "could not generate a final answer" in fallback.lower()

    def test_upstream_builtin_tools_respects_enabled_tools(self):
        req = ChatCompletionRequest(
            model = "default",
            messages = [{"role": "user", "content": "hello"}],
            stream = True,
            enable_tools = True,
            enabled_tools = ["python"],
        )
        tools = _upstream_builtin_tools_for_payload(req)
        names = [t.get("function", {}).get("name") for t in tools]
        assert names == ["python"]

    def test_upstream_server_tools_enabled_with_default_iterations(self):
        req = ChatCompletionRequest(
            model = "default",
            messages = [{"role": "user", "content": "hello"}],
            stream = True,
            enable_tools = True,
        )
        assert _upstream_server_tools_enabled(req) is True

    def test_upstream_server_tools_disabled_when_iterations_zero(self):
        req = ChatCompletionRequest(
            model = "default",
            messages = [{"role": "user", "content": "hello"}],
            stream = True,
            enable_tools = True,
            max_tool_calls_per_message = 0,
        )
        assert _upstream_server_tools_enabled(req) is False

    def test_wiki_llm_available_true_with_upstream_only(self, monkeypatch):
        class _DummyLlama:
            is_loaded = False

        class _DummyBackend:
            active_model_name = None

        monkeypatch.setattr(
            inference_routes,
            "get_llama_cpp_backend",
            lambda: _DummyLlama(),
        )
        monkeypatch.setattr(
            inference_routes,
            "get_inference_backend",
            lambda: _DummyBackend(),
        )
        monkeypatch.setattr(inference_routes, "_llm_upstream_enabled", lambda: True)

        assert _wiki_llm_available() is True

    def test_route_wiki_llm_stub_accepts_content_parts_from_upstream(self, monkeypatch):
        class _DummyLlama:
            is_loaded = False

        class _DummyBackend:
            active_model_name = None

        class _FakeResponse:
            status_code = 200
            text = "{\"ok\":true}"

            def json(self):
                return {
                    "choices": [
                        {
                            "message": {
                                "content": [
                                    {
                                        "type": "text",
                                        "text": '{"summary":"ok","entities":[],"concepts":[]}',
                                    }
                                ]
                            }
                        }
                    ]
                }

        class _FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def post(self, url, json, headers):
                return _FakeResponse()

        monkeypatch.setattr(inference_routes, "get_llama_cpp_backend", lambda: _DummyLlama())
        monkeypatch.setattr(inference_routes, "get_inference_backend", lambda: _DummyBackend())
        monkeypatch.setattr(inference_routes, "_llm_upstream_enabled", lambda: True)
        monkeypatch.setattr(
            inference_routes,
            "_llm_upstream_base_url",
            lambda: "https://example.test/v1",
        )
        monkeypatch.setattr(
            inference_routes,
            "_resolve_llm_upstream_model",
            lambda _requested: "dummy/model",
        )
        monkeypatch.setattr(inference_routes, "_llm_upstream_headers", lambda: {})
        monkeypatch.setattr(inference_routes.httpx, "Client", _FakeClient)

        out = _route_wiki_llm_stub("Extract structured knowledge from the source.")
        assert out == '{"summary":"ok","entities":[],"concepts":[]}'

    def test_route_wiki_llm_stub_prefers_upstream_for_structured_json_prompts(
        self,
        monkeypatch,
    ):
        class _DummyLlama:
            is_loaded = True

            def generate_chat_completion(self, **kwargs):
                raise AssertionError("Local GGUF backend should not run for strict JSON extraction")

        class _DummyBackend:
            active_model_name = "local-model"

            def generate_chat_response(self, **kwargs):
                raise AssertionError(
                    "Transformer backend should not run for strict JSON extraction"
                )

        observed_body: dict[str, object] = {}

        class _FakeResponse:
            status_code = 200
            text = "{\"ok\":true}"

            def json(self):
                return {
                    "choices": [
                        {
                            "message": {
                                "content": '{"summary":"upstream","entities":[],"concepts":[]}'
                            }
                        }
                    ]
                }

        class _FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def post(self, url, json, headers):
                observed_body.update(json)
                return _FakeResponse()

        monkeypatch.setattr(inference_routes, "get_llama_cpp_backend", lambda: _DummyLlama())
        monkeypatch.setattr(inference_routes, "get_inference_backend", lambda: _DummyBackend())
        monkeypatch.setattr(inference_routes, "_llm_upstream_enabled", lambda: True)
        monkeypatch.setattr(inference_routes, "_WIKI_LLM_THINKING_ENABLED", True)
        monkeypatch.setattr(inference_routes, "_WIKI_LLM_REASONING_STYLE", "reasoning_effort")
        monkeypatch.setattr(inference_routes, "_WIKI_LLM_REASONING_EFFORT", "high")
        monkeypatch.setattr(inference_routes, "_WIKI_LLM_PRESERVE_THINKING", False)
        monkeypatch.setattr(
            inference_routes,
            "_llm_upstream_base_url",
            lambda: "https://example.test/v1",
        )
        monkeypatch.setattr(
            inference_routes,
            "_resolve_llm_upstream_model",
            lambda _requested: "dummy/model",
        )
        monkeypatch.setattr(inference_routes, "_llm_upstream_headers", lambda: {})
        monkeypatch.setattr(inference_routes.httpx, "Client", _FakeClient)

        out = _route_wiki_llm_stub(
            "Return strict JSON with keys: summary, entities, concepts."
        )
        assert out == '{"summary":"upstream","entities":[],"concepts":[]}'
        assert observed_body.get("response_format") == {"type": "json_object"}
        assert "enable_thinking" not in observed_body
        assert observed_body.get("reasoning_effort") == "high"
        assert "chat_template_kwargs" not in observed_body
        assert int(observed_body.get("max_tokens") or 0) >= 2000

    def test_wants_wiki_structured_json_is_case_insensitive_and_phrase_flexible(self):
        assert _wants_wiki_structured_json("Return strict JSON with keys: summary")
        assert _wants_wiki_structured_json("return STRICT json only with this schema:")
        assert _wants_wiki_structured_json("You are a JSON repair assistant.")
        assert not _wants_wiki_structured_json("Summarize this source in bullets.")

    def test_route_wiki_llm_stub_treats_schema_phrase_as_structured_json(
        self,
        monkeypatch,
    ):
        observed_body: dict[str, object] = {}

        class _DummyLlama:
            is_loaded = True

            def generate_chat_completion(self, **kwargs):
                raise AssertionError("Local GGUF backend should not run for strict JSON extraction")

        class _DummyBackend:
            active_model_name = "local-model"

            def generate_chat_response(self, **kwargs):
                raise AssertionError(
                    "Transformer backend should not run for strict JSON extraction"
                )

        class _FakeResponse:
            status_code = 200
            text = "{\"ok\":true}"

            def json(self):
                return {
                    "choices": [
                        {
                            "message": {
                                "content": '{"keep_missing":[],"related_to_existing":[],"reject":[]}'
                            }
                        }
                    ]
                }

        class _FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def post(self, url, json, headers):
                observed_body.update(json)
                return _FakeResponse()

        monkeypatch.setattr(inference_routes, "get_llama_cpp_backend", lambda: _DummyLlama())
        monkeypatch.setattr(inference_routes, "get_inference_backend", lambda: _DummyBackend())
        monkeypatch.setattr(inference_routes, "_llm_upstream_enabled", lambda: True)
        monkeypatch.setattr(inference_routes, "_WIKI_LLM_THINKING_ENABLED", True)
        monkeypatch.setattr(inference_routes, "_WIKI_LLM_REASONING_STYLE", "reasoning_effort")
        monkeypatch.setattr(inference_routes, "_WIKI_LLM_REASONING_EFFORT", "high")
        monkeypatch.setattr(inference_routes, "_WIKI_LLM_PRESERVE_THINKING", False)
        monkeypatch.setattr(
            inference_routes,
            "_llm_upstream_base_url",
            lambda: "https://example.test/v1",
        )
        monkeypatch.setattr(
            inference_routes,
            "_resolve_llm_upstream_model",
            lambda _requested: "dummy/model",
        )
        monkeypatch.setattr(inference_routes, "_llm_upstream_headers", lambda: {})
        monkeypatch.setattr(inference_routes.httpx, "Client", _FakeClient)

        out = _route_wiki_llm_stub(
            "Return strict JSON only with this schema: keep_missing, related_to_existing, reject."
        )
        assert out == '{"keep_missing":[],"related_to_existing":[],"reject":[]}'
        assert observed_body.get("response_format") == {"type": "json_object"}
        assert int(observed_body.get("max_tokens") or 0) >= 2000

    def test_route_wiki_llm_stub_prefers_upstream_for_non_structured_prompts(
        self,
        monkeypatch,
    ):
        class _DummyLlama:
            is_loaded = True

            def generate_chat_completion(self, **kwargs):
                raise AssertionError("Local GGUF backend should not run when upstream preference is enabled")

        class _DummyBackend:
            active_model_name = "local-model"

            def generate_chat_response(self, **kwargs):
                raise AssertionError(
                    "Transformer backend should not run when upstream preference is enabled"
                )

        observed_body: dict[str, object] = {}

        class _FakeResponse:
            status_code = 200
            text = "{\"ok\":true}"

            def json(self):
                return {
                    "choices": [
                        {
                            "message": {
                                "content": "upstream non-structured output"
                            }
                        }
                    ]
                }

        class _FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def post(self, url, json, headers):
                observed_body.update(json)
                return _FakeResponse()

        monkeypatch.setattr(inference_routes, "get_llama_cpp_backend", lambda: _DummyLlama())
        monkeypatch.setattr(inference_routes, "get_inference_backend", lambda: _DummyBackend())
        monkeypatch.setattr(inference_routes, "_llm_upstream_enabled", lambda: True)
        monkeypatch.setattr(inference_routes, "_WIKI_LLM_PREFER_UPSTREAM", True)
        monkeypatch.setattr(inference_routes, "_WIKI_LLM_THINKING_ENABLED", True)
        monkeypatch.setattr(inference_routes, "_WIKI_LLM_REASONING_STYLE", "reasoning_effort")
        monkeypatch.setattr(inference_routes, "_WIKI_LLM_REASONING_EFFORT", "high")
        monkeypatch.setattr(inference_routes, "_WIKI_LLM_PRESERVE_THINKING", False)
        monkeypatch.setattr(
            inference_routes,
            "_llm_upstream_base_url",
            lambda: "https://example.test/v1",
        )
        monkeypatch.setattr(
            inference_routes,
            "_resolve_llm_upstream_model",
            lambda _requested: "dummy/model",
        )
        monkeypatch.setattr(inference_routes, "_llm_upstream_headers", lambda: {})
        monkeypatch.setattr(inference_routes.httpx, "Client", _FakeClient)

        out = _route_wiki_llm_stub("Summarize this source and list key concepts.")
        assert out == "upstream non-structured output"
        assert observed_body.get("reasoning_effort") == "high"

    def test_route_wiki_llm_stub_can_disable_upstream_preference(
        self,
        monkeypatch,
    ):
        class _DummyLlama:
            is_loaded = True

            def generate_chat_completion(self, **kwargs):
                return ["local output"]

        class _DummyBackend:
            active_model_name = None

        monkeypatch.setattr(inference_routes, "get_llama_cpp_backend", lambda: _DummyLlama())
        monkeypatch.setattr(inference_routes, "get_inference_backend", lambda: _DummyBackend())
        monkeypatch.setattr(inference_routes, "_llm_upstream_enabled", lambda: True)
        monkeypatch.setattr(inference_routes, "_WIKI_LLM_PREFER_UPSTREAM", False)

        out = _route_wiki_llm_stub("Summarize this source and list key concepts.")
        assert out == "local output"

    def test_route_wiki_llm_stub_applies_enable_thinking_style_for_local_gguf(
        self,
        monkeypatch,
    ):
        observed_kwargs: dict[str, object] = {}

        class _DummyLlama:
            is_loaded = True

            def generate_chat_completion(self, **kwargs):
                observed_kwargs.update(kwargs)
                return ["local thinking output"]

        class _DummyBackend:
            active_model_name = None

        monkeypatch.setattr(inference_routes, "get_llama_cpp_backend", lambda: _DummyLlama())
        monkeypatch.setattr(inference_routes, "get_inference_backend", lambda: _DummyBackend())
        monkeypatch.setattr(inference_routes, "_WIKI_LLM_PREFER_UPSTREAM", False)
        monkeypatch.setattr(inference_routes, "_llm_upstream_enabled", lambda: False)
        monkeypatch.setattr(inference_routes, "_WIKI_LLM_THINKING_ENABLED", True)
        monkeypatch.setattr(inference_routes, "_WIKI_LLM_REASONING_STYLE", "enable_thinking")
        monkeypatch.setattr(inference_routes, "_WIKI_LLM_REASONING_EFFORT", "high")
        monkeypatch.setattr(inference_routes, "_WIKI_LLM_PRESERVE_THINKING", True)

        out = _route_wiki_llm_stub("Summarize this source and list key concepts.")
        assert out == "local thinking output"
        assert observed_kwargs.get("enable_thinking") is True
        assert observed_kwargs.get("preserve_thinking") is True
        assert "reasoning_effort" not in observed_kwargs

    def test_route_wiki_llm_stub_retries_upstream_without_reasoning_fields(
        self,
        monkeypatch,
    ):
        call_bodies: list[dict[str, object]] = []

        class _DummyLlama:
            is_loaded = False

        class _DummyBackend:
            active_model_name = None

        class _FakeResponse:
            def __init__(self, status_code: int, body: dict[str, object]):
                self.status_code = status_code
                self.text = "{\"ok\":true}" if status_code == 200 else "{\"error\":\"bad_request\"}"
                self._body = body

            def json(self):
                return self._body

        class _FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def post(self, url, json, headers):
                body_copy = dict(json)
                call_bodies.append(body_copy)
                if "reasoning_effort" in body_copy:
                    return _FakeResponse(400, {"error": "unsupported_field"})
                return _FakeResponse(
                    200,
                    {
                        "choices": [
                            {
                                "message": {
                                    "content": "upstream compatibility output"
                                }
                            }
                        ]
                    },
                )

        monkeypatch.setattr(inference_routes, "get_llama_cpp_backend", lambda: _DummyLlama())
        monkeypatch.setattr(inference_routes, "get_inference_backend", lambda: _DummyBackend())
        monkeypatch.setattr(inference_routes, "_WIKI_LLM_PREFER_UPSTREAM", True)
        monkeypatch.setattr(inference_routes, "_llm_upstream_enabled", lambda: True)
        monkeypatch.setattr(inference_routes, "_WIKI_LLM_THINKING_ENABLED", True)
        monkeypatch.setattr(inference_routes, "_WIKI_LLM_REASONING_STYLE", "reasoning_effort")
        monkeypatch.setattr(inference_routes, "_WIKI_LLM_REASONING_EFFORT", "high")
        monkeypatch.setattr(inference_routes, "_WIKI_LLM_PRESERVE_THINKING", False)
        monkeypatch.setattr(
            inference_routes,
            "_llm_upstream_base_url",
            lambda: "https://example.test/v1",
        )
        monkeypatch.setattr(
            inference_routes,
            "_resolve_llm_upstream_model",
            lambda _requested: "dummy/model",
        )
        monkeypatch.setattr(inference_routes, "_llm_upstream_headers", lambda: {})
        monkeypatch.setattr(inference_routes.httpx, "Client", _FakeClient)

        out = _route_wiki_llm_stub("Summarize this source and list key concepts.")
        assert out == "upstream compatibility output"
        assert len(call_bodies) >= 2
        assert "reasoning_effort" in call_bodies[0]
        assert all("reasoning_effort" not in body for body in call_bodies[1:])

    def test_route_wiki_llm_stub_accepts_legacy_choice_text_from_upstream(
        self,
        monkeypatch,
    ):
        class _DummyLlama:
            is_loaded = False

        class _DummyBackend:
            active_model_name = None

        class _FakeResponse:
            status_code = 200
            text = "{\"ok\":true}"

            def json(self):
                return {
                    "choices": [
                        {
                            "text": '{"summary":"legacy","entities":[],"concepts":[]}'
                        }
                    ]
                }

        class _FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def post(self, url, json, headers):
                return _FakeResponse()

        monkeypatch.setattr(inference_routes, "get_llama_cpp_backend", lambda: _DummyLlama())
        monkeypatch.setattr(inference_routes, "get_inference_backend", lambda: _DummyBackend())
        monkeypatch.setattr(inference_routes, "_llm_upstream_enabled", lambda: True)
        monkeypatch.setattr(
            inference_routes,
            "_llm_upstream_base_url",
            lambda: "https://example.test/v1",
        )
        monkeypatch.setattr(
            inference_routes,
            "_resolve_llm_upstream_model",
            lambda _requested: "dummy/model",
        )
        monkeypatch.setattr(inference_routes, "_llm_upstream_headers", lambda: {})
        monkeypatch.setattr(inference_routes.httpx, "Client", _FakeClient)

        out = _route_wiki_llm_stub("Extract structured knowledge from the source.")
        assert out == '{"summary":"legacy","entities":[],"concepts":[]}'

    def test_completions_fallback_toggle_defaults_to_enabled(self, monkeypatch):
        monkeypatch.setattr(inference_routes, "_llm_upstream_enabled", lambda: True)
        monkeypatch.setattr(
            inference_routes,
            "_LLM_UPSTREAM_ENABLE_COMPLETIONS_FALLBACK",
            True,
        )
        assert _llm_upstream_completions_fallback_enabled() is True

    def test_completions_fallback_toggle_respects_disable(self, monkeypatch):
        monkeypatch.setattr(inference_routes, "_llm_upstream_enabled", lambda: True)
        monkeypatch.setattr(
            inference_routes,
            "_LLM_UPSTREAM_ENABLE_COMPLETIONS_FALLBACK",
            False,
        )
        assert _llm_upstream_completions_fallback_enabled() is False

    def test_embeddings_fallback_toggle_defaults_to_disabled(self, monkeypatch):
        monkeypatch.setattr(inference_routes, "_llm_upstream_enabled", lambda: True)
        monkeypatch.setattr(
            inference_routes,
            "_LLM_UPSTREAM_ENABLE_EMBEDDINGS_FALLBACK",
            False,
        )
        assert _llm_upstream_embeddings_fallback_enabled() is False

    def test_embeddings_fallback_toggle_can_be_enabled(self, monkeypatch):
        monkeypatch.setattr(inference_routes, "_llm_upstream_enabled", lambda: True)
        monkeypatch.setattr(
            inference_routes,
            "_LLM_UPSTREAM_ENABLE_EMBEDDINGS_FALLBACK",
            True,
        )
        assert _llm_upstream_embeddings_fallback_enabled() is True

    def test_upstream_tool_loop_replays_reasoning_content_in_follow_up(
        self,
        monkeypatch,
    ):
        req = ChatCompletionRequest(
            model = "default",
            messages = [{"role": "user", "content": "hi"}],
            stream = False,
            enable_tools = True,
            enable_thinking = True,
            max_tool_calls_per_message = 2,
        )

        monkeypatch.setattr(
            inference_routes,
            "_upstream_builtin_tools_for_payload",
            lambda _payload: [
                {
                    "type": "function",
                    "function": {
                        "name": "python",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        )
        monkeypatch.setattr(
            inference_routes,
            "_upstream_tool_use_nudge",
            lambda _model, _tools: "",
        )

        import core.inference.tools as _tools_mod

        monkeypatch.setattr(
            _tools_mod,
            "execute_tool",
            lambda *_a, **_k: "tool-ok",
        )

        request_bodies: list[dict] = []

        async def _fake_call(body: dict[str, object]) -> dict:
            request_bodies.append(json.loads(json.dumps(body)))
            if len(request_bodies) == 1:
                return {
                    "choices": [
                        {
                            "finish_reason": "tool_calls",
                            "message": {
                                "content": None,
                                "reasoning_content": "reasoning-trace-1",
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "python",
                                            "arguments": '{"code":"print(1)"}',
                                        },
                                    }
                                ],
                            },
                        }
                    ],
                    "usage": {"prompt_tokens": 12, "completion_tokens": 8},
                }

            return {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "done"},
                    }
                ],
                "usage": {"prompt_tokens": 18, "completion_tokens": 6},
            }

        monkeypatch.setattr(
            inference_routes,
            "_openai_upstream_chat_non_streaming_json_from_body",
            _fake_call,
        )

        async def _collect_events() -> list[dict]:
            collected = []
            async for event in _openai_upstream_tool_loop_events(req, "dummy/model"):
                collected.append(event)
            return collected

        events = asyncio.run(_collect_events())

        assert len(request_bodies) >= 2

        second_messages = request_bodies[1].get("messages") or []
        assistant_msgs = [
            msg for msg in second_messages if isinstance(msg, dict) and msg.get("role") == "assistant"
        ]
        assert assistant_msgs, f"No assistant replay message in second request: {second_messages!r}"
        assert assistant_msgs[-1].get("reasoning_content") == "reasoning-trace-1"
        assert assistant_msgs[-1].get("tool_calls"), "tool_calls missing in replay assistant message"

        content_events = [e for e in events if e.get("type") == "content"]
        assert content_events
        assert content_events[-1].get("text") == "done"


# =====================================================================
# _looks_like_history_intent — explicit phrase detection
# =====================================================================


class TestHistoryIntentDetection:
    def test_explicit_chat_history_phrase_detected(self):
        assert _looks_like_history_intent("Use chat history to answer this") is True

    def test_previous_message_phrase_detected(self):
        assert (
            _looks_like_history_intent("What did I ask in the previous message?")
            is True
        )

    def test_tokenization_query_not_treated_as_history(self):
        query = "Explain tokenization and token limits for this model"
        assert _looks_like_history_intent(query) is False

    def test_api_token_query_not_treated_as_history(self):
        query = "How should I rotate an API token safely?"
        assert _looks_like_history_intent(query) is False

    def test_remember_without_history_phrase_is_not_history(self):
        assert _looks_like_history_intent("Remember to include caveats") is False


class TestManualWikiChatHistorySave:
    def test_save_chat_history_creates_then_updates_same_thread_file(
        self,
        monkeypatch,
        tmp_path: Path,
    ):
        vault = tmp_path / "vault"
        monkeypatch.setattr(inference_routes, "_WIKI_VAULT_ROOT", vault)
        monkeypatch.setattr(inference_routes, "_WIKI_WATCHER_ENABLED", True)

        created = inference_routes._save_chat_history_to_route_wiki(
            thread_id = "thread-abc",
            thread_title = "Session A",
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "First question"}],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "reasoning", "text": "Internal thoughts"},
                        {"type": "text", "text": "Initial answer"},
                    ],
                },
            ],
        )

        assert created["status"] == "ok"
        assert created["operation"] == "created"
        first_path = Path(created["file_path"])
        assert first_path.exists()

        created_text = first_path.read_text(encoding = "utf-8")
        assert "Thread ID: thread-abc" in created_text
        assert "Thread Title: Session A" in created_text
        assert "### Thinking" in created_text
        assert "Internal thoughts" in created_text
        assert "Initial answer" in created_text

        updated = inference_routes._save_chat_history_to_route_wiki(
            thread_id = "thread-abc",
            thread_title = "Session A",
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Second question"}],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Updated answer"}],
                },
            ],
        )

        assert updated["status"] == "ok"
        assert updated["operation"] == "updated"
        assert updated["file_path"] == created["file_path"]

        updated_text = first_path.read_text(encoding = "utf-8")
        assert "Updated answer" in updated_text
        assert "Second question" in updated_text
        assert "First question" not in updated_text

    def test_save_chat_history_ingests_immediately_when_watcher_disabled(
        self,
        monkeypatch,
        tmp_path: Path,
    ):
        vault = tmp_path / "vault"
        monkeypatch.setattr(inference_routes, "_WIKI_VAULT_ROOT", vault)
        monkeypatch.setattr(inference_routes, "_WIKI_WATCHER_ENABLED", False)

        ingested_paths: list[Path] = []

        class _DummyIngestor:
            def ingest_file(self, file_path, contributor = None):
                ingested_paths.append(Path(file_path))
                return "ok"

        monkeypatch.setattr(
            inference_routes,
            "_get_route_wiki_components",
            lambda: (None, _DummyIngestor()),
        )

        result = inference_routes._save_chat_history_to_route_wiki(
            thread_id = "thread-def",
            thread_title = None,
            messages = [
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Answer"}],
                }
            ],
        )

        assert result["status"] == "ok"
        assert result["ingested_immediately"] is True
        assert len(ingested_paths) == 1
        assert ingested_paths[0] == Path(result["file_path"])


# =====================================================================
# _friendly_error — httpx transport failures
# =====================================================================


class TestFriendlyErrorHttpx:
    """The async pass-through helpers talk to llama-server via httpx.
    When the subprocess is down, httpx raises RequestError subclasses
    whose string form (``"All connection attempts failed"``, ``"[Errno 111]
    Connection refused"``, ...) does NOT contain the substring
    ``"Lost connection to llama-server"`` the sync path uses, so the
    previous substring-only `_friendly_error` returned a useless generic
    message. These tests pin the new isinstance-based mapping.
    """

    def _req(self):
        return httpx.Request("POST", "http://127.0.0.1:65535/v1/chat/completions")

    def test_connect_error_mapped(self):
        exc = httpx.ConnectError("All connection attempts failed", request = self._req())
        assert "Lost connection" in _friendly_error(exc)

    def test_read_error_mapped(self):
        exc = httpx.ReadError("EOF", request = self._req())
        assert "Lost connection" in _friendly_error(exc)

    def test_remote_protocol_error_mapped(self):
        exc = httpx.RemoteProtocolError("peer closed", request = self._req())
        assert "Lost connection" in _friendly_error(exc)

    def test_read_timeout_mapped(self):
        exc = httpx.ReadTimeout("timed out", request = self._req())
        assert "Lost connection" in _friendly_error(exc)

    def test_non_httpx_unchanged(self):
        # Non-httpx exceptions still fall through to the existing substring
        # heuristics — a context-size message must still produce the
        # "Message too long" path.
        ctx_msg = (
            "request (4096 tokens) exceeds the available context size (2048 tokens)"
        )
        assert "Message too long" in _friendly_error(ValueError(ctx_msg))

    def test_generic_exception_returns_generic_message(self):
        assert (
            _friendly_error(RuntimeError("unrelated")) == "An internal error occurred"
        )
