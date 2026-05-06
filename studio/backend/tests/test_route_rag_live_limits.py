# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import os
import sys


_backend = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _backend)

import routes.inference as inference_routes


def test_live_route_rag_limits_reads_env_each_call(monkeypatch):
    monkeypatch.setenv("UNSLOTH_WIKI_RAG_MAX_PAGES", "16")
    monkeypatch.setenv("UNSLOTH_WIKI_RAG_MAX_CHARS_PER_PAGE", "18000")
    monkeypatch.setenv("UNSLOTH_WIKI_RAG_MAX_TOTAL_CHARS", "120000")
    monkeypatch.setenv("UNSLOTH_WIKI_RAG_INCLUDE_SOURCE_PAGES", "false")

    pages, chars_per_page, total_chars, include_sources = (
        inference_routes._live_route_rag_limits()
    )

    assert pages == 16
    assert chars_per_page == 18000
    assert total_chars == 120000
    assert include_sources is False

    monkeypatch.setenv("UNSLOTH_WIKI_RAG_MAX_PAGES", "12")
    pages2, _, _, _ = inference_routes._live_route_rag_limits()
    assert pages2 == 12


def test_rag_debug_defaults_use_live_route_limits(monkeypatch):
    monkeypatch.setenv("UNSLOTH_WIKI_RAG_MAX_PAGES", "16")
    monkeypatch.setenv("UNSLOTH_WIKI_RAG_MAX_CHARS_PER_PAGE", "18000")
    monkeypatch.setenv("UNSLOTH_WIKI_RAG_MAX_TOTAL_CHARS", "120000")

    response = inference_routes._to_rag_debug_response({"query": "xm"})

    assert response.applied_limits["max_pages"] == 16
    assert response.applied_limits["max_chars_per_page"] == 18000
    assert response.applied_limits["max_total_chars"] == 120000
