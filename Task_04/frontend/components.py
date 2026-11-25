"""Reusable rendering helpers for chat messages and RAG source references."""

from __future__ import annotations

from typing import Any

import streamlit as st


def render_sources(sources: list[dict[str, Any]]) -> None:
    if not sources:
        return

    with st.expander(f"Supporting sources · {len(sources)} retrieved passage(s)", expanded=False):
        for index, source in enumerate(sources):
            citation = source.get("citation_id", "Source")
            filename = source.get("source", "Unknown document")
            page = source.get("page")
            location = f"Page {page}" if page is not None else "Page unavailable"
            score = source.get("relevance_score")
            score_label = f" · relevance {float(score):.3f}" if score is not None else ""

            st.markdown(f"**[{citation}] {filename} — {location}{score_label}**")
            excerpt = source.get("excerpt", "")
            if excerpt:
                st.caption(excerpt)
            if index < len(sources) - 1:
                st.divider()


def render_message(message: dict[str, Any]) -> None:
    role = message.get("role", "assistant")
    avatar = "🤖" if role == "assistant" else "👤"
    with st.chat_message(role, avatar=avatar):
        st.markdown(message.get("content", ""))
        if role == "assistant":
            render_sources(message.get("sources", []))
