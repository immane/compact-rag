"""Small reusable display primitives for admin list pages."""

from __future__ import annotations

from html import escape

import streamlit as st


def render_compact_stat(label: str, value: str | int, detail: str = "") -> None:
    """Render a compact inline stat that avoids Streamlit's oversized metric type."""
    detail_html = f"<small>{escape(detail)}</small>" if detail else ""
    st.markdown(
        '<div class="rag-inline-stat">'
        f"<span>{escape(label)}</span>"
        f"<strong>{escape(str(value))}</strong>"
        f"{detail_html}</div>",
        unsafe_allow_html=True,
    )


def render_metadata(label: str, value: str | int) -> None:
    """Render a small label/value pair for card rows."""
    st.markdown(
        '<div class="rag-row-meta">'
        f"<span>{escape(label)}</span><strong>{escape(str(value))}</strong></div>",
        unsafe_allow_html=True,
    )


def render_page_intro(section: str) -> None:
    """Render the compact section eyebrow above a page title."""
    st.markdown(
        f'<div class="rag-eyebrow">{escape(section)}</div>',
        unsafe_allow_html=True,
    )


def render_page_subtitle(description: str) -> None:
    """Render supporting copy below the page title."""
    st.markdown(
        f'<p class="rag-subtitle">{escape(description)}</p>',
        unsafe_allow_html=True,
    )
