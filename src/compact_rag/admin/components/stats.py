"""Stat card component using st.metric."""

from __future__ import annotations

import streamlit as st


def render_stat_card(
    title: str,
    value: int | str,
    delta: int | str | None = None,
    icon: str | None = None,
) -> None:
    label = f"{icon} {title}" if icon else title
    st.metric(label=label, value=value, delta=delta)
