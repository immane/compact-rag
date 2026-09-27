"""Conversations page — list, detail view, JSON/CSV export."""

from __future__ import annotations

import csv
from html import escape
import io
import json

import streamlit as st

from compact_rag.admin.client import AdminAPIClient
from compact_rag.admin.components.display import render_page_intro, render_page_subtitle


def render(client: AdminAPIClient) -> None:
    render_page_intro("Workspace / Conversations")
    st.title("💬 Conversations")
    render_page_subtitle("Browse conversation history and inspect source citations.")

    page = st.number_input("Page", min_value=1, value=1, key="conv_page")

    try:
        data = client.list_conversations(page=page)
        items = data.get("data", [])
        pagination = data.get("pagination", {})

        st.caption(
            f"Total: {pagination.get('total', 0)} conversation(s) | Page {page}/{pagination.get('total_pages', 0)}"
        )

        if not items:
            st.info("No conversations yet")
            return

        for conv in items:
            conv_id = conv.get("id", "")
            title = conv.get("title", "Untitled")
            model = conv.get("model", "")
            msg_count = conv.get("message_count", 0)
            created = conv.get("created_at", "")

            with st.container(border=True):
                cols = st.columns([3.3, 1, 1.1, .8], vertical_alignment="center")
                with cols[0]:
                    st.markdown(f'<div class="rag-card-title">{escape(title)}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="rag-card-description">{escape(model)} · {escape(created[:10])}</div>', unsafe_allow_html=True)
                with cols[1]:
                    st.markdown(
                        '<div class="rag-mini-stat"><span>Messages</span>'
                        f"<strong>{msg_count}</strong></div>",
                        unsafe_allow_html=True,
                    )
                with cols[2]:
                    detail_key = f"conv_detail_{conv_id}"
                    if st.button("🔍 View", key=f"conv_view_{conv_id}"):
                        st.session_state[detail_key] = not st.session_state.get(
                            detail_key, False
                        )
                        if st.session_state[detail_key]:
                            st.session_state["conv_detail_data"] = None
                        st.rerun()

                if st.session_state.get(detail_key):
                    with st.container(border=True):
                        try:
                            if st.session_state.get("conv_detail_data") is None:
                                conv_detail = client.get_conversation(conv_id)
                                st.session_state["conv_detail_data"] = conv_detail
                            else:
                                conv_detail = st.session_state["conv_detail_data"]

                            messages = conv_detail.get("messages", [])

                            st.markdown(
                                '<div class="rag-transcript-heading">Conversation transcript</div>',
                                unsafe_allow_html=True,
                            )
                            export_col1, export_col2, export_col3 = st.columns(3)
                            with export_col1:
                                if st.button("📥 JSON", key=f"json_{conv_id}"):
                                    json_str = json.dumps(
                                        messages, indent=2, default=str
                                    )
                                    st.download_button(
                                        "Download JSON",
                                        json_str,
                                        file_name=f"conversation_{conv_id[:8]}.json",
                                    )
                            with export_col2:
                                if st.button("📊 CSV", key=f"csv_{conv_id}"):
                                    buf = io.StringIO()
                                    writer = csv.DictWriter(
                                        buf,
                                        fieldnames=[
                                            "role",
                                            "content",
                                            "sources",
                                            "token_count",
                                        ],
                                    )
                                    writer.writeheader()
                                    for m in messages:
                                        writer.writerow(
                                            {
                                                "role": m.get("role", ""),
                                                "content": m.get("content", ""),
                                                "sources": json.dumps(
                                                    m.get("sources", [])
                                                ),
                                                "token_count": m.get("token_count", 0),
                                            }
                                        )
                                    st.download_button(
                                        "Download CSV",
                                        buf.getvalue(),
                                        file_name=f"conversation_{conv_id[:8]}.csv",
                                    )
                            with export_col3:
                                if st.button(
                                    "❌ Close",
                                    key=f"conv_close_{conv_id}",
                                    type="secondary",
                                ):
                                    st.session_state[detail_key] = False
                                    st.session_state["conv_detail_data"] = None
                                    st.rerun()

                            with st.container(key="conversation-transcript"):
                                for msg in messages:
                                    role_icon = "🧑" if msg.get("role") == "user" else "🤖"
                                    with st.container(border=True):
                                        st.markdown(
                                            f'<div class="rag-message-meta">{role_icon} '
                                            f'{escape(msg.get("role", "").upper())} · '
                                            f'{escape(msg.get("created_at", "")[:19])}</div>',
                                            unsafe_allow_html=True,
                                        )
                                        st.markdown(msg.get("content", ""))
                                        sources = msg.get("sources")
                                        if sources:
                                            with st.expander("📎 Sources"):
                                                src_data = (
                                                    json.loads(sources)
                                                    if isinstance(sources, str)
                                                    else sources
                                                )
                                                if isinstance(src_data, list):
                                                    for src in src_data:
                                                        st.markdown(
                                                            f'<div class="rag-row-meta"><strong>{escape(str(src.get("filename", "?")))}</strong>'
                                                            f'<span>Relevance {src.get("score", 0):.3f}</span></div>',
                                                            unsafe_allow_html=True,
                                                        )
                                                else:
                                                    st.json(src_data)
                        except Exception as e:
                            st.error(f"Failed to load conversation: {e}")

                with cols[3]:
                    delete_key = f"conv_del_{conv_id}"
                    if st.button("🗑️", key=f"conv_del_btn_{conv_id}"):
                        st.session_state[delete_key] = True
                        st.rerun()

                    if st.session_state.get(delete_key):
                        st.warning(f"Delete **{title}**?")
                        cc1, cc2 = st.columns(2)
                        with cc1:
                            if st.button(
                                "Confirm", key=f"conv_confirm_{conv_id}", type="primary"
                            ):
                                try:
                                    client.delete_conversation(conv_id)
                                    st.success(f"Deleted '{title}'")
                                    st.session_state[delete_key] = False
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Failed: {e}")
                        with cc2:
                            if st.button(
                                "Cancel", key=f"conv_cancel_{conv_id}", type="secondary"
                            ):
                                st.session_state[delete_key] = False
                                st.rerun()
    except Exception as e:
        st.error(f"Failed to load conversations: {e}")
