"""Conversations page — master-detail browser with chat transcript."""

from __future__ import annotations

import csv
from html import escape
import io
import json

import streamlit as st

from compact_rag.admin.client import AdminAPIClient
from compact_rag.admin.components.display import render_page_intro, render_page_subtitle


def _messages_to_csv(messages: list[dict]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(
        buf,
        fieldnames=["role", "content", "sources", "token_count"],
    )
    writer.writeheader()
    for m in messages:
        writer.writerow(
            {
                "role": m.get("role", ""),
                "content": m.get("content", ""),
                "sources": json.dumps(m.get("sources", [])),
                "token_count": m.get("token_count", 0),
            }
        )
    return buf.getvalue()


def _render_transcript(messages: list[dict]) -> None:
    for msg in messages:
        role = msg.get("role", "assistant")
        with st.chat_message("user" if role == "user" else "assistant"):
            st.markdown(msg.get("content", ""))
            meta_bits = [escape(role.upper())]
            created = msg.get("created_at", "")
            if created:
                meta_bits.append(escape(created[:19]))
            tokens = msg.get("token_count")
            if tokens:
                meta_bits.append(f"{tokens} tokens")
            latency = msg.get("latency_ms")
            if latency:
                meta_bits.append(f"{latency} ms")
            st.markdown(
                f'<div class="rag-message-meta">{" · ".join(meta_bits)}</div>',
                unsafe_allow_html=True,
            )
            sources = msg.get("sources")
            if sources:
                with st.expander("📎 Sources"):
                    src_data = (
                        json.loads(sources) if isinstance(sources, str) else sources
                    )
                    if isinstance(src_data, list):
                        for src in src_data:
                            st.markdown(
                                f'<div class="rag-row-meta"><strong>{escape(str(src.get("filename", "?")))}</strong>'
                                f"<span>Relevance {src.get('score', 0):.3f}</span></div>",
                                unsafe_allow_html=True,
                            )
                    else:
                        st.json(src_data)


def render(client: AdminAPIClient) -> None:
    render_page_intro("Workspace / Conversations")
    st.title("💬 Conversations")
    render_page_subtitle("Browse conversation history and inspect source citations.")

    if "conv_selected" not in st.session_state:
        st.session_state.conv_selected = None
    if "conv_detail_cache" not in st.session_state:
        st.session_state.conv_detail_cache = {}

    try:
        data = client.list_conversations(page=1, page_size=100)
        items = data.get("data", [])
        total = data.get("pagination", {}).get("total", len(items))
    except Exception as e:
        st.error(f"Failed to load conversations: {e}")
        return

    with st.container(border=True, key="conv_workspace"):
        list_col, detail_col = st.columns([1, 1.85], gap="small")

        with list_col:
            with st.container(key="conv_list_pane"):
                st.markdown(
                    '<div class="rag-panel-heading">Conversations</div>',
                    unsafe_allow_html=True,
                )
                search = (
                    st.text_input(
                        "Search conversations",
                        placeholder="Search by title…",
                        key="conv_search",
                        label_visibility="collapsed",
                    )
                    .strip()
                    .lower()
                )
                visible = [
                    c for c in items if search in str(c.get("title", "")).lower()
                ]
                st.caption(f"{len(visible)} of {total} conversations")

                valid_ids = {c.get("id", "") for c in items}
                if st.session_state.conv_selected not in valid_ids:
                    st.session_state.conv_selected = (
                        items[0].get("id", "") if items else None
                    )

                with st.container(height=600, border=False, key="conv_list_scroll"):
                    if not visible:
                        st.info("No matching conversations")
                    for conv in visible:
                        conv_id = conv.get("id", "")
                        title = conv.get("title", "Untitled") or "Untitled"
                        model = conv.get("model", "")
                        msg_count = conv.get("message_count", 0)
                        last_active = conv.get("updated_at", "") or conv.get(
                            "created_at", ""
                        )
                        with st.container(key=f"conv_item_{conv_id}"):
                            if st.button(
                                title,
                                key=f"conv_open_{conv_id}",
                                use_container_width=True,
                                type="primary"
                                if conv_id == st.session_state.conv_selected
                                else "secondary",
                            ):
                                st.session_state.conv_selected = conv_id
                                st.rerun()
                            st.caption(
                                f"{model} · {msg_count} messages · {last_active[:16]}"
                            )

        with detail_col:
            with st.container(key="conv_detail_pane"):
                conv_id = st.session_state.conv_selected or ""
                conv = next((c for c in items if c.get("id", "") == conv_id), None)
                if conv is None or not conv_id:
                    st.info("Select a conversation on the left.")
                    return

                cache = st.session_state.conv_detail_cache
                if conv_id not in cache:
                    try:
                        with st.spinner("Loading conversation…"):
                            cache[conv_id] = client.get_conversation(conv_id)
                    except Exception as e:
                        st.error(f"Failed to load conversation: {e}")
                        return
                messages = (cache[conv_id] or {}).get("messages", [])

                header, actions = st.columns([3, 1.6], vertical_alignment="center")
                with header:
                    st.markdown(
                        f'<div class="rag-panel-heading">{escape(conv.get("title", "Untitled") or "Untitled")}</div>',
                        unsafe_allow_html=True,
                    )
                    st.caption(
                        f"{conv.get('model', '')} · {len(messages)} messages · Read-only"
                    )
                with actions:
                    with st.popover("⋯  Options", use_container_width=True):
                        st.download_button(
                            "Download JSON",
                            json.dumps(messages, indent=2, default=str),
                            file_name=f"conversation_{conv_id[:8]}.json",
                            key=f"dl_json_{conv_id}",
                            use_container_width=True,
                        )
                        st.download_button(
                            "Download CSV",
                            _messages_to_csv(messages),
                            file_name=f"conversation_{conv_id[:8]}.csv",
                            key=f"dl_csv_{conv_id}",
                            use_container_width=True,
                        )
                        if st.button(
                            "Delete conversation",
                            key=f"conv_del_btn_{conv_id}",
                            use_container_width=True,
                        ):
                            st.session_state[f"conv_del_{conv_id}"] = True
                            st.rerun()

                delete_key = f"conv_del_{conv_id}"
                if st.session_state.get(delete_key):
                    st.warning(f"Delete **{escape(conv.get('title', ''))}**?")
                    yes, no = st.columns(2)
                    with yes:
                        if st.button(
                            "Confirm deletion",
                            key=f"conv_confirm_{conv_id}",
                            type="primary",
                            use_container_width=True,
                        ):
                            try:
                                client.delete_conversation(conv_id)
                                st.session_state[delete_key] = False
                                st.session_state.conv_selected = None
                                cache.pop(conv_id, None)
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed: {e}")
                    with no:
                        if st.button(
                            "Cancel",
                            key=f"conv_cancel_{conv_id}",
                            use_container_width=True,
                        ):
                            st.session_state[delete_key] = False
                            st.rerun()

                with st.container(height=570, border=False, key="conv_chat"):
                    if messages:
                        _render_transcript(messages)
                    else:
                        st.info("No messages in this conversation.")
                st.markdown(
                    '<div class="rag-chat-readonly">Conversation history · Read only</div>',
                    unsafe_allow_html=True,
                )
