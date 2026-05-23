"""Conversations page — list, detail view, JSON/CSV export."""

from __future__ import annotations

import csv
import io
import json

import streamlit as st

from compact_rag.admin.client import AdminAPIClient


def render(client: AdminAPIClient) -> None:
    st.title("💬 Conversations")

    page = st.number_input("Page", min_value=1, value=1, key="conv_page")

    try:
        data = client.list_conversations(page=page)
        items = data.get("data", [])
        pagination = data.get("pagination", {})

        st.caption(f"Total: {pagination.get('total', 0)} conversation(s) | Page {page}/{pagination.get('total_pages', 0)}")

        if not items:
            st.info("No conversations yet")
            return

        for conv in items:
            conv_id = conv.get("id", "")
            title = conv.get("title", "Untitled")
            model = conv.get("model", "")
            msg_count = conv.get("message_count", 0)
            created = conv.get("created_at", "")

            with st.container():
                cols = st.columns([3, 1, 1, 1, 1])
                with cols[0]:
                    st.markdown(f"**{title}**")
                    st.caption(f"Model: {model} | {created[:10]}")
                with cols[1]:
                    st.metric("Messages", msg_count)
                with cols[2]:
                    detail_key = f"conv_detail_{conv_id}"
                    if st.button("🔍 View", key=f"conv_view_{conv_id}"):
                        st.session_state[detail_key] = not st.session_state.get(detail_key, False)
                        if st.session_state[detail_key]:
                            st.session_state["conv_detail_data"] = None
                        st.rerun()

                if st.session_state.get(detail_key):
                    with st.container():
                        try:
                            if st.session_state.get("conv_detail_data") is None:
                                conv_detail = client.get_conversation(conv_id)
                                st.session_state["conv_detail_data"] = conv_detail
                            else:
                                conv_detail = st.session_state["conv_detail_data"]

                            messages = conv_detail.get("messages", [])

                            st.subheader(f"📝 {title}")
                            export_col1, export_col2, export_col3 = st.columns(3)
                            with export_col1:
                                if st.button("📥 JSON", key=f"json_{conv_id}"):
                                    json_str = json.dumps(messages, indent=2, default=str)
                                    st.download_button(
                                        "Download JSON",
                                        json_str,
                                        file_name=f"conversation_{conv_id[:8]}.json",
                                    )
                            with export_col2:
                                if st.button("📊 CSV", key=f"csv_{conv_id}"):
                                    buf = io.StringIO()
                                    writer = csv.DictWriter(buf, fieldnames=["role", "content", "sources", "token_count"])
                                    writer.writeheader()
                                    for m in messages:
                                        writer.writerow({
                                            "role": m.get("role", ""),
                                            "content": m.get("content", ""),
                                            "sources": json.dumps(m.get("sources", [])),
                                            "token_count": m.get("token_count", 0),
                                        })
                                    st.download_button(
                                        "Download CSV",
                                        buf.getvalue(),
                                        file_name=f"conversation_{conv_id[:8]}.csv",
                                    )
                            with export_col3:
                                if st.button("❌ Close", key=f"conv_close_{conv_id}", type="secondary"):
                                    st.session_state[detail_key] = False
                                    st.session_state["conv_detail_data"] = None
                                    st.rerun()

                            for msg in messages:
                                role_icon = "🧑" if msg.get("role") == "user" else "🤖"
                                st.markdown(f"{role_icon} **{msg.get('role', '').upper()}** — {msg.get('created_at', '')[:19]}")
                                st.markdown(msg.get("content", ""))
                                sources = msg.get("sources")
                                if sources:
                                    with st.expander("📎 Sources"):
                                        src_data = json.loads(sources) if isinstance(sources, str) else sources
                                        if isinstance(src_data, list):
                                            for src in src_data:
                                                st.caption(f"- {src.get('filename', '?')} (score: {src.get('score', 0):.3f})")
                                        else:
                                            st.json(src_data)
                                st.divider()
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
                            if st.button("Confirm", key=f"conv_confirm_{conv_id}", type="primary"):
                                try:
                                    client.delete_conversation(conv_id)
                                    st.success(f"Deleted '{title}'")
                                    st.session_state[delete_key] = False
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Failed: {e}")
                        with cc2:
                            if st.button("Cancel", key=f"conv_cancel_{conv_id}", type="secondary"):
                                st.session_state[delete_key] = False
                                st.rerun()
                st.divider()
    except Exception as e:
        st.error(f"Failed to load conversations: {e}")
