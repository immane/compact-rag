"""Playground page — interactive chat interface with RAG settings."""

from __future__ import annotations

import streamlit as st

from compact_rag.admin.client import AdminAPIClient


def render(client: AdminAPIClient) -> None:
    st.title("🎮 Playground")

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    with st.sidebar:
        st.subheader("⚙️ RAG Settings")

        try:
            collections_data = client.list_collections(page=1, page_size=100)
            collection_names = [c.get("name", "") for c in collections_data.get("data", [])]
            if not collection_names:
                collection_names = ["default"]
        except Exception:
            collection_names = ["default"]

        collection = st.selectbox("Collection", collection_names, key="pg_collection")
        top_k = st.slider("Top-K Results", 1, 20, 10, key="pg_top_k")
        use_hybrid = st.toggle("Hybrid Search", value=True, key="pg_hybrid")
        use_rerank = st.toggle("Rerank", value=True, key="pg_rerank")
        temperature = st.slider("Temperature", 0.0, 2.0, 0.1, 0.05, key="pg_temp")
        use_stream = st.toggle("Streaming", value=False, key="pg_stream")

        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_messages = []
            st.rerun()

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("citations"):
                with st.expander("📎 Sources"):
                    for cite in msg["citations"]:
                        snippet = cite.get("content_snippet", cite.get("snippet", ""))
                        st.caption(
                            f"**{cite.get('filename', '?')}** "
                            f"(p.{cite.get('page_number', '?')}, score: {cite.get('score', 0):.3f})"
                        )
                        st.caption(f"_{snippet[:200]}_")

    prompt = st.chat_input("Ask a question about your documents...")
    if prompt:
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        api_messages = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.chat_messages
        ]

        with st.chat_message("assistant"):
            if use_stream:
                placeholder = st.empty()
                full_content = ""
                try:
                    for chunk in client.chat_stream(
                            messages=api_messages,
                            collection=collection,
                            top_k=top_k,
                            temperature=temperature,
                            use_rerank=use_rerank,
                            use_hybrid=use_hybrid,
                        ):
                        full_content += chunk
                        placeholder.markdown(full_content + "▌")
                    placeholder.markdown(full_content)
                    content = full_content
                    citations = []
                except Exception as e:
                    content = f"Error: {e}"
                    st.error(content)
                    citations = []
            else:
                with st.spinner("Thinking..."):
                    try:
                        response = client.chat(
                            messages=api_messages,
                            collection=collection,
                            top_k=top_k,
                            temperature=temperature,
                            use_rerank=use_rerank,
                            use_hybrid=use_hybrid,
                        )
                        content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
                        st.markdown(content)
                        citations = response.get("choices", [{}])[0].get("message", {}).get("citations", [])
                    except Exception as e:
                        content = f"Error: {e}"
                        st.error(content)
                        citations = []

            if citations:
                with st.expander("📎 Sources"):
                    for cite in citations:
                        snippet = cite.get("content_snippet", cite.get("snippet", ""))
                        st.caption(
                            f"**{cite.get('filename', '?')}** "
                            f"(p.{cite.get('page_number', '?')}, score: {cite.get('score', 0):.3f})"
                        )
                        st.caption(f"_{snippet[:200]}_")

        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": content,
            "citations": citations,
        })
