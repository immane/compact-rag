"""Dashboard page — system overview with stats and health."""

from __future__ import annotations

from html import escape

import streamlit as st

from compact_rag.admin.client import AdminAPIClient
from compact_rag.admin.components.status import render_status_badge


def render(client: AdminAPIClient) -> None:
    st.markdown('<div class="rag-eyebrow">Workspace / Overview</div>', unsafe_allow_html=True)
    st.title("📊 Dashboard")
    st.markdown(
        '<p class="rag-subtitle">Monitor your knowledge base and service health at a glance.</p>',
        unsafe_allow_html=True,
    )

    try:
        health_data = client.health()
    except Exception:
        health_data = {
            "api": "unknown",
            "database": "unknown",
            "chromadb": "unknown",
            "storage": "unknown",
        }

    try:
        info_data = client.info()
    except Exception:
        info_data = {
            "version": "?",
            "llm_provider": "?",
            "llm_model": "?",
            "storage_backend": "?",
            "embedding_model": "?",
        }

    try:
        collections_data = client.list_collections(page=1, page_size=1)
        collection_count = collections_data.get("pagination", {}).get("total", 0)
    except Exception:
        collection_count = 0

    try:
        docs_data = client.list_documents(page=1, page_size=1)
        doc_count = docs_data.get("pagination", {}).get("total", 0)
    except Exception:
        doc_count = 0

    try:
        conv_data = client.list_conversations(page=1, page_size=1)
        conv_count = conv_data.get("pagination", {}).get("total", 0)
    except Exception:
        conv_count = 0

    try:
        jobs_data = client.list_ingestion_jobs(page=1, page_size=5)
        recent_jobs = jobs_data.get("data", [])
    except Exception:
        recent_jobs = []

    try:
        storage_info = client.list_storage_files()
        storage_count = len(storage_info.get("data", []))
    except Exception:
        storage_count = "?"

    col1, col2, col3, col4 = st.columns(4, gap="medium")
    with col1:
        st.metric("📄 Documents", doc_count)
    with col2:
        st.metric("📁 Collections", collection_count)
    with col3:
        st.metric("💬 Conversations", conv_count)
    with col4:
        st.metric("💾 Storage Files", storage_count)

    st.write("")

    with st.container(border=True):
        st.subheader("🔌 Service Health")
        st.markdown(
            '<p class="rag-section-note">Live status of the services powering retrieval and storage.</p>',
            unsafe_allow_html=True,
        )
        health_cols = st.columns(4)
        components = ["api", "database", "chromadb", "storage"]
        labels = ["API", "Database", "ChromaDB", "Storage"]
        for i, (comp, label) in enumerate(zip(components, labels)):
            status_val = health_data.get(comp, "unknown")
            with health_cols[i]:
                st.markdown(f"**{label}**")
                st.markdown(render_status_badge(status_val), unsafe_allow_html=True)

    st.write("")

    col_left, col_right = st.columns([1.3, 1], gap="medium")

    with col_left:
        with st.container(border=True):
            st.subheader("⚡ Recent Ingestion Jobs")
            st.caption("Latest document processing activity")
            if recent_jobs:
                import pandas as pd

                rows = []
                for j in recent_jobs[:5]:
                    total_files = j.get("total_files", 0)
                    processed_files = j.get("processed_files", 0)
                    total_chunks = j.get("total_chunks", 0)
                    if (
                        j.get("status") == "completed"
                        and total_files > 0
                        and processed_files == 0
                    ):
                        processed_files = total_files
                    if j.get("status") == "completed" and total_chunks == 0:
                        total_chunks = "-"
                    rows.append(
                        {
                            "ID": j.get("id", "")[:8],
                            "Status": j.get("status", "pending"),
                            "Progress": f"{processed_files}/{total_files}",
                            "Chunks": total_chunks,
                        }
                    )
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.info("No ingestion jobs yet. Upload a document to get started.")

    with col_right:
        with st.container(border=True):
            st.subheader("⚙️ System Config")
            st.caption("Active runtime configuration")
            config_items = {
                "Version": info_data.get("version", "?"),
                "LLM Provider": info_data.get("llm_provider", "?"),
                "LLM Model": info_data.get("llm_model", "?"),
                "Storage": info_data.get("storage_backend", "?"),
                "Embedding": info_data.get("embedding_model", "?"),
            }
            for key, val in config_items.items():
                st.markdown(
                    f'<div class="rag-config-row"><span>{escape(key)}</span>'
                    f'<strong>{escape(str(val))}</strong></div>',
                    unsafe_allow_html=True,
                )
