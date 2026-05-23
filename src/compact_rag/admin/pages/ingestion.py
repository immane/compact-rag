"""Ingestion jobs page — list with statuses, progress, and error details."""

from __future__ import annotations

import streamlit as st

from compact_rag.admin.client import AdminAPIClient
from compact_rag.admin.components.status import render_status_badge


def render(client: AdminAPIClient) -> None:
    st.title("⚙️ Ingestion Jobs")

    col1, col2, col3 = st.columns(3)
    with col1:
        filter_status = st.selectbox("Status", ["", "running", "completed", "failed"], key="ij_status")
    with col2:
        filter_collection = st.text_input("Collection", placeholder="Optional", key="ij_collection")
    with col3:
        page = st.number_input("Page", min_value=1, value=1, key="ij_page")

    try:
        data = client.list_ingestion_jobs(
            status=filter_status or None,
            collection=filter_collection or None,
            page=page,
        )
        items = data.get("data", [])
        pagination = data.get("pagination", {})

        st.caption(f"Total: {pagination.get('total', 0)} job(s) | Page {page}/{pagination.get('total_pages', 0)}")

        if not items:
            st.info("No ingestion jobs found")
            return

        for job in items:
            job_id = job.get("id", "")
            status = job.get("status", "pending")
            total_files = job.get("total_files", 0)
            processed = job.get("processed_files", 0)
            total_chunks = job.get("total_chunks", 0)
            errors = job.get("errors", []) or []
            started = job.get("started_at", "")
            completed = job.get("completed_at", "")
            created = job.get("created_at", "")
            display_processed = processed
            if status == "completed" and total_files > 0 and processed == 0:
                display_processed = total_files
            display_chunks = total_chunks
            if status == "completed" and total_chunks == 0:
                display_chunks = None

            with st.container():
                cols = st.columns([2, 1, 2, 1])
                with cols[0]:
                    st.markdown(f"**Job:** {job_id[:12]}...")
                    collection_id = job.get("collection_id", "")
                    st.caption(f"Collection: {collection_id[:8]}..." if collection_id else "Collection: -")
                with cols[1]:
                    st.markdown(render_status_badge(status), unsafe_allow_html=True)
                with cols[2]:
                    if total_files > 0:
                        progress = min(display_processed / total_files, 1.0)
                        st.progress(progress, text=f"Files: {display_processed}/{total_files}")
                    st.caption(f"Chunks: {display_chunks if display_chunks is not None else '-'}")
                with cols[3]:
                    if started or created:
                        st.caption(f"Started: {(started or created)[:19]}")
                    if completed:
                        st.caption(f"Done: {completed[:19]}")

                if errors:
                    with st.expander(f"Errors ({len(errors)})"):
                        for err in errors:
                            st.warning(str(err))

                st.divider()
    except Exception as e:
        st.error(f"Failed to load ingestion jobs: {e}")
