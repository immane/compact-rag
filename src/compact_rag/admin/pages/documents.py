"""Documents page — upload, filter, list, and delete documents."""

from __future__ import annotations

import streamlit as st

from compact_rag.admin.client import AdminAPIClient
from compact_rag.admin.components.status import render_status_badge


def render(client: AdminAPIClient) -> None:
    st.title("📄 Documents")

    with st.expander("📤 Upload Document", expanded=False):
        uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx", "txt", "md", "html"])
        collection = st.text_input("Target Collection", value="default", key="upload_collection")
        if uploaded_file and st.button("Upload & Ingest", type="primary"):
            try:
                result = client.upload_document(
                    file_data=uploaded_file.read(),
                    filename=uploaded_file.name,
                    collection=collection,
                )
                st.success(f"Ingested: {result.get('filename', '')} — {result.get('status', '')}")
                st.rerun()
            except Exception as e:
                st.error(f"Upload failed: {e}")

    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        filter_collection = st.text_input("Collection", value="", placeholder="Filter by collection", key="doc_filter_col")
    with col2:
        filter_status = st.selectbox("Status", ["", "pending", "completed", "failed", "skipped"], key="doc_filter_status")
    with col3:
        page = st.number_input("Page", min_value=1, value=1, key="doc_page")

    try:
        data = client.list_documents(
            collection=filter_collection or None,
            status=filter_status or None,
            page=page,
            page_size=20,
        )
        items = data.get("data", [])
        pagination = data.get("pagination", {})

        st.caption(f"Total: {pagination.get('total', 0)} document(s) | Page {page}/{pagination.get('total_pages', 0)}")

        if not items:
            st.info("No documents found")
            return

        for doc in items:
            doc_id = doc.get("id", "")
            filename = doc.get("filename", "")
            file_type = doc.get("file_type", "")
            status = doc.get("status", "pending")
            chunks = doc.get("chunk_count", 0)
            tables = doc.get("table_count", 0)
            page_count = doc.get("page_count") or 0
            error_msg = doc.get("error_message", "")

            with st.container():
                cols = st.columns([3, 1, 1, 1, 1])
                with cols[0]:
                    st.markdown(f"**{filename}**")
                    st.caption(f"ID: {doc_id[:8]}... | Type: {file_type} | Pages: {page_count}")
                    if error_msg:
                        st.caption(f"Error: {error_msg}")
                with cols[1]:
                    st.markdown(render_status_badge(status), unsafe_allow_html=True)
                with cols[2]:
                    st.caption(f"Chunks: {chunks}")
                with cols[3]:
                    st.caption(f"Tables: {tables}")
                with cols[4]:
                    detail_key = f"detail_{doc_id}"
                    if st.button("🔍 Detail", key=f"detail_btn_{doc_id}"):
                        st.session_state[detail_key] = not st.session_state.get(detail_key, False)
                        st.rerun()

                    delete_key = f"delete_doc_{doc_id}"
                    if st.button("🗑️", key=f"del_btn_{doc_id}"):
                        st.session_state[delete_key] = True
                        st.rerun()

                if st.session_state.get(detail_key):
                    with st.expander(f"Document Details — {filename}", expanded=True):
                        try:
                            detail = client.get_document(doc_id)
                            st.json(detail)
                        except Exception as e:
                            st.error(f"Failed to load detail: {e}")
                        if st.button("Close", key=f"close_{doc_id}"):
                            st.session_state[detail_key] = False
                            st.rerun()

                if st.session_state.get(delete_key):
                    st.warning(f"Delete **{filename}**?")
                    cc1, cc2 = st.columns(2)
                    with cc1:
                        if st.button("Confirm", key=f"confirm_del_{doc_id}", type="primary"):
                            try:
                                client.delete_document(doc_id)
                                st.success(f"Deleted '{filename}'")
                                st.session_state[delete_key] = False
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed: {e}")
                    with cc2:
                        if st.button("Cancel", key=f"cancel_del_{doc_id}", type="secondary"):
                            st.session_state[delete_key] = False
                            st.rerun()
                st.divider()
    except Exception as e:
        st.error(f"Failed to load documents: {e}")
