"""Collections page — CRUD table with create/delete."""

from __future__ import annotations

from html import escape

import streamlit as st

from compact_rag.admin.client import AdminAPIClient
from compact_rag.admin.components.display import (
    render_compact_stat,
    render_page_intro,
    render_page_subtitle,
)


def render(client: AdminAPIClient) -> None:
    render_page_intro("Workspace / Knowledge base")
    st.title("📁 Collections")
    render_page_subtitle("Organize indexed content into searchable collections.")

    with st.expander("➕ Create Collection", expanded=False):
        with st.form("create_collection_form"):
            name = st.text_input("Name", placeholder="my-collection")
            description = st.text_area(
                "Description", placeholder="Optional description"
            )
            embedding_model = st.text_input(
                "Embedding Model", value="BAAI/bge-small-zh-v1.5"
            )
            col1, col2 = st.columns(2)
            with col1:
                chunk_size = st.number_input(
                    "Chunk Size", min_value=100, max_value=5000, value=500
                )
            with col2:
                chunk_overlap = st.number_input(
                    "Chunk Overlap", min_value=0, max_value=1000, value=50
                )
            submitted = st.form_submit_button("Create", type="primary")
            if submitted and name:
                try:
                    client.create_collection(
                        name=name,
                        description=description,
                        embedding_model=embedding_model,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )
                    st.success(f"Collection '{name}' created!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed: {e}")

    page = st.number_input("Page", min_value=1, value=1, key="coll_page")
    page_size = 20

    try:
        data = client.list_collections(page=page, page_size=page_size)
        items = data.get("data", [])
        pagination = data.get("pagination", {})
        total = pagination.get("total", 0)
        total_pages = pagination.get("total_pages", 0)

        st.caption(f"{total} collections · Page {page} of {total_pages or 1}")

        if not items:
            st.info("No collections found")
            return

        for item in items:
            col_id = item.get("id", "")
            col_name = item.get("name", "")
            col_desc = item.get("description", "")
            col_docs = item.get("document_count", 0)
            col_embed = item.get("embedding_model", "")
            created = item.get("created_at", "")

            with st.container(border=True):
                cols = st.columns([3.4, 1, 1.4, 1.1], vertical_alignment="center")
                with cols[0]:
                    st.markdown(
                        f'<div class="rag-card-title">{escape(col_name)}</div>',
                        unsafe_allow_html=True,
                    )
                    if col_desc:
                        st.markdown(
                            f'<div class="rag-card-description">{escape(col_desc)}</div>',
                            unsafe_allow_html=True,
                        )
                    st.markdown(
                        f'<div class="rag-card-description">Embedding · {escape(col_embed)}</div>',
                        unsafe_allow_html=True,
                    )
                with cols[1]:
                    render_compact_stat("Documents", col_docs)
                with cols[2]:
                    if created:
                        st.markdown(
                            f'<div class="rag-row-meta"><span>Created</span><strong>{created[:10]}</strong></div>',
                            unsafe_allow_html=True,
                        )
                with cols[3]:
                    delete_key = f"delete_col_{col_id}"
                    if delete_key not in st.session_state:
                        st.session_state[delete_key] = False

                    if st.button("🗑️ Delete", key=f"btn_{col_id}"):
                        st.session_state[delete_key] = True
                        st.rerun()

                    if st.session_state.get(delete_key):
                        st.warning(
                            f"Delete **{col_name}**? This removes all documents."
                        )
                        cc1, cc2 = st.columns(2)
                        with cc1:
                            if st.button(
                                "Confirm", key=f"confirm_{col_id}", type="primary"
                            ):
                                try:
                                    client.delete_collection(col_name)
                                    st.success(f"Deleted '{col_name}'")
                                    st.session_state[delete_key] = False
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Failed: {e}")
                        with cc2:
                            if st.button(
                                "Cancel", key=f"cancel_{col_id}", type="secondary"
                            ):
                                st.session_state[delete_key] = False
                                st.rerun()
    except Exception as e:
        st.error(f"Failed to load collections: {e}")
