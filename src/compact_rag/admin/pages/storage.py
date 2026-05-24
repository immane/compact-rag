"""Storage page — file list, filters, usage stats, preview, download, and cleanup."""

from __future__ import annotations

import streamlit as st

from compact_rag.admin.client import AdminAPIClient


def render(client: AdminAPIClient) -> None:
    st.title("💾 Storage")

    col1, col2, col3 = st.columns(3)

    try:
        storage_data = client.list_storage_files()
        items = storage_data.get("data", []) if isinstance(storage_data, dict) else []
    except Exception:
        items = []

    total_files = len(items)
    total_size = 0
    temp_files = 0
    persistent_files = 0
    for item in items:
        total_size += item.get("file_size", 0)
        stype = item.get("storage_type", "")
        if stype == "temp":
            temp_files += 1
        else:
            persistent_files += 1

    with col1:
        st.metric("Total Files", total_files)
    with col2:
        st.metric("Total Size", f"{total_size / 1024 / 1024:.1f} MB")
    with col3:
        st.metric("Temp Files", temp_files, delta=f"{persistent_files} persistent")

    st.markdown("---")

    type_filter = st.selectbox(
        "Type", ["all", "persistent", "temp"], key="storage_type_filter"
    )

    filtered = items
    if type_filter != "all":
        filtered = [f for f in items if f.get("storage_type", "") == type_filter]

    if not filtered:
        st.info("No files found")
    else:
        for f_item in filtered:
            storage_key = f_item.get("storage_key", "")
            filename = f_item.get("filename", "?")
            file_size = f_item.get("file_size", 0)
            storage_type = f_item.get("storage_type", "?")
            content_type = f_item.get("content_type", "")

            with st.container():
                cols = st.columns([3, 1, 1, 1])
                with cols[0]:
                    st.markdown(f"**{filename}**")
                    st.caption(f"Key: {storage_key}")
                    st.caption(
                        f"Type: {content_type or 'unknown'} | Size: {file_size / 1024:.1f} KB"
                    )
                with cols[1]:
                    st.caption(storage_type)
                with cols[2]:
                    url = client.get_file_url(storage_key)
                    st.link_button("🔗 Download", url)
                with cols[3]:
                    if st.button("🗑️", key=f"storage_del_{storage_key}"):
                        try:
                            client.delete_storage_file(storage_key)
                            st.success(f"Deleted '{filename}'")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed: {e}")
                st.divider()

    st.markdown("---")

    st.subheader("🧹 Temp File Cleanup")
    if st.button("Clean Expired Temp Files", type="primary"):
        try:
            result = client.clean_temp_files()
            st.success(f"Cleanup complete: {result.get('cleaned', 0)} files removed")
        except Exception as e:
            st.warning(f"Cleanup may not be available: {e}")
