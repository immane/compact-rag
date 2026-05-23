"""API Keys page — CRUD table, create form, and activation toggle."""

from __future__ import annotations

import streamlit as st

from compact_rag.admin.client import AdminAPIClient
from compact_rag.admin.components.status import render_status_badge


def render(client: AdminAPIClient) -> None:
    st.title("🔑 API Keys")

    with st.expander("➕ Create API Key", expanded=False):
        with st.form("create_key_form"):
            name = st.text_input("Key Name", placeholder="my-service-key")
            perms = st.multiselect(
                "Permissions",
                ["read", "write", "admin"],
                default=["read"],
            )
            submitted = st.form_submit_button("Generate Key", type="primary")
            if submitted and name:
                try:
                    result = client.create_api_key(name=name, permissions=perms)
                    raw_key = result.get("key", "")
                    st.success("API key created! Copy the key below — it won't be shown again.")
                    st.code(raw_key, language="text")
                    st.caption("Save this key now. After you navigate away or refresh, it cannot be retrieved.")
                except Exception as e:
                    st.error(f"Failed: {e}")

    st.markdown("---")

    page = st.number_input("Page", min_value=1, value=1, key="key_page")

    try:
        data = client.list_api_keys(page=page)
        items = data.get("data", [])
        pagination = data.get("pagination", {})

        st.caption(f"Total: {pagination.get('total', 0)} key(s) | Page {page}/{pagination.get('total_pages', 0)}")

        if not items:
            st.info("No API keys found")
            return

        for key in items:
            key_id = key.get("id", "")
            key_name = key.get("name", "")
            key_prefix = key.get("key_prefix", "****")
            permissions = key.get("permissions", [])
            is_active = key.get("is_active", False)
            created = key.get("created_at", "")
            expires = key.get("expires_at", "")

            with st.container():
                cols = st.columns([3, 1, 1, 1])
                with cols[0]:
                    st.markdown(f"**{key_name}**")
                    st.code(f"{key_prefix}")
                    st.caption(f"Permissions: {', '.join(permissions)}")
                with cols[1]:
                    status = "active" if is_active else "inactive"
                    st.markdown(render_status_badge(status), unsafe_allow_html=True)
                with cols[2]:
                    if created:
                        st.caption(f"Created: {created[:10]}")
                    if expires:
                        st.caption(f"Expires: {expires[:10]}")
                with cols[3]:
                    toggle_label = "Deactivate" if is_active else "Activate"
                    if st.button(toggle_label, key=f"toggle_{key_id}"):
                        try:
                            client.toggle_api_key(key_id, not is_active)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed: {e}")

                    delete_key = f"key_del_{key_id}"
                    if st.button("🗑️", key=f"key_del_btn_{key_id}"):
                        st.session_state[delete_key] = True
                        st.rerun()

                    if st.session_state.get(delete_key):
                        st.warning(f"Delete **{key_name}**?")
                        cc1, cc2 = st.columns(2)
                        with cc1:
                            if st.button("Confirm", key=f"key_confirm_{key_id}", type="primary"):
                                try:
                                    client.delete_api_key(key_id)
                                    st.success(f"Deleted '{key_name}'")
                                    st.session_state[delete_key] = False
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Failed: {e}")
                        with cc2:
                            if st.button("Cancel", key=f"key_cancel_{key_id}", type="secondary"):
                                st.session_state[delete_key] = False
                                st.rerun()
                st.divider()
    except Exception as e:
        st.error(f"Failed to load API keys: {e}")
