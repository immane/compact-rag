"""Admin-defined tools and dynamic API data sources."""

from __future__ import annotations

import json

import streamlit as st

from compact_rag.admin.client import AdminAPIClient


def _load_config(client: AdminAPIClient) -> dict | None:
    try:
        return client.get_tools_config()
    except Exception as e:
        st.error(f"Failed to load tools config: {e}")
        return None


def render(client: AdminAPIClient) -> None:
    st.markdown('<div class="rag-eyebrow">Workspace / Tools</div>', unsafe_allow_html=True)
    st.title("🛠️ Tools")
    st.markdown(
        '<p class="rag-subtitle">Define chat tools dynamically, connect APIs, and configure scheduled data sources.</p>',
        unsafe_allow_html=True,
    )

    cfg = _load_config(client)
    if cfg is None:
        return

    with st.container(border=True):
        st.subheader("🔌 Tools")
        enabled = st.toggle(
            "Enable configured tools in chat",
            value=bool(cfg.get("enabled", True)),
            key="tools_enabled",
        )
        st.caption("The function names, request schemas, endpoints, and authentication headers below are all user-defined.")

    tools = cfg.get("tools", [])
    names = [item.get("name", "") for item in tools]
    selected = st.selectbox("Edit tool", ["(new tool)", *names], key="dynamic_tool_select")
    current = next((item for item in tools if item.get("name") == selected), {})
    editor_current = dict(current)
    editor_current.pop("headers_configured", None)
    current_text = json.dumps(editor_current, ensure_ascii=False, indent=2) if current else json.dumps(
        {
            "name": "search_records",
            "description": "Search records relevant to the query",
            "kind": "vector_search",
            "enabled": True,
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Search query"}},
                "required": ["query"],
            },
            "collection": "default",
            "query_argument": "query",
            "top_k": 5,
        }, ensure_ascii=False, indent=2,
    )
    edited = st.text_area(
        "Tool definition (JSON)", value=current_text, height=360,
        help="kind=http supports GET/POST and {{argument}} templates in URL/headers/query/body. kind=vector_search searches a collection.",
        key="dynamic_tool_json",
    )
    if tools:
        st.caption("Configured tools: " + " · ".join(names))
    save_col, delete_col = st.columns([1, 5])
    with save_col:
        if st.button("Save Tool", type="primary", key="save_dynamic_tool"):
            try:
                definition = json.loads(edited)
                if not isinstance(definition, dict):
                    raise ValueError("Tool definition must be a JSON object")
                new_tools = [tool for tool in tools if tool.get("name") != selected]
                new_tools = [tool for tool in new_tools if tool.get("name") != definition.get("name")]
                new_tools.append(definition)
                client.save_tools_config(enabled, new_tools)
                st.success(f"Saved tool '{definition.get('name', '')}'")
                st.rerun()
            except Exception as e:
                st.error(f"Save failed: {e}")
    with delete_col:
        if selected != "(new tool)" and st.button("Delete Tool", key="delete_dynamic_tool"):
            try:
                client.save_tools_config(enabled, [tool for tool in tools if tool.get("name") != selected])
                st.success(f"Deleted '{selected}'")
                st.rerun()
            except Exception as e:
                st.error(f"Delete failed: {e}")

    with st.expander("Test configured tool"):
        if names:
            test_name = st.selectbox("Tool", names, key="dynamic_tool_test_name")
            test_args_text = st.text_area("Arguments (JSON)", "{}", key="dynamic_tool_test_args")
            if st.button("Run tool", key="run_dynamic_tool_test"):
                try:
                    arguments = json.loads(test_args_text)
                    if test_name:
                        st.json(client.test_dynamic_tool(test_name, arguments))
                except Exception as e:
                    st.error(f"Tool test failed: {e}")
        else:
            st.info("Add a tool definition to enable testing.")

    st.write("")

    _render_sources(client)

    st.write("")

def _render_sources(client: AdminAPIClient) -> None:
    with st.container(border=True):
        st.subheader("🗂️ Dynamic Sources")
        st.caption(
            "Paginated API sources (comments, reports, …) synced into "
            "collections. Triggered daily by an external cron calling "
            "`POST /v1/ingestion/sources/sync`."
        )
        try:
            sources = client.get_sources().get("sources", [])
        except Exception as e:
            st.error(f"Failed to load sources: {e}")
            return

        if not sources:
            st.info("No sources configured yet.")
        for s in sources:
            name = s.get("name", "?")
            last = s.get("last_result") or {}
            summary = (
                f"fetched {last.get('fetched', '?')}, "
                f"+{last.get('completed', '?')}/={last.get('skipped', '?')}/"
                f"x{last.get('failed', '?')}"
                if last
                else "never synced"
            )
            with st.container(border=True):
                c1, c2, c3 = st.columns([3.5, 1, 1], vertical_alignment="center")
                with c1:
                    status = "Enabled" if s.get("enabled") else "Disabled"
                    st.markdown(f'<div class="rag-card-title">{name} <span class="rag-card-description">· {status}</span></div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="rag-card-description">Collection · {s.get("collection", "")}</div>', unsafe_allow_html=True)
                    st.caption(f"Last sync: {s.get('last_sync') or 'Never'} · {summary} · token {'configured' if s.get('auth_token_configured') else 'not configured'}")
                with c2:
                    if st.button("🔄 Sync", key=f"sync_{name}"):
                        try:
                            with st.spinner(f"Syncing {name}…"):
                                result = client.sync_sources(source=name)
                            st.json(result)
                        except Exception as e:
                            st.error(f"Sync failed: {e}")
                with c3:
                    if st.button("🗑️ Delete", key=f"del_src_{name}"):
                        try:
                            client.delete_source(name)
                            st.success(f"Deleted '{name}'")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Delete failed: {e}")

        if st.button("🔄 Sync All Enabled", key="sync_all"):
            try:
                with st.spinner("Syncing all sources…"):
                    result = client.sync_sources()
                st.json(result)
            except Exception as e:
                st.error(f"Sync failed: {e}")

        names = [s.get("name", "") for s in sources]
        choice = st.selectbox("Edit source", ["(new source)", *names], key="src_edit")
        current = next((s for s in sources if s.get("name") == choice), {})

        def _text(key: str, default: str = "") -> str:
            value = current.get(key, default)
            return str(value) if value is not None else default

        with st.form("source_form"):
            f_name = st.text_input("Name (a-z, 0-9, -, _)", value=_text("name"))
            f_collection = st.text_input("Target collection", value=_text("collection", "default"))
            f_base = st.text_input("Base URL", value=_text("base_url"))
            f_path = st.text_input("Path", value=_text("path", "/"))
            f_auth_header = st.text_input("Auth header", value=_text("auth_header", "x-auth-token"))
            f_token = st.text_input(
                "Auth token", value="", type="password",
                placeholder="空 = 保持不变" if current.get("auth_token_configured") else "Required",
            )
            c1, c2, c3 = st.columns(3)
            with c1:
                f_page_param = st.text_input("Page param", value=_text("page_param", "page"))
                f_page_size = st.number_input("Page size", min_value=1, max_value=1000,
                                              value=int(current.get("page_size", 100) or 100))
            with c2:
                f_items = st.text_input("Items path", value=_text("items_path", "data"))
                f_code = st.text_input("Code path", value=_text("code_path", "code"))
                f_success = st.number_input("Success code", value=int(current.get("success_code", 0) or 0))
            with c3:
                f_paginator = st.text_input("Paginator path", value=_text("paginator_path", "paginator"))
                f_current = st.text_input("Current-page field", value=_text("current_page_field", "current"))
                f_last = st.text_input("Last-page field", value=_text("last_page_field", "last"))
            f_id = st.text_input("ID field", value=_text("id_field", "id"))
            f_updated = st.text_input("Updated-at field (empty = full sync each time)",
                                      value=_text("updated_at_field"))
            f_title = st.text_input("Title template", value=_text("title_template", "#{id}"))
            body_default = current.get("body_fields", ["content"])
            f_body = st.text_input("Body fields (comma separated)",
                                   value=", ".join(body_default) if isinstance(body_default, list) else str(body_default or ""))
            f_enabled = st.checkbox("Enabled", value=bool(current.get("enabled", True)))
            submitted = st.form_submit_button("Save Source", type="primary")
        if submitted:
            if not (f_name or "").strip():
                st.warning("Name is required.")
                return
            entry = {
                "name": (f_name or "").strip(),
                "collection": (f_collection or "").strip() or "default",
                "base_url": (f_base or "").strip(),
                "path": (f_path or "").strip() or "/",
                "auth_header": (f_auth_header or "").strip() or "x-auth-token",
                "auth_token": f_token or "",
                "page_param": (f_page_param or "").strip() or "page",
                "page_size": int(f_page_size),
                "items_path": (f_items or "").strip() or "data",
                "code_path": (f_code or "").strip() or "code",
                "success_code": int(f_success),
                "paginator_path": (f_paginator or "").strip() or "paginator",
                "current_page_field": (f_current or "").strip() or "current",
                "last_page_field": (f_last or "").strip() or "last",
                "id_field": (f_id or "").strip() or "id",
                "updated_at_field": (f_updated or "").strip() or None,
                "title_template": (f_title or "").strip() or "#{id}",
                "body_fields": [b.strip() for b in (f_body or "").split(",") if b.strip()] or ["content"],
                "enabled": bool(f_enabled),
            }
            if choice not in ("(new source)", entry["name"]):
                others = [s for s in sources if s.get("name") != choice]
            else:
                others = [s for s in sources if s.get("name") != entry["name"]]
            try:
                client.save_sources([*others, entry])
                st.success(f"Saved '{entry['name']}'")
                st.rerun()
            except Exception as e:
                st.error(f"Save failed: {e}")
