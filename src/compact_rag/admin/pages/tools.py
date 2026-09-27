"""Tools page — commerce tools master switch, product/order config, dry-run tests."""

from __future__ import annotations

import streamlit as st

from compact_rag.admin.client import AdminAPIClient


def _load_config(client: AdminAPIClient) -> dict | None:
    try:
        return client.get_commerce_config()
    except Exception as e:
        st.error(f"Failed to load tools config: {e}")
        return None


def render(client: AdminAPIClient) -> None:
    st.markdown('<div class="rag-eyebrow">Workspace / Tools</div>', unsafe_allow_html=True)
    st.title("🛠️ Tools")
    st.markdown(
        '<p class="rag-subtitle">Configure product lookup and order-link generation used by chat.</p>',
        unsafe_allow_html=True,
    )

    cfg = _load_config(client)
    if cfg is None:
        return

    products = cfg.get("products", {})
    order = cfg.get("order", {})

    with st.container(border=True):
        st.subheader("🔌 Master Switch")
        enabled = st.toggle(
            "Enable commerce tools (lookup_products / create_order_link)",
            value=bool(cfg.get("commerce_enabled", True)),
            key="tools_enabled",
        )
        st.caption(
            "When off, chat answers never trigger product lookup or order links."
        )
        if st.button("Save Switch", type="primary", key="save_switch"):
            try:
                client.update_commerce_config({"commerce_enabled": enabled})
                st.success("Saved.")
                st.rerun()
            except Exception as e:
                st.error(f"Save failed: {e}")

    st.write("")

    col_left, col_right = st.columns(2, gap="medium")

    with col_left:
        with st.container(border=True):
            st.subheader("📦 Product Lookup")
            st.caption("Knowledge-base collection first, external API as fallback.")
            p_collection = st.text_input(
                "Product collection", value=products.get("collection", "products")
            )
            p_top_k = st.number_input(
                "Top-K", min_value=1, max_value=50,
                value=int(products.get("top_k", 5) or 5),
            )
            p_api_base = st.text_input(
                "External product API base (optional)",
                value=products.get("api_base", ""),
                placeholder="https://product-api.example.com",
            )
            p_api_key = st.text_input(
                "External product API key",
                value="",
                type="password",
                placeholder="空 = 保持不变"
                if products.get("api_key_configured")
                else "Optional",
            )
            if products.get("api_key_configured"):
                st.caption("✅ API key already configured (leave blank to keep).")
            if st.button("Save Products", type="primary", key="save_products"):
                patch: dict = {
                    "products": {
                        "collection": p_collection,
                        "top_k": p_top_k,
                        "api_base": p_api_base,
                    }
                }
                if p_api_key:
                    patch["products"]["api_key"] = p_api_key
                try:
                    client.update_commerce_config(patch)
                    st.success("Saved.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Save failed: {e}")

    with col_right:
        with st.container(border=True):
            st.subheader("🛒 Order Links")
            st.caption(f"Mode: **{order.get('mode', 'unconfigured')}**")
            o_api_base = st.text_input(
                "Order service base URL (optional, takes precedence)",
                value=order.get("api_base", ""),
                placeholder="https://order-api.example.com",
            )
            o_api_key = st.text_input(
                "Order service API key",
                value="",
                type="password",
                placeholder="空 = 保持不变"
                if order.get("api_key_configured")
                else "Optional",
            )
            if order.get("api_key_configured"):
                st.caption("✅ API key already configured (leave blank to keep).")
            o_create_path = st.text_input(
                "Create-link path", value=order.get("create_path", "/orders/link")
            )
            o_template = st.text_area(
                "URL template (template mode)",
                value=order.get("url_template", ""),
                placeholder="https://shop.example.com/order?product={product_id}&qty={quantity}&exp={expires}&sig={signature}",
            )
            o_secret = st.text_input(
                "Signing secret (template mode)",
                value="",
                type="password",
                placeholder="空 = 保持不变"
                if order.get("signing_secret_configured")
                else "Required for template mode",
            )
            if order.get("signing_secret_configured"):
                st.caption("✅ Signing secret already configured (leave blank to keep).")
            o_ttl = st.number_input(
                "Link TTL (minutes)", min_value=1, max_value=1440,
                value=int(order.get("link_ttl_minutes", 30) or 30),
            )
            if st.button("Save Order", type="primary", key="save_order"):
                order_patch: dict = {
                    "api_base": o_api_base,
                    "create_path": o_create_path,
                    "url_template": o_template,
                    "link_ttl_minutes": o_ttl,
                }
                if o_api_key:
                    order_patch["api_key"] = o_api_key
                if o_secret:
                    order_patch["signing_secret"] = o_secret
                try:
                    client.update_commerce_config({"order": order_patch})
                    st.success("Saved.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Save failed: {e}")

    st.write("")

    _render_sources(client)

    st.write("")

    with st.container(border=True):
        st.subheader("🧪 Dry Run")
        st.caption("Test lookup and link generation without going through chat.")
        t_col1, t_col2 = st.columns(2, gap="medium")
        with t_col1:
            disease = st.text_input("Disease / symptom", value="", key="test_disease")
            if st.button("Test Lookup", key="test_lookup"):
                if not (disease or "").strip():
                    st.warning("Enter a disease first.")
                else:
                    try:
                        result = client.test_product_lookup((disease or "").strip())
                        items = result.get("products", [])
                        if not items:
                            st.info(result.get("error", "No products found."))
                        else:
                            for p in items:
                                st.markdown(
                                    f"- **{p.get('name', '?')}** "
                                    f"`{p.get('product_id', '')}` "
                                    f"(score: {p.get('score', 0)}, via {p.get('source', '?')})"
                                )
                    except Exception as e:
                        st.error(f"Lookup failed: {e}")
        with t_col2:
            pid = st.text_input("Product ID", value="", key="test_pid")
            qty = st.number_input("Quantity", min_value=1, value=1, key="test_qty")
            if st.button("Test Order Link", key="test_order"):
                if not (pid or "").strip():
                    st.warning("Enter a product ID first.")
                else:
                    try:
                        result = client.test_order_link((pid or "").strip(), int(qty))
                        if result.get("url"):
                            st.link_button("Open order link", result["url"])
                            st.caption(f"Expires: {result.get('expires_at', '?')}")
                        else:
                            st.warning(result.get("error", "No URL generated."))
                    except Exception as e:
                        st.error(f"Generation failed: {e}")


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
            c1, c2, c3, c4 = st.columns([3, 2, 2, 2])
            with c1:
                status = "🟢" if s.get("enabled") else "⚪"
                st.markdown(f"**{status} {name}** → `{s.get('collection', '')}`")
                st.caption(f"Last sync: {s.get('last_sync') or '-'} · {summary}")
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
            with c4:
                st.caption(f"token: {'✅' if s.get('auth_token_configured') else '❌'}")
            st.divider()

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
