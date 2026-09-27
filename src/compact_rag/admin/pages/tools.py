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

    with st.container(border=True):
        st.subheader("🧪 Dry Run")
        st.caption("Test lookup and link generation without going through chat.")
        t_col1, t_col2 = st.columns(2, gap="medium")
        with t_col1:
            disease = st.text_input("Disease / symptom", value="", key="test_disease")
            if st.button("Test Lookup", key="test_lookup"):
                if not disease.strip():
                    st.warning("Enter a disease first.")
                else:
                    try:
                        result = client.test_product_lookup(disease.strip())
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
                if not pid.strip():
                    st.warning("Enter a product ID first.")
                else:
                    try:
                        result = client.test_order_link(pid.strip(), int(qty))
                        if result.get("url"):
                            st.link_button("Open order link", result["url"])
                            st.caption(f"Expires: {result.get('expires_at', '?')}")
                        else:
                            st.warning(result.get("error", "No URL generated."))
                    except Exception as e:
                        st.error(f"Generation failed: {e}")
