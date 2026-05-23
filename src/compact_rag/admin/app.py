"""Streamlit admin dashboard main entry point."""

from __future__ import annotations

import streamlit as st

from compact_rag.admin.client import AdminAPIClient
from compact_rag.admin.config import get_api_base_url, is_auth_required, verify_password

st.set_page_config(
    page_title="Compact-RAG Admin",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Authentication ───────────────────────────────────────────

AUTH_REQUIRED = is_auth_required()

if "authenticated" not in st.session_state:
    st.session_state.authenticated = not AUTH_REQUIRED

if not st.session_state.authenticated:
    st.title("🔍 Compact-RAG Admin")
    st.markdown("### Authentication Required")
    st.caption(
        "Set `ADMIN_PASSWORD` environment variable to protect the admin dashboard. "
        "Leave unset for development (local access only)."
    )

    with st.form("login_form"):
        password = st.text_input("Password", type="password", placeholder="Enter admin password")
        submitted = st.form_submit_button("Login", use_container_width=True)

        if submitted:
            if verify_password(password):
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Invalid password")

    st.stop()

# ── Logout button ────────────────────────────────────────────

if AUTH_REQUIRED:
    if st.sidebar.button("🚪 Logout", use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()

# ── Session Setup ────────────────────────────────────────────

if "api_base_url" not in st.session_state:
    st.session_state.api_base_url = get_api_base_url()

# Rebuild client on each rerun so session state never keeps a stale HTTP session.
st.session_state.client = AdminAPIClient(base_url=st.session_state.api_base_url)


def _check_api_health() -> tuple[bool, str]:
    try:
        client = AdminAPIClient(base_url=st.session_state.api_base_url)
        health_data = client.health()
        statuses = []
        for component in ["api", "database", "chromadb", "storage"]:
            val = health_data.get(component, "unknown")
            statuses.append(f"{component}: {val}")
        return True, " | ".join(statuses)
    except Exception as e:
        return False, str(e)


st.sidebar.title("🔍 Compact-RAG Admin")

url_input = st.sidebar.text_input(
    "API Base URL",
    value=st.session_state.api_base_url,
    key="api_url_input",
)
if url_input != st.session_state.api_base_url:
    st.session_state.api_base_url = url_input
    st.session_state.client = AdminAPIClient(base_url=url_input)
    st.rerun()

if st.sidebar.button("🔌 Test Connection", use_container_width=True):
    ok, msg = _check_api_health()
    if ok:
        st.sidebar.success(msg)
    else:
        st.sidebar.error(f"Connection failed: {msg}")

PAGES = {
    "📊 Dashboard": "dashboard",
    "📁 Collections": "collections",
    "📄 Documents": "documents",
    "⚙️ Ingestion": "ingestion",
    "💬 Conversations": "conversations",
    "🎮 Playground": "playground",
    "🔑 API Keys": "api_keys",
    "💾 Storage": "storage",
}

page_choice = st.sidebar.radio("Navigation", list(PAGES.keys()), key="nav")
page_module = PAGES[page_choice]

client = st.session_state.client


def _render_page(module_name: str) -> None:
    try:
        if module_name == "dashboard":
            from compact_rag.admin.pages.dashboard import render as render_page
        elif module_name == "collections":
            from compact_rag.admin.pages.collections import render as render_page
        elif module_name == "documents":
            from compact_rag.admin.pages.documents import render as render_page
        elif module_name == "ingestion":
            from compact_rag.admin.pages.ingestion import render as render_page
        elif module_name == "conversations":
            from compact_rag.admin.pages.conversations import render as render_page
        elif module_name == "playground":
            from compact_rag.admin.pages.playground import render as render_page
        elif module_name == "api_keys":
            from compact_rag.admin.pages.api_keys import render as render_page
        elif module_name == "storage":
            from compact_rag.admin.pages.storage import render as render_page
        else:
            st.error(f"Unknown page: {module_name}")
            return
        render_page(client)
    except Exception as e:
        st.error(f"Failed to load page: {e}")
        st.warning("Make sure the API server is running and accessible.")


_render_page(page_module)
