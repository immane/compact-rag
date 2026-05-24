"""Status badge component — colored HTML badges."""

from __future__ import annotations



_STATUS_COLORS: dict[str, str] = {
    "ok": "#28a745",
    "active": "#28a745",
    "completed": "#28a745",
    "healthy": "#28a745",
    "running": "#007bff",
    "processing": "#007bff",
    "pending": "#6c757d",
    "degraded": "#ffc107",
    "warning": "#ffc107",
    "error": "#dc3545",
    "failed": "#dc3545",
    "unknown": "#6c757d",
    "inactive": "#6c757d",
    "skipped": "#17a2b8",
}


def render_status_badge(status: str) -> str:
    color = _STATUS_COLORS.get(status.lower(), "#6c757d")
    return (
        f'<span style="display:inline-block;padding:2px 8px;border-radius:10px;'
        f'background-color:{color};color:white;font-size:12px;font-weight:600;">'
        f'{status.upper()}</span>'
    )
