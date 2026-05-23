"""Chart components using plotly."""

from __future__ import annotations

import streamlit as st


def render_bar_chart(data: list[dict], x: str, y: str, title: str = "") -> None:
    try:
        import plotly.express as px
        import pandas as pd

        df = pd.DataFrame(data)
        if df.empty or x not in df.columns or y not in df.columns:
            st.info("No data available for chart")
            return
        fig = px.bar(df, x=x, y=y, title=title)
        fig.update_layout(margin=dict(l=20, r=20, t=30, b=20), height=250)
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.info("Plotly not installed. Install with: pip install plotly")
    except Exception as e:
        st.warning(f"Chart error: {e}")
