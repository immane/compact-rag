"""Shared styling for the Streamlit administration interface."""

import streamlit as st


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] {
            background: light-dark(#f4f7fc, #101a2b);
        }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stSidebar"] {
            background: #101f39;
            border-right: 1px solid #263a59;
        }
        [data-testid="stSidebar"] * { color: #e5edff; }
        [data-testid="stSidebar"] input {
            color: #172b4d;
            background: #fff;
        }
        [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
        [data-testid="stSidebar"] .stCaption { color: #a8bbd8; }
        [data-testid="stSidebar"] .rag-brand span { color: #87b5ff; }
        [data-testid="stSidebar"] [role="radiogroup"] label {
            padding: .44rem .65rem;
            border-radius: .65rem;
            margin-bottom: .12rem;
        }
        [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
            background: #264777;
        }
        [data-testid="stSidebar"] [data-testid="stRadioOption"]
        > div > div:not([data-testid="stMarkdownContainer"]) { display: none; }
        [data-testid="stMainBlockContainer"] {
            max-width: 1360px;
            padding-top: 2.5rem;
            padding-bottom: 4rem;
        }
        [data-testid="stMainBlockContainer"] h1 {
            color: light-dark(#172b4d, #edf3ff);
            font-weight: 750;
            letter-spacing: -.04em;
        }
        [data-testid="stMainBlockContainer"] h2,
        [data-testid="stMainBlockContainer"] h3 {
            color: light-dark(#172b4d, #edf3ff);
            letter-spacing: -.025em;
        }
        [data-testid="stMainBlockContainer"] [data-testid="stVerticalBlockBorderWrapper"] {
            background: light-dark(#fff, #1b283c);
            border-color: light-dark(#e3eaf5, #34445f);
        }
        [data-testid="stVerticalBlockBorderWrapper"]:has(> div[data-testid="stVerticalBlock"]) {
            border-radius: 1rem;
        }
        [data-testid="stMetric"] {
            background: light-dark(#fff, #1b283c);
            border: 1px solid light-dark(#e3eaf5, #34445f);
            border-radius: 1rem;
            padding: 1.15rem 1.3rem;
            box-shadow: 0 10px 30px rgba(30, 60, 110, .045);
        }
        [data-testid="stMetricLabel"] { color: light-dark(#647691, #aabbd2); }
        [data-testid="stMetricValue"] { color: light-dark(#172b4d, #edf3ff); }
        [data-testid="stMainBlockContainer"] [data-testid="stForm"],
        [data-testid="stMainBlockContainer"] [data-testid="stExpander"],
        [data-testid="stMainBlockContainer"] [data-testid="stDataFrame"] {
            border-radius: 1rem;
            border-color: light-dark(#e3eaf5, #34445f);
        }
        .rag-brand {
            padding: .65rem .2rem .8rem;
            font-weight: 750;
            font-size: 1.15rem;
            letter-spacing: -.03em;
        }
        .rag-eyebrow {
            color: light-dark(#356cdd, #90b5ff);
            font-size: .74rem;
            font-weight: 750;
            letter-spacing: .12em;
            text-transform: uppercase;
        }
        .rag-subtitle { color: light-dark(#647691, #aabbd2); margin: -.65rem 0 1.4rem; }
        .rag-section-note { color: light-dark(#647691, #aabbd2); margin: -.7rem 0 1rem; }
        .rag-config-row {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            border-bottom: 1px solid light-dark(#e3eaf5, #34445f);
            padding: .65rem 0;
            font-size: .91rem;
        }
        .rag-config-row:last-child { border-bottom: 0; }
        .rag-config-row span { color: light-dark(#647691, #aabbd2); }
        .rag-config-row strong { color: light-dark(#172b4d, #edf3ff); text-align: right; overflow-wrap: anywhere; }
        .rag-mini-stat {
            display: flex;
            flex-direction: column;
            gap: .15rem;
            padding: .35rem .6rem;
            border-left: 1px solid light-dark(#e3eaf5, #34445f);
        }
        .rag-mini-stat span {
            color: light-dark(#647691, #aabbd2);
            font-size: .72rem;
            line-height: 1.2;
        }
        .rag-mini-stat strong {
            color: light-dark(#172b4d, #edf3ff);
            font-size: 1rem;
            font-weight: 650;
            line-height: 1.25;
        }
        .rag-transcript-heading {
            color: light-dark(#647691, #aabbd2);
            font-size: .78rem;
            font-weight: 700;
            letter-spacing: .08em;
            margin: .5rem 0 1rem;
            text-transform: uppercase;
        }
        .st-key-conversation-transcript [data-testid="stMarkdownContainer"] p,
        .st-key-conversation-transcript [data-testid="stMarkdownContainer"] li {
            font-size: .94rem;
            line-height: 1.65;
        }
        .st-key-conversation-transcript [data-testid="stMarkdownContainer"] h1 {
            font-size: 1.28rem;
            letter-spacing: -.02em;
        }
        .st-key-conversation-transcript [data-testid="stMarkdownContainer"] h2 {
            font-size: 1.16rem;
            letter-spacing: -.015em;
        }
        .st-key-conversation-transcript [data-testid="stMarkdownContainer"] h3 {
            font-size: 1.04rem;
            letter-spacing: 0;
        }
        .st-key-conversation-transcript [data-testid="stMarkdownContainer"] h4,
        .st-key-conversation-transcript [data-testid="stMarkdownContainer"] h5,
        .st-key-conversation-transcript [data-testid="stMarkdownContainer"] h6 {
            font-size: .96rem;
            letter-spacing: 0;
        }
        @media (max-width: 720px) {
            [data-testid="stMainBlockContainer"] { padding-top: 1.5rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
