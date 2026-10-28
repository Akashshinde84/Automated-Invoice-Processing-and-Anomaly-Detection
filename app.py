"""Minimal Streamlit landing page for the phase-one scaffold."""

import streamlit as st


def main() -> None:
    st.set_page_config(page_title="Invoice Processing", layout="wide")
    st.title("Automated Invoice Processing and Anomaly Detection")
    st.info("Phase 1: project setup complete. Processing features are planned.")
    st.markdown(
        "Future phases will add OCR, invoice parsing, validation, anomaly and "
        "duplicate detection, CSV exports, and dashboard visualizations."
    )


if __name__ == "__main__":
    main()
