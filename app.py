import streamlit as st
from tools.telemetry_clash import compare_drivers, plot_clash




st.set_page_config(page_title = "F1 Data Science", page_icon= "🏎️", layout = "wide")

st.title("F1 Pit Wall")

tab_clash = st.tabs(["Telemetry Clash"])

with tab_clash[0]:
    st.header("Telemetry Clash - who gains here?")
    col1,col2,col3,col4,col5 = st.columns(5)
    year = col1.selectbox("Year", [2024, 2025,2023,2022],index=0)
    gp = col2.text_input("Grand Prix", "Bahrain")
    session_type = col3.selectbox("Session", ["Q", "R", "FP1", "FP2", "FP3"], index=0)
    driver_a = col4.text_input("Driver A", "VER")
    driver_b = col5.text_input("Driver B", "LEC")
 
    if st.button("Compare", type="primary"):
        with st.spinner("Loading session and computing deltas..."):
            results = compare_drivers(year, gp, session_type, driver_a, driver_b)
            fig = plot_clash(results)
        st.pyplot(fig)
 
        v = results["validation"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted gap", f"{v['predicted_dt_ms']:+.0f} ms")
        c2.metric("Actual gap", f"{v['actual_dt_ms']:+.0f} ms")
        c3.metric("Reconstruction error", f"{v['error_ms']:+.1f} ms")



