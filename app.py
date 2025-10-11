import streamlit as st
from main import run_analysis

st.set_page_config(page_title="Investment Research Agent", layout="wide")

st.title("Investment Research Dashboard")
st.markdown("Analyze stocks using multi-agent financial intelligence")

# Input field for stock symbol
symbol = st.text_input("Enter stock symbol:", "NVDA")

# Button to trigger analysis
if st.button("Run Analysis"):
    with st.spinner(f"Running investment analysis for {symbol}..."):
        try:
            result = run_analysis(symbol)
            st.success(f"Analysis for {symbol} completed!")
            st.write(result)  # Display output (dict, df, or summary text)
        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")

