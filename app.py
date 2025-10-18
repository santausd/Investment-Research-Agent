import streamlit as st
from main import run_analysis

st.set_page_config(page_title="Investment Research Agent", layout="wide")

st.title(" Investment Research Dashboard")
st.markdown("Analyze stocks using multi-agent financial intelligence")

# Input field for stock symbol
symbol = st.text_input("Enter stock symbol:", "NVDA")

# Button to trigger analysis
if st.button(" Run Analysis"):
    with st.spinner(f"Running investment analysis for {symbol}..."):
        try:
            result = run_analysis(symbol)
            st.success(f" Analysis for {symbol} completed!")

            # Display Final Thesis
            st.subheader(" Final Investment Thesis")
            st.write(result.get("final_thesis", "No thesis generated."))

            # Display Evaluation Metrics
            if "evaluation" in result:
                eval_data = result["evaluation"]

                st.subheader(" Evaluation Summary")
                st.metric("Clarity", eval_data["clarity"])
                st.metric("Accuracy", eval_data["accuracy"])
                st.metric("Rigor", eval_data["rigor"])

                st.markdown(f"**Evaluator Source:** {eval_data.get('source', 'unknown')}")
                with st.expander("View Full Evaluation Summary"):
                    st.write(eval_data.get("evaluation_summary", "No detailed summary available."))

        except Exception as e:
            st.error(f" Error during analysis: {str(e)}")

