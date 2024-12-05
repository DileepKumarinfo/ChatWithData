import os
import streamlit as st
from pandasai import SmartDataframe
from pandasai.callbacks import BaseCallback
from pandasai.llm import OpenAI
from pandasai.responses.response_parser import ResponseParser

from data import load_data


class StreamlitCallback(BaseCallback):
    def __init__(self, container) -> None:
        """Initialize callback handler."""
        self.container = container

    def on_code(self, response: str):
        self.container.code(response)


class StreamlitResponse(ResponseParser):
    def __init__(self, context) -> None:
        super().__init__(context)

    def format_dataframe(self, result):
        st.write("Formatting DataFrame...")
        st.dataframe(result["value"])
        return result["value"]

    def format_plot(self, result):
        st.write("Formatting Plot...")
        plot_path = result["value"]
        try:
            st.image(plot_path, caption="Fraud Amount for Top 10 CUSTOMER_IDs", use_container_width=True)
            st.success("Plot displayed successfully.")
        except Exception as e:
            st.error(f"Error displaying plot: {e}")
        return plot_path

    def format_other(self, result):
        st.write("Formatting Other...")
        st.write(result["value"])
        return result["value"]



st.write("# Chat with Credit Card Fraud Dataset 🦙")

# Debug: Show OpenAI API Key status
api_key = os.getenv("OPENAI_API_KEY")
# if not api_key:
#     st.error("OpenAI API Key is not set. Please set it before proceeding.")
# else:
#     st.success("OpenAI API Key is set.")

# Load data
try:
    df = load_data("./data")
    # st.write("Dataset loaded successfully!")
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

with st.expander("🔎 Dataframe Preview"):
    st.write(df.tail(3))

# Text area and button for submitting the query
query = st.text_area("🗣️ Chat with Dataframe")
if st.button("Submit Query"):
    if query:
        st.write("Processing your query...")

        # Set up LLM and query engine
        try:
            llm = OpenAI(api_token=api_key)
            query_engine = SmartDataframe(
                df,
                config={
                    "llm": llm,
                    "response_parser": StreamlitResponse,
                    "callback": StreamlitCallback(st.container()),
                },
            )
            
            # Process query
            try:
                answer = query_engine.chat(query)
                # st.write("Answer:")
                # st.write(answer)
            except Exception as e:
                st.error(f"Error processing query: {e}")

        except Exception as e:
            st.error(f"Error initializing query engine: {e}")
    else:
        st.warning("Please enter a query before submitting.")
