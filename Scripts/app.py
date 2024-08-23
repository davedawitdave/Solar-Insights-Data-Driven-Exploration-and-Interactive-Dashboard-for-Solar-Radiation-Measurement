import streamlit as st
import pandas as pd
from pathlib import Path
from cleaning import Cleaning
from plots import Charts

# Set the base directory to the notebook folder
base_dir = Path(__file__).parent.parent

# Load the data
csv_path = base_dir / "data" / "west_3_countries.csv"
data = pd.read_csv(csv_path)


# Initialize the Charts class
chart = Charts(data)

# Create a sidebar for selecting visualizations
st.sidebar.title("Visualization Selector")
plot_type = st.sidebar.selectbox(
    "Choose a plot type", ["Correlation Matrix", "Time Series", "Boxplot"]
)

# Display visualizations based on selection
if plot_type == "Correlation Matrix":
    st.title("Correlation Matrix")
    chart.correlation_matrix()
elif plot_type == "Time Series":
    st.title("Time Series")
    variables = st.multiselect("Select variables to plot", data.columns)
    aggregation = st.selectbox(
        "Aggregation Level", ["Daily", "Weekly", "Monthly", "Yearly"]
    )
    chart.time_series(variables, agg_level=aggregation)
elif plot_type == "Boxplot":
    st.title("Boxplot")
    column = st.selectbox("Select a column for boxplot", data.columns)
    chart.boxplot(column)
