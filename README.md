# Solar Insights: Interactive Exploration and Dashboard for Solar Radiation Data

## Project Overview

This project aims to develop an interactive dashboard and data exploration tool for solar radiation measurement data. The project involves analyzing and visualizing data collected from multiple locations, with a focus on key metrics like Global Horizontal Irradiance (GHI), Direct Normal Irradiance (DNI), and Diffuse Horizontal Irradiance (DHI). The goal is to provide meaningful insights into solar radiation patterns and facilitate better decision-making in the solar energy domain.

## Business Objectives

1. **Understand Solar Radiation Patterns**: Analyze the solar radiation data to identify trends and patterns that can inform solar energy generation strategies.
2. **Data-Driven Decision Making**: Provide an interactive tool for stakeholders to explore solar radiation data, aiding in the planning and optimization of solar energy systems.
3. **Visualization and Reporting**: Create a user-friendly dashboard that allows for easy visualization of complex data, supporting both exploratory analysis and formal reporting.

## Key Features

- **Interactive Dashboard**: Built using Streamlit, allowing users to explore data through various visualizations.
- **Time Series Analysis**: Analyze solar radiation and related metrics over time, with options to aggregate data by day, week, month, or year.
- **Correlation Analysis**: Identify and visualize correlations between different metrics to uncover relationships and potential causality.
- **Data Quality Checks**: Ensure the integrity of the data by identifying missing values, outliers, and incorrect entries.

## Data

The data includes solar radiation measurements and related environmental metrics from multiple locations. Key columns include:
- **'GHI'**: Global Horizontal Irradiance (W/m²)
- **'DNI'**: Direct Normal Irradiance (W/m²)
- **'DHI'**: Diffuse Horizontal Irradiance (W/m²)
- **'ModA'**: Sensor A readings
- **'ModB'**: Sensor B readings
- **'Tamb'**: Ambient temperature
- **'TModA'** and **'TModB'**: Module temperatures
- **'WS'**: Wind Speed
- **'BP'**: Barometric Pressure

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/solar-insights.git

