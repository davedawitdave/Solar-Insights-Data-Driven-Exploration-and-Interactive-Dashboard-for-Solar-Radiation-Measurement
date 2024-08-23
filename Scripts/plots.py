import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np


class Charts:
    def __init__(self, data):
        """
        Initializes the Charts class with the given dataset.

        Parameters:
        data (pd.DataFrame): The dataframe containing the data to be visualized.

        """
        self.data = data

    def correlation_heatmap(self, title="heatmap"):
        """
        Displays a heatmap of the correlation matrix of the dataframe.
        """
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            self.data.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5
        )
        plt.title(title)
        plt.show()

    def time_series_analysis(self, columns, date_column, aggregations=["D", "M", "Y"]):
        """
        Displays time series plots for the specified columns, with options to aggregate data by day, week, month, or year.

        Parameters:
        columns (list of str): The columns to plot.
        aggregation (str): Aggregation level ('D' for daily, 'W' for weekly, 'M' for monthly, 'Y' for yearly).
        """
        # Convert the date_column to datetime and set it as the index
        if date_column not in self.data.columns:
            raise KeyError(f"{date_column} not found in DataFrame columns.")

        self.data[date_column] = pd.to_datetime(self.data[date_column])
        self.data.set_index(date_column, inplace=True)

        # Prepare for plotting
        for aggregation in aggregations:
            aggregated_data = self.data.resample(aggregation).mean()

            # Check if data is empty after aggregation
            if aggregated_data.empty:
                print(f"No data available for {aggregation} aggregation.")
                continue

            # Customize background
            plt.figure(figsize=(14, 6))
            plt.gca().set_facecolor("lightgrey")  # Set background color
            sns.set_style("whitegrid")  # Set seaborn style

            for column in columns:
                if column not in aggregated_data.columns:
                    print(
                        f"Warning: {column} not found in aggregated data for {aggregation}."
                    )
                    continue

                sns.lineplot(
                    data=aggregated_data,
                    x=aggregated_data.index,
                    y=column,
                    label=column,
                )

            plt.title(f"Time Series Analysis ({aggregation})")
            plt.xlabel("Time")
            plt.ylabel("Values")
            plt.legend()
            plt.grid(True)
            plt.show()

    def histogram(
        self,
        column,
        bins=8,
        title="Histogram",
        xlabel="Values",
        figsize=(10, 6),
        num_cols=1,
    ):
        """
        Creates a histogram or a grid of histograms for the specified column(s).

        Parameters:
        column (str or list): A single column name or a list of column names to create histograms for.
        bins (int): The number of bins for the histogram(s).
        title (str or list): The title of the histogram(s). Can be a single string or a list of titles.
        xlabel (str or list): The label for the x-axis. Can be a single string or a list of labels.
        figsize (tuple): The size of the figure (width, height).
        num_cols (int): The number of columns in the grid layout (only used when multiple columns are provided).
        """
        # Ensure the column(s) exist in the DataFrame
        if isinstance(column, str):
            column = [column]
        for col in column:
            if col not in self.data.columns:
                raise KeyError(f"{col} not found in DataFrame.")

        num_rows = (len(column) + num_cols - 1) // num_cols

        plt.figure(figsize=figsize)
        for i, col in enumerate(column):
            plt.subplot(num_rows, num_cols, i + 1)
            sns.histplot(
                self.data[col], bins=bins, kde=True, color="skyblue", edgecolor="black"
            )
            plt.title(title[i] if isinstance(title, list) else title, fontsize=14)
            plt.xlabel(xlabel[i] if isinstance(xlabel, list) else xlabel, fontsize=12)
            plt.ylabel("Frequency", fontsize=12)
            plt.grid(True)

        plt.tight_layout()
        plt.show()

    def scatter_plot(self, x_col, y_col, title="Scatter Plot"):
        """
        Creates a scatter plot between two variables.

        Parameters:
        x_col (str): The column name for the x-axis.
        y_col (str): The column name for the y-axis.
        title (str): The title of the scatter plot.
        """
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=self.data, x=x_col, y=y_col)
        plt.title(title)
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.grid(True)
        plt.show()

    def pairplot(self):
        """
        Generates a pairplot for the dataframe to visualize relationships between pairs of features.
        """
        sns.pairplot(self.data)
        plt.show()

    def boxplot(self, column, title="Boxplot"):
        """
        Displays a boxplot for the specified column.

        Parameters:
        column (str): The column name to plot the boxplot for.
        title (str): The title of the boxplot.
        """
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=self.data[column], palette="pastel")
        plt.title(title)
        plt.grid(True)
        plt.show()

    def bar_chart(self, x_col, y_col, title="Bar Chart"):
        """
        Plots a bar chart for the specified x and y columns.

        Parameters:
        x_col (str): The column name for the x-axis.
        y_col (str): The column name for the y-axis.
        title (str): The title of the bar chart.
        """
        plt.figure(figsize=(10, 6))
        sns.barplot(data=self.data, x=x_col, y=y_col)
        plt.title(title)
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.grid(True)
        plt.show()

    def bubble_chart(
        self,
        x_col,
        y_col,
        size_col,
        color_col=None,
        tooltip_cols=None,
        title="Bubble Chart",
    ):
        """
        Generates a bubble chart, where the size of bubbles represents another variable.

        Parameters:
        x_col (str): The column name for the x-axis.
        y_col (str): The column name for the y-axis.
        size_col (str): The column name that determines the size of the bubbles.
        color_col (str, optional): The column name for coloring the bubbles.
        title (str): The title of the bubble chart.
        """
        missing_cols = [
            col
            for col in [x_col, y_col, size_col]
            + ([] if color_col is None else [color_col])
            + ([] if tooltip_cols is None else tooltip_cols)
            if col not in self.data.columns
        ]
        if missing_cols:
            raise KeyError(
                f"The following columns were not found in the DataFrame: {', '.join(missing_cols)}"
            )

        trace = go.Scatter(
            x=self.data[x_col],
            y=self.data[y_col],
            mode="markers",
            marker=dict(
                size=self.data[size_col],
                color=self.data[color_col] if color_col else "blue",
                opacity=0.7,
            ),
            text=(
                self.data[tooltip_cols].to_string(index=False) if tooltip_cols else None
            ),
            hovertemplate=(
                "<br>".join(
                    [
                        f"{x_col}: %{{x:.2f}}",
                        f"{y_col}: %{{y:.2f}}",
                        f"{size_col}: %{{marker.size:.2f}}",
                        "%{text}",
                    ]
                )
                if tooltip_cols
                else None
            ),
        )

        fig = go.Figure(data=[trace])
        fig.update_layout(
            title=title,
            xaxis_title=x_col,
            yaxis_title=y_col,
            xaxis_range=[self.data[x_col].min(), self.data[x_col].max()],
            yaxis_range=[self.data[y_col].min(), self.data[y_col].max()],
            width=800,
            height=600,
        )
        fig.show()

    def z_score_analysis(self, column, title="Z-Score Analysis"):
        """
        Calculates and plots the Z-score for a specified column to identify outliers.

        Parameters:
        column (str): The column name to perform Z-score analysis on.
        title (str): The title of the Z-score plot.
        """
        self.data["z_score"] = (
            self.data[column] - self.data[column].mean()
        ) / self.data[column].std()
        plt.figure(figsize=(10, 6))
        plt.plot(self.data["z_score"])
        plt.axhline(y=3, color="r", linestyle="--")
        plt.axhline(y=-3, color="r", linestyle="--")
        plt.title(title)
        plt.xlabel("Index")
        plt.ylabel("Z-Score")
        plt.grid(True)
        plt.show()

    def polar_plot(self, speed_col, direction_col, title="Wind Polar Plot"):
        """
        Creates a polar scatter plot for wind speed and direction.

        Parameters:
        speed_col (str): The column name representing wind speed.
        direction_col (str): The column name representing wind direction.
        title (str): The title of the wind polar plot.
        """
        # Ensure the columns exist
        if speed_col not in self.data.columns or direction_col not in self.data.columns:
            raise KeyError(f"{speed_col} or {direction_col} not found in DataFrame.")

        # Convert wind direction from degrees to radians
        wind_dir_rad = np.radians(self.data[direction_col])

        # Create a figure for the polar plot
        plt.figure(figsize=(10, 8))

        # Create a polar plot
        ax = plt.subplot(111, projection="polar")
        ax.set_theta_zero_location("N")  # North at the top
        ax.set_theta_direction(-1)  # Clockwise direction

        # Plot wind speed data
        ax.scatter(wind_dir_rad, self.data[speed_col], alpha=0.6, edgecolors="w")

        # Set labels
        ax.set_title("Wind Speed and Direction", va="bottom", fontsize=16)
        ax.set_xlabel("Wind Direction (Degrees)")
        ax.set_ylabel("Wind Speed")

        plt.grid(True)
        plt.show()


# Example usage:
# charts = Charts(data)
# charts.correlation_heatmap()
# charts.timeseries_plot('Timestamp', 'GHI')
# charts.histogram('GHI')
# charts.scatter_plot('GHI', 'DNI')
# charts.pairplot()
# charts.boxplot('GHI')
# charts.bar_chart('WD', 'WS')
# charts.bubble_chart('GHI', 'Tamb', 'RH')
# charts.z_score_analysis('GHI')
# charts.polar_plot('WS', 'WD')
