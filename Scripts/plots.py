import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pandas as pd


class Charts:
    def __init__(self, data):
        """
        Initializes the Charts class with the given dataset.

        Parameters:
        data (pd.DataFrame): The dataframe containing the data to be visualized.

        """
        self.data = data

    def correlation_heatmap(self):
        """
        Displays a heatmap of the correlation matrix of the dataframe.
        """
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            self.data.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5
        )
        plt.title("Correlation Heatmap")
        plt.show()

    def timeseries_plot(self, x_col, y_col, title="Time Series Plot"):
        """
        Plots a time series graph for a given x and y column.

        Parameters:
        x_col (str): The column name for the x-axis (typically 'Timestamp').
        y_col (str): The column name for the y-axis.
        title (str): The title of the plot.
        """
        plt.figure(figsize=(14, 6))
        plt.plot(self.data[x_col], self.data[y_col])
        plt.title(title)
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.grid(True)
        plt.show()

    def histogram(self, column, bins=30, title="Histogram"):
        """
        Creates a histogram for the specified column.

        Parameters:
        column (str): The column name to plot the histogram for.
        bins (int): Number of bins for the histogram.
        title (str): The title of the histogram.
        """
        plt.figure(figsize=(10, 6))
        plt.hist(self.data[column], bins=bins, edgecolor="black")
        plt.title(title)
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.grid(True)
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
        sns.boxplot(y=self.data[column])
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
        self, x_col, y_col, size_col, color_col=None, title="Bubble Chart"
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
        fig = px.scatter(
            self.data,
            x=x_col,
            y=y_col,
            size=size_col,
            color=color_col,
            title=title,
            size_max=60,
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
        fig = px.scatter_polar(
            self.data,
            r=speed_col,
            theta=direction_col,
            color=speed_col,
            title=title,
            color_continuous_scale=px.colors.sequential.Viridis,
        )
        fig.show()


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
