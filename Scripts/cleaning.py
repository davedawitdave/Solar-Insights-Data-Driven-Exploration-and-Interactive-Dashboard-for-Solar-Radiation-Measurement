class Cleaning:
    def __init__(self, data):
        self.data = data

    def impute_outliers(self):
        """
        Impute outliers in the data.
        """
        # Identify numeric columns
        numeric_cols = [
            col
            for col in self.data.columns
            if self.data[col].dtype in ["int64", "float64"]
        ]

        # Perform Z-score analysis and imputation
        for col in numeric_cols:
            self.z_score_analysis(col)
            self.impute_col(col)

    def z_score_analysis(self, col):
        """
        Perform Z-score analysis on a given column.
        """
        self.data["z_score"] = (self.data[col] - self.data[col].mean()) / self.data[
            col
        ].std()

    def impute_col(self, col):
        """
        Impute outliers in a given column.
        """
        # Replace outliers above 3 standard deviations
        outlier_mask = abs(self.data["z_score"]) > 3
        if outlier_mask.any():
            self.data.loc[outlier_mask, col] = self.impute_by_upper_quartile(col)

        # Remove the z_score column
        self.data = self.data.drop("z_score", axis=1)

    def impute_by_mean(self, col):
        """
        Impute missing values with the mean of the column.
        """
        return self.data[col].mean()

    def impute_by_median(self, col):
        """
        Impute missing values with the median of the column.
        """
        return self.data[col].median()

    def impute_by_mode(self, col):
        """
        Impute missing values with the mode of the column.
        """
        return self.data[col].mode()[0]

    def impute_by_upper_quartile(self, col):
        """
        Impute missing values with the upper quartile of the column.
        """
        return self.data[col].quantile(0.75)
