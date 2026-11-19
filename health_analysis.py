import numpy as np
from scipy import stats
import matplotlib.pyplot as plt 



def ci_normal(data, z=1.96):
    """
    Function that calculates the CI, 
    Returns the lower and upper confidence interval for the mean
    z is 1.96 and corresponds to 95% ,data is a pandas Series
    """
    m = data.mean()
    s = data.std(ddof=1)
    se = s/ np.sqrt(len(data))
    lower = m - z*se
    upper = m + z*se
    return lower, upper


def welch_test(smokers, non_smokers, alternative="greater"):
    """
    Function that calculates a welch test
    smokers, non_smokers: pandas Series with blood pressure values
    alternative: "greater" tests if the mean of the first group is larger.
    Returns the t statistic and the p-value.
    """
    t_stat,  p_val = stats.ttest_ind(smokers, non_smokers, equal_var=False, alternative=alternative)

    return t_stat, p_val


class HealthAnalyzer:
    """ Collects, organizes, and summarizes health data analyses. """

    def __init__(self, df):
        self.df = df.copy()

    def summary_stats(self):
        """ Calculates basic statistics (mean, median, min, max) for the numeric columns. """
      
        cols = ["age", "weight", "height", "systolic_bp", "cholesterol"]
        return self.df[cols].agg(["mean", "median", "min", "max"]).T.round(2)

    def plot_bp_histogram(self):
        """ Plots a histogram of systolic blood pressure. """

        fig, ax = plt.subplots(figsize=(8,6))
        ax.hist(self.df["systolic_bp"])
        ax.axvline(self.df["systolic_bp"].mean(), color="red", linestyle="--", label="Mean for the whole Dataset")
        ax.set_title("Histogram Over Blood Pressure")
        ax.set_xlabel("Blood pressure(mmHg)")
        ax.set_ylabel("Amount")
        ax.grid(True, axis="y")
        ax.legend()
        plt.tight_layout()
        plt.show()

    def fit_regression(self):
        """ A regression calculating the connection between blood pressure and age
           Returns x, y and the coefficients (intercept and slope). 
        """

        x = self.df[["age"]].to_numpy()
        y = self.df["systolic_bp"].to_numpy()
        ones = np.ones(len(x))
        X = np.column_stack((ones, x))
        beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
        return x, y, beta_hat
