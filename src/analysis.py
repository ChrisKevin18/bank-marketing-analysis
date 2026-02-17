
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(path):
    return pd.read_csv(path)

def basic_info(df):
    print("Data Types:\n", df.dtypes)
    print("\nMissing Values:\n", df.isnull().sum())
    print("\nSummary Statistics:\n", df.describe())

def plot_target_distribution(df):
    plt.figure(figsize=(6,4))
    sns.countplot(x='y', data=df)
    plt.title("Target Variable Distribution")
    plt.savefig("images/target_distribution.png")
    plt.close()

def plot_correlation(df):
    numerical_df = df.select_dtypes(include=['int64', 'float64'])
    plt.figure(figsize=(10,6))
    sns.heatmap(numerical_df.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.savefig("images/correlation_heatmap.png")
    plt.close()

if __name__ == "__main__":
    df = load_data("data/bankmarketing.csv")
    basic_info(df)
    plot_target_distribution(df)
    plot_correlation(df)
