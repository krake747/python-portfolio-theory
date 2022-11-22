import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from cvxopt import matrix, solvers
sns.set_style("darkgrid")
solvers.options["show_progress"] = False


def log_returns(x):
    """This function transforms stock prices into logarithmic returns for an entire dataframe."""
    return np.log(x.pct_change()+1)


def simple_returns(x):
    """This function transforms stock prices into simple returns for an entire dataframe."""
    return x.pct_change()


def sharp_ratio(sigma, mu, weights, rfr):
    return (mu @ weights + rfr) / np.sqrt(weights @ sigma @ weights)


def minimum_variance_portfolio(sigma, lb: float, ub: float):
    """This function calculates the weights of the Minimum Variance Portfolio"""
    n = sigma.shape[1]

    P = matrix(np.array(sigma), tc="d")
    q = matrix(np.zeros(n), tc="d")

    lb_diag = np.diag(np.repeat(-1, n))
    ub_diag = np.diag(np.repeat(1, n))

    G = matrix(np.concatenate((lb_diag, ub_diag)), tc="d")
    h = matrix(np.concatenate((np.repeat(-lb, n), np.repeat(ub, n))), tc="d")

    A = matrix(np.repeat(1, n), tc="d").T
    b = matrix(1.0, tc="d")

    sol = solvers.qp(P, q, G, h, A, b)
    return np.array(sol["x"]).flatten()


def maximum_return_portfolio(sigma, mu, lb: float, ub: float):
    """This function calculates the weights of the Maximum Return Portfolio"""
    if (lb < -1 or ub > 1):
        raise Exception(f"Out of Bonds [{lb}, {ub}]. Confine lb and ub within a range of [-1.0 and 1.0]")

    if (lb < 0):
        return _calc_mrp_with_short_sale(sigma, mu, lb, ub)

    return _calc_mrp_without_short_sale(sigma, mu, lb, ub)


def efficient_frontier_portfolio(sigma, mu, lb: float, ub: float, target_mu: float = None):
    """This function calculates the weights of an Efficient Frontier Portfolio"""
    if (lb < -1 or ub > 1):
        raise Exception(f"Out of Bonds [{lb}, {ub}]. Confine lb and ub within a range of [-1.0 and 1.0]")

    return _calc_mrp_with_short_sale(sigma, mu, lb, ub, target_mu)


def create_efficient_frontier_portfolios(sigma, mu, lb: float, ub: float, number_pfs: int = 100) -> list[tuple]:
    """This function creates the mu-sigma portfolio profiles of the Efficient Frontier"""
    pfs = np.arange(0, number_pfs)
    mvp = portfolio_mean_variance(sigma, mu, minimum_variance_portfolio(sigma, lb, ub))
    mrp = portfolio_mean_variance(sigma, mu, maximum_return_portfolio(sigma, mu, lb, ub))

    targets_mu = mvp[0] + pfs * (mrp[0] - mvp[0]) / number_pfs

    return [portfolio_mean_variance(sigma, mu,
                                    efficient_frontier_portfolio(sigma, mu, lb, ub, targets_mu[i])) for i in pfs]


def portfolio_mean_variance(sigma, mu, weights) -> tuple[float, float]:
    portfolio_mu = mu @ weights
    portfolio_std = np.sqrt(weights @ sigma @ weights)
    return (portfolio_mu, portfolio_std)


def _calc_mrp_with_short_sale(sigma, mu, lb: float, ub: float, target_mu: float = None):
    """This function calculates the weights of the Maximum Return Portfolio with short sale"""
    if target_mu is None:
        target_mu = max(mu)

    n = sigma.shape[1]

    P = matrix(np.array(sigma), tc="d")
    q = matrix(-np.array(mu), tc="d")

    lb_diag = np.diag(np.repeat(-1, n))
    ub_diag = np.diag(np.repeat(1, n))

    G = matrix(np.concatenate((lb_diag, ub_diag)), tc="d")
    h = matrix(np.concatenate((np.repeat(-lb, n), np.repeat(ub, n))), tc="d")

    A = matrix([list(np.repeat(1.0, n)), list(np.array(mu))]).T
    b = matrix([1.0, target_mu], tc="d")

    sol = solvers.qp(P, q, G=G, h=h, A=A, b=b)
    return np.array(sol["x"]).flatten()


def _calc_mrp_without_short_sale(sigma, mu, lb: float, ub: float):
    """This function calculates the weights of the Maximum Return Portfolio without short sale"""
    n = sigma.shape[1]

    P = matrix(np.array(sigma), tc="d")
    q = matrix(-np.array(mu), tc="d")

    lb_diag = np.diag(np.repeat(-1, n))
    ub_diag = np.diag(np.repeat(1, n))

    G = matrix(np.concatenate((lb_diag, ub_diag)), tc="d")
    h = matrix(np.concatenate((np.repeat(-lb, n), np.repeat(ub, n))), tc="d")

    A = matrix(np.repeat(1, n), tc="d").T
    b = matrix(1.0, tc="d")

    sol = solvers.qp(P, q, G, h, A, b)
    return np.array(sol["x"]).flatten()


def main() -> None:
    df = pd.read_excel("data/portfolio_ten_stocks.xlsx", index_col="DATE")
    df_log_returns = log_returns(df).fillna(0)

    # Moments
    mu_monthly = df_log_returns.mean()
    std_monthly = df_log_returns.std()
    cov_matrix = df_log_returns.cov()

    # Annually Scaled
    mu = mu_monthly * 12
    std = std_monthly * np.sqrt(12)
    sigma = cov_matrix * 12

    # MVP
    weights_mvp = minimum_variance_portfolio(sigma, -1.0, 1.0)
    weights_mvp_long = minimum_variance_portfolio(sigma, 0.0, 1.0)
    weights_mvp_lb5_ub35 = minimum_variance_portfolio(sigma, 0.05, 0.35)

    # MRP
    weights_mrp = maximum_return_portfolio(sigma, mu, -1.0, 1.0)
    weights_mrp_long = maximum_return_portfolio(sigma, mu, 0, 1)
    weights_mrp_lb5_ub35 = maximum_return_portfolio(sigma, mu, 0.05, 0.35)

    summary_mu_std = {
        "MVP": portfolio_mean_variance(sigma, mu, weights_mvp),
        "MVP Long": portfolio_mean_variance(sigma, mu, weights_mvp_long),
        "MVP 5 35": portfolio_mean_variance(sigma, mu, weights_mvp_lb5_ub35),
        "MRP": portfolio_mean_variance(sigma, mu, weights_mrp),
        "MRP Long": portfolio_mean_variance(sigma, mu, weights_mrp_long),
        "MRP 5 35": portfolio_mean_variance(sigma, mu, weights_mrp_lb5_ub35)
    }

    for key, val in summary_mu_std.items():
        print(f"{key:10}: mu={val[0]*100:.2f}%, std={val[1]*100:.2f}%")

    # Frontier Pfs
    frontier = create_efficient_frontier_portfolios(sigma, mu, -1.0, 1.0)
    frontier_long = create_efficient_frontier_portfolios(sigma, mu, 0, 1.0)
    frontier_lb5_ub35 = create_efficient_frontier_portfolios(sigma, mu, 0.05, 0.35)

    # Tangential Portfolio (would require some equivalent function to fmincon i guess or use analytical formula)

    df_concatenated = pd.concat([
        pd.DataFrame.from_records(zip(mu, std), columns=["mu", "std"]).assign(dataset="assets"),
        pd.DataFrame.from_records(frontier, columns=["mu", "std"]).assign(dataset="frontier"),
        pd.DataFrame.from_records(frontier_long, columns=["mu", "std"]).assign(dataset="frontier_long"),
        pd.DataFrame.from_records(frontier_lb5_ub35, columns=["mu", "std"]).assign(dataset="frontier_5_35")])

    f, ax = plt.subplots(figsize=(12, 8))
    ax.set_title("Mu-Std Scatter Plot")
    sns.scatterplot(data=df_concatenated, x="std", y="mu", legend="auto", hue="dataset", style="dataset", ax=ax)
    f.get_figure().savefig("figures/mu-std-frontier.png")


if __name__ == "__main__":
    main()
