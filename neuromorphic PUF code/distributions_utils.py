import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import norm, kstest
from scipy import stats

def fit_gmm(data, n_components):
    """
    Fit a Gaussian Mixture Model (GMM) with a specified number of components.

    Parameters:
        data (array-like): Input data to model.
        n_components (int): Number of Gaussian components to fit.

    Returns:
        GaussianMixture: Fitted GMM object.
    """
    gmm = GaussianMixture(n_components=n_components, random_state=0)
    gmm.fit(data.reshape(-1, 1))  # Ensure data is 2D
    return gmm

def gmm_pdf_cdf(x, gmm):
    """
    Compute the PDF and CDF of a Gaussian Mixture Model (GMM) at specified values.

    Parameters:
        x (np.ndarray): Input values (1D array).
        gmm (GaussianMixture): A fitted GMM.

    Returns:
        tuple: (pdf, cdf), both are NumPy arrays of the same shape as x.
    """
    weights = gmm.weights_  # Mixing proportions
    means = gmm.means_.flatten()  # Gaussian means
    covariances = gmm.covariances_.flatten()  # Variances of each component

    pdf = np.zeros_like(x, dtype=np.float64)
    cdf = np.zeros_like(x, dtype=np.float64)

    # Aggregate PDF and CDF from all components
    for weight, mean, covariance in zip(weights, means, covariances):
        std_dev = np.sqrt(covariance)
        pdf += weight * norm.pdf(x, loc=mean, scale=std_dev)
        cdf += weight * norm.cdf(x, loc=mean, scale=std_dev)

    return pdf, cdf

def ks_test_gmm(data, gmm):
    """
    Perform a one-sample Kolmogorov-Smirnov (KS) test to compare the empirical
    CDF of the data to the GMM's CDF.

    Parameters:
        data (array-like): The raw data samples.
        gmm (GaussianMixture): Fitted GMM to test against.

    Returns:
        float: KS statistic (maximum absolute difference between CDFs).
    """
    sorted_data = np.sort(data)  # Required for proper empirical CDF
    empirical_cdf = np.arange(1, len(data) + 1) / len(data)  # Empirical CDF
    _, gmm_cdf_vals = gmm_pdf_cdf(sorted_data, gmm)  # Model CDF

    ks_stat = np.max(np.abs(empirical_cdf - gmm_cdf_vals))  # KS distance
    return ks_stat

def best_fit_distribution(data, max_gmm_components=5):
    """
    Try multiple GMMs with different numbers of components and select the best one
    using the Kolmogorov-Smirnov test statistic.

    Parameters:
        data (array-like): 1D data array to fit.
        max_gmm_components (int): Max number of GMM components to try.

    Returns:
        dict: Dictionary containing:
            - 'name': name of the best model
            - 'distribution': the best GMM model object
            - 'parameters': dict of model parameters (weights, means, covariances)
            - 'ks_statistic': KS test statistic for the best model
    """
    best_distribution = None
    best_params = None
    best_statistic = np.inf  # Start with a large value
    best_distribution_name = None

    # Try GMMs with 1 to max_gmm_components
    for n_components in range(1, max_gmm_components + 1):
        try:
            gmm = GaussianMixture(n_components=n_components, random_state=0)
            gmm.fit(data.reshape(-1, 1))  # 2D format expected by GMM
            ks_stat = ks_test_gmm(data, gmm)  # Evaluate goodness of fit

            # If this model fits better (smaller KS statistic), save it
            if ks_stat < best_statistic:
                best_distribution = gmm
                best_params = {
                    'weights': gmm.weights_,
                    'means': gmm.means_,
                    'covariances': gmm.covariances_
                }
                best_statistic = ks_stat
                best_distribution_name = f'Gaussian Mixture ({n_components} components)'

        except Exception as e:
            print(f"Could not fit GMM with {n_components} components: {e}")

    return {
        'name': best_distribution_name,
        'distribution': best_distribution,
        'parameters': best_params,
        'ks_statistic': best_statistic
    }
