"""Auxiliary functions:

ComputeError    Pearson rho, RMSE, MAE, CAE
Iterable        Is an object iterable?
IsIterable      Is an object iterable and not a string?
SurrogateData   ebisuzaki, random shuffle, seasonal
PlotObsPred     Plot observations & predictions
PlotCoef        Plot s-map coefficients
PlotCCM
PlotEmbedDimension    Plot E,rho from EmbedDimension
PlotPredictNonlinear  Plot theta,rho from PredictNonlinear
"""

# python modules
from math import floor, pi, cos
from cmath import exp
from random import sample, uniform, normalvariate

# package modules
from numpy import absolute, any, arange, corrcoef, fft, isfinite
from numpy import mean, max, nan, ptp, std, sqrt, zeros
from pandas import DataFrame
from scipy.interpolate import UnivariateSpline
from matplotlib.pyplot import show, axhline


def ComputeError(obs, pred, digits=6):
    """Pearson rho, MAE, CAE, RMSE
    Remove nan from obs, pred for corrcoeff.
    """

    notNan = isfinite(pred)
    if any(~notNan):
        pred = pred[notNan]
        obs = obs[notNan]

    notNan = isfinite(obs)
    if any(~notNan):
        pred = pred[notNan]
        obs = obs[notNan]

    if len(pred) < 5:
        msg = (
            f"ComputeError(): Not enough data ({len(pred)}) to "
            + " compute error statistics."
        )
        print(msg)
        return {"rho": nan, "MAE": nan, "RMSE": nan}

    rho = round(corrcoef(obs, pred)[0, 1], digits)
    err = obs - pred
    MAE = round(max(err), digits)
    CAE = round(absolute(err).sum(), digits)
    RMSE = round(sqrt(mean(err**2)), digits)

    D = {"rho": rho, "MAE": MAE, "CAE": CAE, "RMSE": RMSE}

    return D


def Iterable(obj):
    """Is an object iterable?"""

    try:
        it = iter(obj)
    except TypeError:
        return False
    return True


def IsIterable(obj):
    """Is an object iterable and not a string?"""

    if Iterable(obj):
        if isinstance(obj, str):
            return False
        else:
            return True
    return False


def SurrogateData(
    dataFrame=None,
    column=None,
    method="ebisuzaki",
    numSurrogates=10,
    alpha=None,
    smooth=0.8,
    outputFile=None,
):
    """Generate surrogate data

    Three methods:

    random_shuffle :
      Sample the data with a uniform distribution.

    ebisuzaki :
      Journal of Climate. A Method to Estimate the Statistical Significance
      of a Correlation When the Data Are Serially Correlated.
      https://doi.org/10.1175/1520-0442(1997)010<2147:AMTETS>2.0.CO;2

      Presumes data are serially correlated with low pass coherence. It is:
      "resampling in the frequency domain. This procedure will not preserve
      the distribution of values but rather the power spectrum (periodogram).
      The advantage of preserving the power spectrum is that resampled series
      retains the same autocorrelation as the original series."

    seasonal :
      Presume a smoothing spline represents the seasonal trend.
      Each surrogate is a summation of the trend, resampled residuals,
      and possibly additive Gaussian noise. Default noise has a standard
      deviation that is the data range / 5.
    """

    if dataFrame is None:
        raise RuntimeError("SurrogateData() empty DataFrame.")

    if column is None:
        raise RuntimeError("SurrogateData() must specify column.")

    # New dataFrame with initial time column
    df = DataFrame(dataFrame.iloc[:, 0])

    if method.lower() == "random_shuffle":
        for s in range(numSurrogates):  # use pandas sample
            surr = dataFrame[column].sample(n=dataFrame.shape[0]).to_numpy()
            df[s + 1] = surr

    elif method.lower() == "ebisuzaki":
        data = dataFrame[column].to_numpy()
        n = dataFrame.shape[0]
        n2 = floor(n / 2)
        sigma = std(data)
        a = fft.fft(data)
        amplitudes = absolute(a)
        amplitudes[0] = 0

        for s in range(numSurrogates):
            thetas = [2 * pi * uniform(0, 1) for x in range(n2 - 1)]
            revThetas = thetas[::-1]
            negThetas = [-x for x in revThetas]
            angles = [0] + thetas + [0] + negThetas
            surrogate_z = [
                A * exp(complex(0, theta)) for A, theta in zip(amplitudes, angles)
            ]

            if n % 2 == 0:  # even length
                surrogate_z[-1] = complex(
                    sqrt(2) * amplitudes[-1] * cos(2 * pi * uniform(0, 1))
                )

            ifft = fft.ifft(surrogate_z) / n

            realifft = [x.real for x in ifft]
            sdevifft = std(realifft)

            # adjust variance of surrogate time series to match original
            scaled = [sigma * x / sdevifft for x in realifft]

            df[s + 1] = scaled

    elif method.lower() == "seasonal":
        y = dataFrame[column].to_numpy()
        n = dataFrame.shape[0]

        # Presume a spline captures the seasonal cycle
        x = arange(n)
        spline = UnivariateSpline(x, y)
        spline.set_smoothing_factor(smooth)
        y_spline = spline(x)

        # Residuals of the smoothing
        residual = list(y - y_spline)

        # spline plus shuffled residuals plus Gaussian noise
        noise = zeros(n)

        # If no noise specified, set std dev to data range / 5
        if alpha is None:
            alpha = ptp(y) / 5

        for s in range(numSurrogates):
            noise = [normalvariate(0, alpha) for z in range(n)]

            df[s + 1] = y_spline + sample(residual, n) + noise

    else:
        raise RuntimeError("SurrogateData() invalid method.")

    df = df.round(8)  # Should be a parameter

    # Rename columns
    columnNames = [column + "_" + str(c + 1) for c in range(numSurrogates)]

    columnNames.insert(0, df.columns[0])  # insert time column name

    df.columns = columnNames

    if outputFile:
        df.to_csv(outputFile, index=False)

    return df


def PlotObsPred(df, title=None, ax=None):
    """Plot observations and predictions with default ρ, RMSE

       If block=True show the plot and block for user to close
       Return pyplot axis of plot"""
    if title is None:
        stats = ComputeError(df["Observations"], df["Predictions"])
        title = f"ρ={round(stats['rho'], 3)}  RMSE={round(stats['RMSE'], 3)}"

    time_col = df.columns[0]
    ax_ = df.plot(time_col, ["Observations", "Predictions"], title=title,
                  ax=ax, linewidth=3)
    
    if ax is None:
        show()

    return ax_


def PlotCoeff(df, title=None, ax=None):
    """Plot S-Map coefficients

       If ax is None call plt.show to display
       Return pyplot axis of plot"""
    time_col = df.columns[0]
    # Coefficient columns can be in any column
    coef_cols = [x for x in df.columns if time_col not in x]
    ax_ = df.plot(time_col, coef_cols, title=title, linewidth=3, ax=ax, subplots=True)

    if ax is None:
        show()

    return ax_


def PlotCCM(df, title=None, ax=None):
    """Plot CCM

       If ax is None call plt.show to display
       Return pyplot axis of plot"""
    if df.shape[1] == 3 :
        # CCM of two different variables
        ax_ = df.plot(
            'LibSize', [df.columns[1], df.columns[2]], title=title, linewidth=3, ax=ax
        )
    elif df.shape[1] == 2 :
        # CCM of degenerate columns : target
        ax_ = df.plot('LibSize', df.columns[1], title=title, linewidth=3, ax=ax)

    ax_.set( xlabel = "Library Size", ylabel = "CCM ρ" )
    axhline(y=0, linewidth=1)

    if ax is None:
        show()

    return ax_


def PlotEmbedDimension(df, title=None, ax=None):
    """Plot embedding dimension

       If ax is None call plt.show to display
       Return pyplot axis of plot"""
    ax_ = df.plot( 'E', 'rho', title=title, linewidth=3, ax=ax )
    ax_.set(xlabel = "Embedding Dimension", ylabel = "Prediction Skill ρ")

    if ax is None:
        show()

    return ax_


def PlotPredictNonlinear(df, title=None, ax=None):
    """Plot S-map Localisation (θ)

       If ax is None call plt.show to display
       Return pyplot axis of plot"""
    ax_ = df.plot( 'theta', 'rho', title=title, linewidth=3, ax=ax )
    ax_.set(xlabel = "S-map Localisation (θ)", ylabel = "Prediction Skill ρ")
    
    if ax is None:
        show()

    return ax_
