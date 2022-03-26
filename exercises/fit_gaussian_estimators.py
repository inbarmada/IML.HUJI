from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    sample = np.random.normal(10, 1, 1000)
    uv = UnivariateGaussian()
    uv.fit(sample)
    print("(", uv.mu_, ", ", uv.var_, ")")

    # Question 2 - Empirically showing sample mean is consistent
    diff_from_mean_array = []
    sample_size = range(10, 1000, 10)
    for i in sample_size:
        uv.fit(sample[:i])
        diff_from_mean_array.append(abs(uv.mu_-10))

    # Draw plot
    plot = go.Figure(data=[go.Scatter(
                     x=list(sample_size),
                     y=diff_from_mean_array,
                     mode='lines',
                     marker_color='rgba(199, 10, 165, .9)')
    ])

    plot.update_layout(
        title="Calculated Mean vs True Mean as a Function of Sample Size",
        xaxis_title="Sample Size",
        yaxis_title="Distance from True Mean",
    )
    plot.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    sample = np.sort(sample)
    pdf = uv.pdf(sample)
    # Draw plot
    pdf_plot = go.Figure()
    pdf_plot.add_trace(go.Scatter(
        x=sample,
        y=pdf,
        mode='lines',
        marker_color='rgba(19, 200, 195, .9)')
    )

    pdf_plot.update_layout(
        title="PDF from Estimators",
        xaxis_title="Sample",
        yaxis_title="Probability",
    )
    pdf_plot.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mean = np.array([0, 0, 4, 0])
    cov = np.array([[1, 0.2, 0, 0.5],
           [0.2, 2, 0, 0],
           [0, 0, 1, 0],
           [0.5, 0, 0, 1]])
    sample = np.random.multivariate_normal(mean, cov, 1000)
    mv = MultivariateGaussian()
    mv.fit(sample)
    print(mv.mu_, "\n", mv.cov_)

    # Question 5 - Likelihood evaluation
    f1 = np.linspace(-10, 10, 200)
    f3 = np.linspace(-10, 10, 200)
    log_likelihood_vals = [[mv.log_likelihood(np.array([a, 0, c, 0]), cov, sample) for c in f3] for a in f1]
    heatmap_plot = go.Figure(data=go.Heatmap(
                   z=log_likelihood_vals,
                   x=f1,
                   y=f3))
    heatmap_plot.update_layout(
        title="Log Likelihood of mu = [f1, 0, f3, 0]",
        xaxis_title="f1",
        yaxis_title="f3"
        )

    heatmap_plot.show()


    # Question 6 - Maximum likelihood
    max_val = np.amax(log_likelihood_vals)
    f1_loc, f3_loc = np.unravel_index(np.argmax(log_likelihood_vals),
                                      np.shape(log_likelihood_vals))
    print("max value = ", max_val,
          "\nmu = [", round(f1[f1_loc], 3),
          ", 0, ", round(f3[f3_loc], 3), ", 0]")


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
