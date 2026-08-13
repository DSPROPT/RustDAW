#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

//! LOWESS: locally weighted scatterplot smoothing.
//!
//! This is what stops the matching EQ from being a comb. The ratio between two
//! spectra is spiky — every place the target happens to have a null and the
//! reference does not asks for enormous gain — and following it literally
//! produces a filter that rings. Fitting a line through the neighbourhood of
//! each point instead keeps the broad tonal difference, which is the part that
//! actually makes one mix sound like another, and discards the rest.
//!
//! A local *linear* fit rather than a local average matters at the ends, where
//! an average has neighbours on one side only and pulls the curve flat.
//!
//! See Cleveland, "Robust Locally Weighted Regression and Smoothing
//! Scatterplots" (1979). Ported from the `statsmodels` implementation that
//! Matchering calls.

/// Smooths `y` sampled at `x`.
///
/// `frac` is the share of the data in each local neighbourhood. `iterations`
/// is Cleveland's robustifying pass count: each one re-weights by how badly
/// the previous fit missed, which pulls the curve away from outliers. `delta`
/// skips points closer together than that distance and fills them in by
/// straight line — on the log grid, where thousands of points sit inside one
/// octave, this is most of the running time.
#[must_use]
pub fn smooth(x: &[f64], y: &[f64], frac: f64, iterations: usize, delta: f64) -> Vec<f64> {
    let n = x.len();
    if n < 3 || y.len() != n {
        return y.to_vec();
    }

    // At least two neighbours, or a line cannot be fitted to them.
    let window = ((frac * n as f64).ceil() as usize).clamp(2, n);
    let mut fitted = vec![0.0; n];
    let mut residual_weights = vec![1.0; n];

    for iteration in 0..=iterations {
        let mut left = 0_usize;
        let mut last_fitted: Option<usize> = None;

        let mut index = 0_usize;
        while index < n {
            // Slide the neighbourhood so it holds the `window` points nearest
            // to x[index].
            while left + window < n {
                let leaving = x[index] - x[left];
                let entering = x[left + window] - x[index];
                if entering >= leaving {
                    break;
                }
                left += 1;
            }
            let right = (left + window).min(n);

            fitted[index] = fit_local(x, y, &residual_weights, index, left, right);

            // Everything skipped since the last fit is filled by interpolating
            // between the two fitted values.
            if let Some(previous) = last_fitted {
                let span = x[index] - x[previous];
                if span > 0.0 {
                    for between in previous + 1..index {
                        let along = (x[between] - x[previous]) / span;
                        fitted[between] = fitted[previous] * (1.0 - along) + fitted[index] * along;
                    }
                }
            }
            last_fitted = Some(index);

            // Advance past every point within `delta` of this one.
            let mut next = index + 1;
            while next < n && x[next] - x[index] <= delta {
                next += 1;
            }
            index = next.max(index + 1);
        }

        // The final point is always fitted, so a skipped tail cannot be left
        // holding zeros.
        if last_fitted != Some(n - 1) {
            let previous = last_fitted.unwrap_or(0);
            fitted[n - 1] = fit_local(x, y, &residual_weights, n - 1, n - window, n);
            let span = x[n - 1] - x[previous];
            if span > 0.0 {
                for between in previous + 1..n - 1 {
                    let along = (x[between] - x[previous]) / span;
                    fitted[between] = fitted[previous] * (1.0 - along) + fitted[n - 1] * along;
                }
            }
        }

        if iteration == iterations {
            break;
        }
        update_robustness(y, &fitted, &mut residual_weights);
    }

    fitted
}

/// Weighted linear regression over one neighbourhood, evaluated at `at`.
fn fit_local(
    x: &[f64],
    y: &[f64],
    robustness: &[f64],
    at: usize,
    left: usize,
    right: usize,
) -> f64 {
    // Tricube weights, scaled by the distance to the furthest neighbour.
    let furthest = (x[at] - x[left]).abs().max((x[right - 1] - x[at]).abs());
    let mut weights = Vec::with_capacity(right - left);
    for index in left..right {
        let weight = if furthest > 0.0 {
            let distance = (x[index] - x[at]).abs() / furthest;
            if distance >= 1.0 {
                0.0
            } else {
                let cube = 1.0 - distance * distance * distance;
                cube * cube * cube
            }
        } else {
            1.0
        };
        weights.push(weight * robustness[index]);
    }

    let total: f64 = weights.iter().sum();
    if total <= 0.0 {
        return y[at];
    }

    let mean_x: f64 = weights
        .iter()
        .zip(x[left..right].iter())
        .map(|(w, value)| w * value)
        .sum::<f64>()
        / total;
    let mean_y: f64 = weights
        .iter()
        .zip(y[left..right].iter())
        .map(|(w, value)| w * value)
        .sum::<f64>()
        / total;

    let mut covariance = 0.0;
    let mut variance = 0.0;
    for (offset, weight) in weights.iter().enumerate() {
        let dx = x[left + offset] - mean_x;
        covariance += weight * dx * (y[left + offset] - mean_y);
        variance += weight * dx * dx;
    }

    // With no spread in x the neighbourhood is a single stack of points and
    // the weighted mean is the whole answer.
    if variance <= f64::EPSILON * total {
        return mean_y;
    }
    mean_y + (covariance / variance) * (x[at] - mean_x)
}

/// Bisquare weights from the residuals, for the next robustifying pass.
fn update_robustness(y: &[f64], fitted: &[f64], weights: &mut [f64]) {
    let mut residuals: Vec<f64> = y
        .iter()
        .zip(fitted.iter())
        .map(|(observed, fit)| (observed - fit).abs())
        .collect();

    let mut sorted = residuals.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = sorted[sorted.len() / 2];
    // Six times the median absolute residual is Cleveland's cutoff: past it a
    // point is an outlier and stops influencing the fit entirely.
    let cutoff = 6.0 * median;
    // A fit that already passes through almost every point leaves no spread to
    // judge outliers by, and scaling by it would reject the whole curve.
    if !cutoff.is_finite() || cutoff <= f64::EPSILON {
        weights.fill(1.0);
        return;
    }

    for (weight, residual) in weights.iter_mut().zip(residuals.drain(..)) {
        let scaled = (residual / cutoff).min(1.0);
        let square = 1.0 - scaled * scaled;
        *weight = square * square;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid(n: usize) -> Vec<f64> {
        (0..n).map(|i| i as f64 / (n - 1) as f64).collect()
    }

    #[test]
    fn a_straight_line_survives_smoothing() {
        let x = grid(200);
        let y: Vec<f64> = x.iter().map(|t| 2.0 * t + 1.0).collect();
        let smoothed = smooth(&x, &y, 0.3, 0, 0.0);
        for (index, (fit, expected)) in smoothed.iter().zip(y.iter()).enumerate() {
            assert!(
                (fit - expected).abs() < 1e-6,
                "point {index}: got {fit}, expected {expected}"
            );
        }
    }

    #[test]
    fn noise_is_reduced_but_the_shape_is_kept() {
        let x = grid(400);
        // A smooth curve plus a deterministic zigzag.
        let clean: Vec<f64> = x.iter().map(|t| (t * 6.0).sin()).collect();
        let noisy: Vec<f64> = clean
            .iter()
            .enumerate()
            .map(|(i, value)| value + if i % 2 == 0 { 0.25 } else { -0.25 })
            .collect();
        let smoothed = smooth(&x, &noisy, 0.15, 0, 0.0);

        let noisy_error: f64 = noisy
            .iter()
            .zip(clean.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        let smoothed_error: f64 = smoothed
            .iter()
            .zip(clean.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            smoothed_error < noisy_error * 0.2,
            "smoothing should remove most of the zigzag: {smoothed_error} vs {noisy_error}"
        );
    }

    #[test]
    fn a_narrow_spike_is_flattened() {
        // The reason this exists: one absurd bin in the ratio must not become
        // an absurd filter gain.
        let x = grid(300);
        let mut y = vec![1.0; 300];
        y[150] = 40.0;
        let smoothed = smooth(&x, &y, 0.1, 0, 0.0);
        assert!(
            smoothed[150] < 6.0,
            "the spike should be pulled down, got {}",
            smoothed[150]
        );
    }

    #[test]
    fn robustifying_iterations_reject_outliers_further() {
        // Ordinary scatter plus one wild point. The scatter matters: the
        // robustness cutoff is a multiple of the median residual, so a fit
        // that already passes exactly through every point has no scale to
        // judge an outlier against.
        let x = grid(200);
        let mut y: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, t)| t * 0.5 + if i % 3 == 0 { 0.02 } else { -0.015 })
            .collect();
        y[100] = 9.0;
        let plain = smooth(&x, &y, 0.2, 0, 0.0);
        let robust = smooth(&x, &y, 0.2, 3, 0.0);
        let truth = x[100] * 0.5;
        assert!(
            (robust[100] - truth).abs() < (plain[100] - truth).abs(),
            "robust {} should beat plain {} against {truth}",
            robust[100],
            plain[100]
        );
    }

    #[test]
    fn a_fit_with_no_residual_scale_is_left_alone() {
        // The degenerate case the guard covers: perfectly linear data, where
        // the median residual is zero and every point would otherwise be
        // scored as infinitely far out.
        let x = grid(100);
        let y: Vec<f64> = x.iter().map(|t| t * 0.5).collect();
        let robust = smooth(&x, &y, 0.2, 3, 0.0);
        for (index, (fit, expected)) in robust.iter().zip(y.iter()).enumerate() {
            assert!(
                (fit - expected).abs() < 1e-6,
                "point {index}: got {fit}, expected {expected}"
            );
        }
    }

    #[test]
    fn delta_skipping_tracks_the_full_fit() {
        let x = grid(500);
        let y: Vec<f64> = x.iter().map(|t| (t * 4.0).sin() + t).collect();
        let full = smooth(&x, &y, 0.2, 0, 0.0);
        let skipped = smooth(&x, &y, 0.2, 0, 0.004);
        for (index, (a, b)) in full.iter().zip(skipped.iter()).enumerate() {
            assert!(
                (a - b).abs() < 0.02,
                "point {index}: full {a} vs skipped {b}"
            );
        }
    }

    #[test]
    fn degenerate_input_is_returned_unchanged() {
        assert_eq!(
            smooth(&[0.0, 1.0], &[1.0, 2.0], 0.5, 0, 0.0),
            vec![1.0, 2.0]
        );
    }
}
