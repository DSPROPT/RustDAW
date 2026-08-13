#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

//! Cubic spline interpolation, used to move the matching curve between a
//! linear frequency grid and a logarithmic one.
//!
//! The end condition is *not-a-knot* — the third derivative is continuous
//! across the second and second-to-last points — because that is what `SciPy`'s
//! `interp1d(kind="cubic")` builds, and the curve this smooths is the one that
//! becomes the EQ. A natural spline, which forces the curvature to zero at both
//! ends, would flatten the response right where the bass lives.
//!
//! Solved for the first derivative at each knot rather than the second. The
//! two formulations describe the same spline, but the second-derivative system
//! has a not-a-knot row that collapses to a zero pivot on an evenly spaced
//! grid — and half the grids here are evenly spaced.
//!
//! Working in `f64`: the log grid crowds thousands of points into the bottom
//! octave, and the spacing between neighbours there falls below what `f32`
//! resolves.

/// A cubic spline through a set of strictly increasing `x` values.
pub struct Spline {
    x: Vec<f64>,
    /// Per-interval coefficients, highest power first, relative to `x[i]`.
    segments: Vec<[f64; 4]>,
}

impl Spline {
    /// Fits a spline through `(x, y)`.
    ///
    /// Returns `None` if there are fewer than four points, or if `x` is not
    /// strictly increasing — both of which mean the caller built the grid
    /// wrong rather than that the data is unusual.
    #[must_use]
    pub fn new(x: &[f64], y: &[f64]) -> Option<Self> {
        let n = x.len();
        if n < 4 || y.len() != n || x.windows(2).any(|pair| pair[1] <= pair[0]) {
            return None;
        }

        let h: Vec<f64> = x.windows(2).map(|pair| pair[1] - pair[0]).collect();
        let slope: Vec<f64> = (0..n - 1).map(|i| (y[i + 1] - y[i]) / h[i]).collect();

        let mut lower = vec![0.0; n];
        let mut diagonal = vec![0.0; n];
        let mut upper = vec![0.0; n];
        let mut rhs = vec![0.0; n];

        for i in 1..n - 1 {
            lower[i] = h[i];
            diagonal[i] = 2.0 * (h[i - 1] + h[i]);
            upper[i] = h[i - 1];
            rhs[i] = 3.0 * (h[i] * slope[i - 1] + h[i - 1] * slope[i]);
        }

        // Not-a-knot at the left. Weighting the first two slopes this way is
        // what makes the third derivative match across the second knot.
        let span = h[0] + h[1];
        diagonal[0] = h[1];
        upper[0] = span;
        rhs[0] = ((h[0] + 2.0 * span) * h[1] * slope[0] + h[0] * h[0] * slope[1]) / span;

        // And its mirror at the right.
        let span = h[n - 3] + h[n - 2];
        diagonal[n - 1] = h[n - 3];
        lower[n - 1] = span;
        rhs[n - 1] = (h[n - 2] * h[n - 2] * slope[n - 3]
            + (2.0 * span + h[n - 2]) * h[n - 3] * slope[n - 2])
            / span;

        let derivative = solve_tridiagonal(&lower, &diagonal, &upper, &rhs)?;

        let segments = (0..n - 1)
            .map(|i| {
                let (d0, d1) = (derivative[i], derivative[i + 1]);
                [
                    (d0 + d1 - 2.0 * slope[i]) / (h[i] * h[i]),
                    (3.0 * slope[i] - 2.0 * d0 - d1) / h[i],
                    d0,
                    y[i],
                ]
            })
            .collect();

        Some(Self {
            x: x.to_vec(),
            segments,
        })
    }

    /// Evaluates the spline at `at`, extrapolating with the end cubics when
    /// `at` falls outside the fitted range.
    #[must_use]
    pub fn evaluate(&self, at: f64) -> f64 {
        let last = self.segments.len() - 1;
        let index = match self
            .x
            .binary_search_by(|probe| probe.partial_cmp(&at).unwrap_or(std::cmp::Ordering::Less))
        {
            Ok(exact) => exact.min(last),
            Err(0) => 0,
            Err(above) => (above - 1).min(last),
        };

        let [cubic, quadratic, linear, constant] = self.segments[index];
        let offset = at - self.x[index];
        constant + offset * (linear + offset * (quadratic + offset * cubic))
    }

    /// Evaluates at every point of `grid`.
    #[must_use]
    pub fn evaluate_all(&self, grid: &[f64]) -> Vec<f64> {
        grid.iter().map(|at| self.evaluate(*at)).collect()
    }
}

/// Thomas algorithm. Returns `None` if the system is singular, which for a
/// spline means the grid had coincident points that slipped through.
fn solve_tridiagonal(
    lower: &[f64],
    diagonal: &[f64],
    upper: &[f64],
    rhs: &[f64],
) -> Option<Vec<f64>> {
    let n = diagonal.len();
    let mut c = vec![0.0; n];
    let mut d = vec![0.0; n];

    if diagonal[0].abs() < f64::EPSILON {
        return None;
    }
    c[0] = upper[0] / diagonal[0];
    d[0] = rhs[0] / diagonal[0];

    for i in 1..n {
        let pivot = diagonal[i] - lower[i] * c[i - 1];
        if pivot.abs() < f64::EPSILON {
            return None;
        }
        c[i] = upper[i] / pivot;
        d[i] = (rhs[i] - lower[i] * d[i - 1]) / pivot;
    }

    let mut solution = vec![0.0; n];
    solution[n - 1] = d[n - 1];
    for i in (0..n - 1).rev() {
        solution[i] = d[i] - c[i] * solution[i + 1];
    }
    Some(solution)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_spline_passes_through_its_knots() {
        let x = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [0.0, 0.8, 0.9, 0.1, -0.8, -1.0];
        let spline = Spline::new(&x, &y).expect("fits");
        for (at, expected) in x.iter().zip(y.iter()) {
            assert!(
                (spline.evaluate(*at) - expected).abs() < 1e-9,
                "at {at}: got {}, expected {expected}",
                spline.evaluate(*at)
            );
        }
    }

    #[test]
    fn a_cubic_is_reproduced_exactly() {
        // Not-a-knot's defining property: with no interior knot condition to
        // violate, a cubic through the points is recovered everywhere, which a
        // natural spline cannot do.
        let f = |t: f64| 2.0 * t * t * t - 3.0 * t * t + t - 5.0;
        let x: Vec<f64> = (0..8).map(f64::from).collect();
        let y: Vec<f64> = x.iter().map(|t| f(*t)).collect();
        let spline = Spline::new(&x, &y).expect("fits");
        for step in 0..70 {
            let at = f64::from(step) * 0.1;
            assert!(
                (spline.evaluate(at) - f(at)).abs() < 1e-6,
                "at {at}: got {}, expected {}",
                spline.evaluate(at),
                f(at)
            );
        }
    }

    #[test]
    fn a_straight_line_stays_straight() {
        let x: Vec<f64> = (0..10).map(f64::from).collect();
        let y: Vec<f64> = x.iter().map(|t| 3.0 * t + 1.0).collect();
        let spline = Spline::new(&x, &y).expect("fits");
        for step in 0..90 {
            let at = f64::from(step) * 0.1;
            assert!((spline.evaluate(at) - (3.0 * at + 1.0)).abs() < 1e-9);
        }
    }

    #[test]
    fn uneven_spacing_is_handled() {
        // The log grid is extremely uneven; this is the shape that matters.
        let x = [0.0, 0.01, 0.05, 0.4, 1.0, 4.0, 20.0];
        let y = [1.0, 1.2, 0.9, 1.4, 0.7, 1.1, 1.0];
        let spline = Spline::new(&x, &y).expect("fits");
        for (at, expected) in x.iter().zip(y.iter()) {
            assert!((spline.evaluate(*at) - expected).abs() < 1e-9);
        }
    }

    #[test]
    fn evenly_spaced_grids_do_not_break_the_solve() {
        // The second-derivative formulation collapses here; this is the
        // regression that put the spline on first derivatives instead.
        let x: Vec<f64> = (0..64).map(f64::from).collect();
        let y: Vec<f64> = x.iter().map(|t| (t * 0.3).sin()).collect();
        let spline = Spline::new(&x, &y).expect("an even grid must still fit");
        for (at, expected) in x.iter().zip(y.iter()) {
            assert!((spline.evaluate(*at) - expected).abs() < 1e-9);
        }
    }

    #[test]
    fn extrapolation_continues_the_end_cubic() {
        let x: Vec<f64> = (0..6).map(f64::from).collect();
        let y: Vec<f64> = x.iter().map(|t| 2.0 * t + 3.0).collect();
        let spline = Spline::new(&x, &y).expect("fits");
        assert!((spline.evaluate(-1.0) - 1.0).abs() < 1e-9);
        assert!((spline.evaluate(7.0) - 17.0).abs() < 1e-9);
    }

    #[test]
    fn too_few_points_or_a_disordered_grid_is_refused() {
        assert!(Spline::new(&[0.0, 1.0, 2.0], &[0.0, 1.0, 2.0]).is_none());
        assert!(Spline::new(&[0.0, 1.0, 1.0, 2.0], &[0.0; 4]).is_none());
        assert!(Spline::new(&[0.0, 2.0, 1.0, 3.0], &[0.0; 4]).is_none());
    }
}
