extern crate rayon;

use rayon::prelude::*;


// TODO: Hard-code less stuff
// TODO: Make API less error prone (e.g. make it harder to confuse X/Y coords)
// TODO: Keep API room for surface plots

// Floating-point type used for axis coordinates
type XCoord = f32;
type YCoord = f32;

// Floating-point type used for fractional pixel coordinates. Absolute positions
// are given from the left side and the bottom side of the graph. 0 is the left
// side of the leftmost pixel / bottom side of the bottom-most pixel.
type Pixels = f32;
type XPixels = Pixels;
type YPixels = Pixels;

// Object-oriented plot struct, which collects all plot-wide parameters and
// provides methods to interact with the plot.
struct Plot2D {
    // Horizontal and vertical axis ranges
    // TODO: Support multiple axes and autoscale
    x_range: (XCoord, XCoord),
    y_range: (YCoord, YCoord),

    // Graphical properties of the plot that will be generated
    x_pixels: u16,
    x_subpixels_per_pixel: u8,  // TODO: Should this vary per-function?
    y_pixels: u16,

    // Recorded traces
    // TODO: Support non-function traces
    traces: Vec<FunctionTrace>,
}

// Trace of a function on a plot
// TODO: Support non-function plotting
struct FunctionTrace {
    // One function sample on LHS of each x subpixel + one on RHS of last pixel.
    //
    // Function samples are given in fractional pixel coordinates, where 0 is
    // the bottom of the bottom-most pixel row and 1 is the top of that pixel.
    //
    y_samples: Box<[YPixels]>,

    // One line height sample is measured in the _middle_ of each horizontal
    // subpixel. The height is given in fractional pixels.
    line_heights: Box<[YPixels]>,
}

impl Plot2D {
    // TODO: Add builder-style plot setup

    // Add a function trace
    // TODO: Support non-function plotting
    pub fn add_function_trace(
        &mut self,
        function: impl Fn(XCoord) -> YCoord + Send + Sync,
        line_thickness: Pixels,
    ) {
        let y_samples = self.compute_function_samples(function);
        let line_heights = self.compute_function_line_heights(&y_samples,
                                                              line_thickness);
        self.traces.push(FunctionTrace { y_samples, line_heights });
    }

    // TODO: Consider whether these functions should go away

    // Compute the vertical samples of a function trace. See y_trace member of
    // the FunctionTrace struct for more info.
    fn compute_function_samples(
        &self,
        function: impl Fn(XCoord) -> YCoord + Send + Sync
    ) -> Box<[YPixels]> {
        // Count how many horizontal subpixels we will generate
        let num_x_subpixels =
                (self.x_pixels as u32) * (self.x_subpixels_per_pixel as u32);

        // Conversion from a sample index to the corresponding x coordinate
        let sample_to_x = |idx: u32| -> XCoord {
            let (min_x, max_x) = self.x_range;
            let fractional_x = (idx as XCoord) / (num_x_subpixels as XCoord);
            min_x + (max_x - min_x) * fractional_x
        };
        // Conversion from a y coordinate to (fractional) vertical pixel
        let y_to_pixel = |y: YCoord| -> YPixels {
            let (min_y, max_y) = self.y_range;
            (y - min_y) / (max_y - min_y) * (self.y_pixels as YCoord)
        };

        // Generate the function samples. Note that we must generate one more
        // sample than we have subpixels because we want a sample on the
        // right-hand side of the rightmost subpixel.
        (0..num_x_subpixels+1).into_par_iter()
                              .map(sample_to_x)
                              .map(function)
                              .map(y_to_pixel)
                              .collect::<Vec<_>>()
                              .into_boxed_slice()
    }

    // Compute line half-height samples in the middle of each x subpixel.
    //
    // "Line height" is the half of the length of the intersection of the
    // vertical axis with a linear interpolant of the function of the given
    // thickness. This tells us how far above and below a given sample the line
    // associated with a function plot should extend.
    //
    // As a bit of trigonometrics will tell you,
    // height = thickness/2 * sqrt(1 + (dy/dx)²).
    //
    fn compute_function_line_heights(
        &self,
        samples: &[YPixels],
        line_thickness: Pixels
    ) -> Box<[YPixels]> {
        let half_thickness = line_thickness / 2.;
        let inv_dx2 = (self.x_subpixels_per_pixel as XPixels).powi(2);
        samples.par_windows(2)
               .map(|y_win| y_win[1] - y_win[0])
               .map(|dy| half_thickness * (1. + dy.powi(2) * inv_dx2).sqrt())
               .collect::<Vec<_>>()
               .into_boxed_slice()
    }
}


#[cfg(test)]
mod tests {
    use ::*;

    // TODO: Build better tests later on
    #[test]
    fn it_works() {
        // Graph parameters
        let mut plot = Plot2D {
            x_range: (0., 6.28),
            y_range: (-1.2, 1.2),
            x_pixels: 8192,
            x_subpixels_per_pixel: 1,
            y_pixels: 4320,
            traces: Vec::new(),
        };

        // Add a function trace of width 42 pixels
        const LINE_THICKNESS: Pixels = 42.;
        plot.add_function_trace(|x| x.sin(), LINE_THICKNESS);

        // Check some properties of the function samples
        let samples = &plot.traces[0].y_samples;
        assert_eq!(samples.len(), (plot.x_pixels + 1) as usize);
        let y_to_pixel = |y|
            (y - plot.y_range.0) / (plot.y_range.1 - plot.y_range.0)
                                 * (plot.y_pixels as YPixels);
        samples.iter().for_each(|s| {
            assert!((*s >= y_to_pixel(-1.)) && (*s <= y_to_pixel(1.)));
        });

        // Check some properties of the line height
        let heights = &plot.traces[0].line_heights;
        assert_eq!(heights.len(), samples.len() - 1);
        heights.iter().for_each(|h| assert!(*h >= LINE_THICKNESS / 2.));
    }
}
