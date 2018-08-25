extern crate rayon;

mod coordinates;

use {
    coordinates::{
        CoordinatesSystem1D,
        FloatCoord,
        IntCoord,
        PixelCoordinates1D,
        PlotCoordinates1D,
    },
    rayon::prelude::*,
};


// TODO: Hard-code less stuff
// TODO: Make API less error prone (e.g. make it harder to confuse X/Y coords)
// TODO: Keep API room for surface plots
// TODO: Separate plot style from plot data

// Floating-point type used for user data and axis ranges
pub type Data = FloatCoord;
pub type XData = Data;
pub type YData = Data;

// Floating-point type used for (fractional) pixel coordinates
pub type FracPixels = FloatCoord;
pub type XPixels = FracPixels;
pub type YPixels = FracPixels;

// Integer type used for whole numbers of pixels
pub type IntPixels = IntCoord;


// General abstraction for 2D data
//
// TODO: Should probably be an nalgebra 2D vector for this
//
pub struct Bidi<T> {
    x: T,
    y: T,
}

// Range of an axis
//
// TODO: Add auto-scale support
// TODO: Add nonlinear scale support
// TODO: Check that start != stop
//
pub struct AxisRange {
    start: FloatCoord,
    stop: FloatCoord,
}


// Object-oriented plot struct, which collects all plot-wide parameters and
// provides methods to interact with the plot.
pub struct Plot2D {
    // Horizontal and vertical axis coordinates
    // TODO: Support multiple axes and autoscale
    // TODO: Support axis styling
    // TODO: Support non-linear scales
    x_axis: PlotCoordinates1D,
    y_axis: PlotCoordinates1D,

    // Graphical properties of the plot that will be generated
    x_pixels: PixelCoordinates1D,
    y_pixels: PixelCoordinates1D,

    // Recorded data
    // TODO: Support non-function data
    data: Vec<FunctionData>,

    // Recorded traces
    // TODO: Remove this once rendering is implemented
    traces: Box<[FunctionTrace]>,
}

// Data of a function plot
struct FunctionData {
    // Level of supersampling that was applied on the horizontal axis
    x_supersampling: u8,

    // One function sample on LHS of each x subpixel + one on RHS of last pixel
    y_samples: Box<[YData]>,

    // Desired line thickness in pixels
    // TODO: Add DPI support
    line_thickness: FracPixels,
}

// Trace of a function on a plot
struct FunctionTrace {
    // Function samples, translated to fractional pixel coordinates on the plot
    y_positions: Box<[YPixels]>,

    // One line height sample is measured in the _middle_ of each horizontal
    // subpixel. The height is given in fractional pixels.
    line_heights: Box<[YPixels]>,
}

impl Plot2D {
    // Create a 2D plot
    //
    // TODO: Can pixel dims be extracted from a Vulkan surface?
    //
    pub fn new(size: Bidi<IntPixels>,
               axis_ranges: Bidi<AxisRange>) -> Self {
        assert!(size.x > 0);
        assert!(size.y > 0);
        assert!(axis_ranges.x.start != axis_ranges.x.stop);
        assert!(axis_ranges.y.start != axis_ranges.y.stop);
        Self {
            x_axis: PlotCoordinates1D::new(axis_ranges.x.start,
                                           axis_ranges.x.stop),
            y_axis: PlotCoordinates1D::new(axis_ranges.y.start,
                                           axis_ranges.y.stop),
            x_pixels: PixelCoordinates1D::new(size.x),
            y_pixels: PixelCoordinates1D::new(size.y),
            data: Vec::new(),
            traces: Box::new([]),
        }
    }

    // TODO: Add optional DPI support and associated setup, can do so by
    //       replacing constructor with builder

    // Add a function trace
    //
    // TODO: Support non-function plotting
    // TODO: Keep trace options and trace styling apart
    // TODO: Find a nice mechanism for handling options
    //
    pub fn add_function(
        &mut self,
        function: impl Fn(XData) -> YData + Send + Sync,
        x_supersampling: u8,
        line_thickness: FracPixels,
    ) {
        assert!(x_supersampling > 0);
        let y_samples = self.compute_function_samples(function,
                                                      x_supersampling);
        self.data.push(FunctionData { x_supersampling,
                                      y_samples,
                                      line_thickness });
    }

    // Render function traces
    //
    // TODO: Should ultimately directly render an image via a graphics API
    //
    pub fn render(&mut self) {
        self.traces = self.data.iter()
                               .map(|data| self.render_function_trace(data))
                               .collect::<Vec<_>>()
                               .into_boxed_slice()
    }

    // TODO: Consider whether these functions should go away

    // Sample a function on plot pixel edges
    fn compute_function_samples(
        &self,
        function: impl Fn(XData) -> YData + Send + Sync,
        x_supersampling: u8
    ) -> Box<[YData]> {
        // This function is not meant to be called directly, so can use debug
        debug_assert!(x_supersampling != 0);

        // Build a pixel axis for x subpixels
        let num_x_subpixels =
            self.x_pixels.num_pixels() * x_supersampling as IntCoord;
        let x_subpixels = PixelCoordinates1D::new(num_x_subpixels);

        // Conversion from a sample index to the corresponding x coordinate
        let subpixel_to_x = x_subpixels.to(&self.x_axis);

        // Generate the function samples. Note that we must generate one more
        // sample than we have subpixels because we want a sample on the
        // right-hand side of the rightmost subpixel.
        (0..num_x_subpixels+1).into_par_iter()
                              .map(|idx| subpixel_to_x.apply(idx as FloatCoord))
                              .map(function)
                              .collect::<Vec<_>>()
                              .into_boxed_slice()
    }

    // Render a function trace from function samples
    fn render_function_trace(&self, trace: &FunctionData) -> FunctionTrace {
        // Convert function samples to pixel coordinates
        let y_to_pixel = self.y_axis.to(&self.y_pixels);
        let y_positions = trace.y_samples.iter()
                                         .map(|&y| y_to_pixel.apply(y))
                                         .collect::<Vec<_>>()
                                         .into_boxed_slice();

        // Compute line heights in the middle of each pixel
        let line_heights = self.compute_function_line_heights(
            trace.x_supersampling,
            &y_positions,
            trace.line_thickness
        );

        // TODO: Do the actual rendering instead of merely recording trace data
        FunctionTrace {
            y_positions,
            line_heights,
        }
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
        x_supersampling: u8,
        y_samples: &[YPixels],
        line_thickness: FracPixels
    ) -> Box<[YPixels]> {
        let half_thickness = line_thickness / 2.;
        let inv_dx2 = (x_supersampling as XPixels).powi(2);
        y_samples.windows(2)
                 .map(|y_win| y_win[1] - y_win[0])
                 .map(|dy| half_thickness * (1. + dy.powi(2) * inv_dx2).sqrt())
                 .collect::<Vec<_>>()
                 .into_boxed_slice()
    }
}


#[cfg(test)]
mod tests {
    use ::*;
    use coordinates::float_coord;

    // TODO: Build better tests later on
    #[test]
    fn it_works() {
        // Graph parameters
        const WIDTH: IntPixels = 8192;
        const HEIGHT: IntPixels = 4320;
        let mut plot =
            Plot2D::new(Bidi { x: WIDTH, y: HEIGHT },
                        Bidi { x: AxisRange { start: 0., stop: 6.28 },
                               y: AxisRange { start: -1.2, stop: 1.2 } });

        // Add two function traces
        const X_SUPERSAMPLING: u8 = 2;
        const LINE_THICKNESS: FracPixels = 42.;
        plot.add_function(|x| x.sin(), X_SUPERSAMPLING, LINE_THICKNESS);
        plot.add_function(|x| x.cos(), X_SUPERSAMPLING, LINE_THICKNESS);

        // Check the recorded function data
        const X_SUBPIXELS: IntPixels = WIDTH * (X_SUPERSAMPLING as IntPixels);
        const X_SUBPIXELS_US: usize = X_SUBPIXELS as usize;
        assert_eq!(plot.data.len(), 2);
        for data in &plot.data {
            assert_eq!(data.x_supersampling, X_SUPERSAMPLING);
            assert_eq!(data.y_samples.len(), X_SUBPIXELS_US + 1);
            assert_eq!(data.line_thickness, LINE_THICKNESS);
        }
        for sample_idx in 0..=X_SUBPIXELS_US {
            let mut y_samples = [0.; 2];
            for trace in 0..2 {
                let y_sample = plot.data[trace].y_samples[sample_idx];
                assert!(y_sample >= -1. && y_sample <= 1.);
                y_samples[trace] = y_sample;
            }
            let sum_squares = y_samples[0].powi(2) + y_samples[1].powi(2);
            assert!((sum_squares - 1.).abs() < 2.*float_coord::EPSILON);
        }

        // Render the function traces
        plot.render();

        // Prepare to check the traces
        let y_to_pixel = plot.y_axis.to(&plot.y_pixels);
        let minus_one_pixel = y_to_pixel.apply(-1.);
        let plus_one_pixel = y_to_pixel.apply(1.);

        // Check the recorded function traces
        assert_eq!(plot.traces.len(), 2);
        for trace in &plot.traces[..] {
            // Check the position samples
            assert_eq!(trace.y_positions.len(), X_SUBPIXELS_US + 1);
            for &y_pixel in &trace.y_positions[..] {
                assert!(y_pixel >= minus_one_pixel);
                assert!(y_pixel <= plus_one_pixel);
            }

            // Check the line heights
            assert_eq!(trace.line_heights.len(), X_SUBPIXELS_US);
            for &height in &trace.line_heights[..] {
                assert!(height >= LINE_THICKNESS / 2.);
            }
        }
    }
}
