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
type Data = FloatCoord;
type XData = Data;
type YData = Data;

// Floating-point type used for (fractional) pixel coordinates
type FracPixels = FloatCoord;
type XPixels = FracPixels;
type YPixels = FracPixels;


// Object-oriented plot struct, which collects all plot-wide parameters and
// provides methods to interact with the plot.
struct Plot2D {
    // Horizontal and vertical axis coordinates
    // TODO: Support multiple axes and autoscale
    // TODO: Support axis styling
    x_axis: PlotCoordinates1D,
    y_axis: PlotCoordinates1D,

    // Graphical properties of the plot that will be generated
    x_pixels: PixelCoordinates1D,
    x_supersampling: u8,  // TODO: Should this vary per-function?
                          // TODO: Shouldn't that be another PixelCoord1D?
    y_pixels: PixelCoordinates1D,

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
        function: impl Fn(XData) -> YData + Send + Sync,
        line_thickness: FracPixels,
    ) {
        let y_samples = self.compute_function_samples(function);
        // TODO: The following steps should be lazy, not eager
        let line_heights = self.compute_function_line_heights(&y_samples,
                                                              line_thickness);
        self.traces.push(FunctionTrace { y_samples, line_heights });
    }

    // TODO: Consider whether these functions should go away

    // Compute the vertical samples of a function trace. See y_trace member of
    // the FunctionTrace struct for more info.
    fn compute_function_samples(
        &self,
        function: impl Fn(XData) -> YData + Send + Sync
    ) -> Box<[YPixels]> {
        // Build a pixel axis for x subpixels
        let num_x_subpixels =
            self.x_pixels.num_pixels() * self.x_supersampling as IntCoord;
        let x_subpixels = PixelCoordinates1D::new(num_x_subpixels);

        // Conversion from a sample index to the corresponding x coordinate
        let subpixel_to_x = x_subpixels.to(&self.x_axis);

        // Conversion from a y coordinate to (fractional) vertical pixel
        let y_to_pixel = self.y_axis.to(&self.y_pixels);

        // Generate the function samples. Note that we must generate one more
        // sample than we have subpixels because we want a sample on the
        // right-hand side of the rightmost subpixel.
        (0..num_x_subpixels+1).into_par_iter()
                              .map(|idx| subpixel_to_x.apply(idx as FloatCoord))
                              .map(function)
                              .map(|y| y_to_pixel.apply(y))
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
        line_thickness: FracPixels
    ) -> Box<[YPixels]> {
        let half_thickness = line_thickness / 2.;
        let inv_dx2 = (self.x_supersampling as XPixels).powi(2);
        // TODO: Reconsider parallelization of this iteration
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
            x_axis: PlotCoordinates1D::new(0., 6.28),
            y_axis: PlotCoordinates1D::new(-1.2, 1.2),
            x_pixels: PixelCoordinates1D::new(8192),
            x_supersampling: 2,
            y_pixels: PixelCoordinates1D::new(4320),
            traces: Vec::new(),
        };

        // Add a function trace of width 42 pixels
        const LINE_THICKNESS: FracPixels = 42.;
        plot.add_function_trace(|x| x.sin(), LINE_THICKNESS);

        // Check some properties of the function samples
        let samples = &plot.traces[0].y_samples;
        let num_x_subpixels =
            plot.x_pixels.num_pixels() * (plot.x_supersampling as IntCoord) + 1;
        assert_eq!(samples.len(), num_x_subpixels as usize);
        let y_to_pixel = plot.y_axis.to(&plot.y_pixels);
        samples.iter().for_each(|s| {
            assert!(*s >= y_to_pixel.apply(-1.));
            assert!(*s <= y_to_pixel.apply(1.));
        });

        // Check some properties of the line height
        let heights = &plot.traces[0].line_heights;
        assert_eq!(heights.len(), samples.len() - 1);
        heights.iter().for_each(|h| assert!(*h >= LINE_THICKNESS / 2.));
    }
}
