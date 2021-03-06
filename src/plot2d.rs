//! 2D (aka XY) plotting functionality goes here

use ::{
    Bidi,
    Color,
    FracPixels,
    IntPixels,
    context::CommonContext,
    coordinates::{
        AxisRange,
        CoordinatesSystem1D,
        FloatCoord,
        IntCoord,
        PixelCoordinates1D,
        PlotCoordinates1D,
        VulkanCoordinates1D,
    },
    XData,
    XPixels,
    YData,
    YPixels,
};

use failure;

use image::{ImageBuffer, Rgba};

use rayon::prelude::*;

use std::sync::Arc;

use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer, ImmutableBuffer},
    command_buffer::{AutoCommandBufferBuilder, CommandBuffer, DynamicState},
    descriptor::PipelineLayoutAbstract,
    format::{ClearValue, Format},
    framebuffer::{Framebuffer, RenderPassAbstract, Subpass},
    image::{AttachmentImage, Dimensions, StorageImage},
    pipeline::{
        GraphicsPipeline,
        vertex::SingleBufferDefinition,
        viewport::Viewport,
    },
    self,
    sync::GpuFuture,
};

use vulkanoob::Result;


// TODO: Hard-code less stuff
// TODO: Make API less error prone (e.g. make it harder to confuse X/Y coords)

// Object-oriented plot struct, which collects all plot-wide parameters and
// provides methods to interact with the plot.
pub struct Plot2D<'a> {
    // Long-lived drawing context
    context: &'a Context,

    // Horizontal and vertical axis coordinates
    // TODO: Support multiple axes and autoscale
    // TODO: Support axis styling
    // TODO: Support non-linear scales
    x_axis: PlotCoordinates1D,
    y_axis: PlotCoordinates1D,

    // Graphical properties of the plot that will be generated
    x_pixels: PixelCoordinates1D,
    y_pixels: PixelCoordinates1D,

    // Background color
    background_color: ClearValue,

    // Recorded data
    // TODO: Support non-function data
    data: Vec<FunctionData>,

    // Recorded traces
    // TODO: Remove this once rendering is implemented
    traces: Box<[FunctionTrace]>,
}
//
// Data of a function plot
struct FunctionData {
    // Level of supersampling that was applied on the horizontal axis
    x_supersampling: u8,

    // One function sample on LHS of each x subpixel + one on RHS of last pixel
    y_samples: Box<[YData]>,

    // Desired line thickness in pixels
    // TODO: Add DPI support
    line_thickness: FracPixels,

    // Line color
    // TODO: Separate style and data better
    color: Color,
}
//
// Trace of a function on a plot
struct FunctionTrace {
    // Function samples, translated to fractional pixel coordinates on the plot
    y_positions: Box<[YPixels]>,

    // One line height sample is measured in the _middle_ of each horizontal
    // subpixel. The height is given in fractional pixels.
    line_heights: Box<[YPixels]>,

    // Vertices of the triangle strip
    strip_vertices: Box<[Vertex]>,
}
//
// A vertex for the Vulkan renderer
#[derive(Copy, Clone)]
struct Vertex {
    // Position of the vertex in Vulkan coordinates
    position: [f32; 2],
}
impl_vertex!(Vertex, position);
//
impl<'a> Plot2D<'a> {
    // Create a 2D plot
    //
    // TODO: Can pixel dims be extracted from a Vulkan surface?
    //
    pub(crate) fn new(
        context: &'a Context,
        size: Bidi<IntPixels>,
        axis_ranges: Bidi<AxisRange>,
        background_color: Color,
    ) -> Result<Self> {
        ensure!(size.x > 0, "Invalid horizontal plot size");
        ensure!(size.y > 0, "Invalid vertical plot size");
        Ok(Self {
            context,
            x_axis: PlotCoordinates1D::new(axis_ranges.x),
            y_axis: PlotCoordinates1D::new(axis_ranges.y.invert()),
            x_pixels: PixelCoordinates1D::new(size.x),
            y_pixels: PixelCoordinates1D::new(size.y),
            background_color: background_color.into(),
            data: Vec::new(),
            traces: Box::new([]),
        })
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
        color: Color
    ) -> Result<()> {
        ensure!(x_supersampling > 0, "Supersampling factor must be positive");
        ensure!(line_thickness > 0., "Line thickness must be positive");
        let y_samples = self.compute_function_samples(function,
                                                      x_supersampling);
        self.data.push(FunctionData { x_supersampling,
                                      y_samples,
                                      line_thickness,
                                      color });
        Ok(())
    }

    // Render function traces to image files
    //
    // TODO: Should be able to render to a window too
    //
    pub fn render(&mut self, filename: &str) -> Result<()> {
        // TODO: For now we keep the full trace data around, later we'll just
        //       drop it at the end and only return the final image.
        // TODO: Think about how testing should be redesigned to account for it.
        self.traces = self.data.iter()
                               .map(|data| self.render_function_trace(data))
                               .collect::<Vec<_>>()
                               .into_boxed_slice();

        // Get quick access to the Vulkan device and command queue
        let device = self.context.common.device.clone();
        let queue = self.context.common.queue.clone();

        // Schedule sending the plot vertices to the GPU
        // TODO: This has pretty bad heap usage patterns, can it be done better?
        let mut vx_bufs = Vec::with_capacity(self.traces.len());
        let mut vx_upload_future: Box<GpuFuture> =
            Box::new(vulkano::sync::now(device.clone()));
        for trace in &self.traces[..] {
            let (vx_buf, vx_future) =
                ImmutableBuffer::from_iter(trace.strip_vertices.iter().cloned(),
                                           BufferUsage {
                                               vertex_buffer: true,
                                               .. BufferUsage::none()
                                           },
                                           queue.clone())?;
            vx_bufs.push(vx_buf);
            vx_upload_future = Box::new(vx_upload_future.join(vx_future));
        }

        // Shorthand for the output image properties
        let width = self.x_pixels.num_pixels();
        let height = self.y_pixels.num_pixels();
        let multisampling_factor = self.context.common.multisampling_factor;

        // Define the viewport transform, now that we know the target image dims
        let dynamic_state = DynamicState {
            viewports: Some(vec![Viewport {
                origin: [0.0, 0.0],
                dimensions: [width as f32, height as f32],
                depth_range: 0.0 .. 1.0,
            }]),
            .. DynamicState::none()
        };

        // Prepare the multisampled image
        let ms_image =
            AttachmentImage::transient_multisampled(device.clone(),
                                                    [width, height],
                                                    multisampling_factor,
                                                    Format::R8G8B8A8Unorm)?;

        // Prepare the target image
        let final_image =
            StorageImage::new(device.clone(),
                              Dimensions::Dim2d { width, height },
                              Format::R8G8B8A8Unorm,
                              Some(queue.family()))?;

        // Attach this image to the render pass using a framebuffer
        let framebuffer = Arc::new(
            Framebuffer::start(self.context.render_pass.clone())
                        .add(ms_image.clone())?
                        .add(final_image.clone())?
                        .build()?
        );

        // Create a buffer to copy the final image contents in
        // TODO: Avoid using CpuAccessibleBuffer?
        let buf_size = width * height * 4;
        let buf =
            CpuAccessibleBuffer::from_iter(device.clone(),
                                           BufferUsage {
                                              transfer_destination: true,
                                              .. BufferUsage::none()
                                           },
                                           (0 .. buf_size).map(|_| 0u8))?;

        // ...and we are ready to build our drawing command buffer
        let clear_values = vec![self.background_color, ClearValue::None];
        let mut command_buffer_builder =
            AutoCommandBufferBuilder::primary_one_time_submit(device.clone(),
                                                              queue.family())?
                                     .begin_render_pass(framebuffer.clone(),
                                                        false,
                                                        clear_values)?;
        for (idx, vx_buf) in vx_bufs.iter().enumerate() {
            let push_constants = fragment_shader::ty::PushConstants {
                trace_color: self.data[idx].color,
            };
            command_buffer_builder =
                command_buffer_builder.draw(self.context.pipeline.clone(),
                                            &dynamic_state,
                                            vx_buf.clone(),
                                            (),
                                            push_constants)?;
        }
        let command_buffer =
            command_buffer_builder.end_render_pass()?
                                  .copy_image_to_buffer(final_image.clone(),
                                                        buf.clone())?
                                  .build()?;

        // Run the draw calls after the vertices are uploaded
        command_buffer.execute_after(vx_upload_future, queue.clone())?
                      .then_signal_fence_and_flush()?
                      .wait(None)?;

        // ...then build an image out of the plot and save it
        let content = buf.read()?;
        let image =
            ImageBuffer::<Rgba<u8>, _>::from_raw(width, height, content)
                        .ok_or(failure::err_msg("Unexpected buffer size"))?;
        Ok(image.save(filename)?)
    }

    // Sample a function on plot pixel edges
    fn compute_function_samples(
        &self,
        function: impl Fn(XData) -> YData + Send + Sync,
        x_supersampling: u8
    ) -> Box<[YData]> {
        // This function is not meant to be called directly, so can use debug
        debug_assert!(x_supersampling > 0);

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

        // Compute the triangle strip directly
        let strip_vertices = self.compute_function_strip_vertices(
            &y_positions,
            &line_heights
        );

        // Transmit the traces to the caller
        FunctionTrace {
            y_positions,
            line_heights,
            strip_vertices,
        }
    }

    // Compute line height samples in the middle of each x subpixel.
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
        debug_assert!(x_supersampling > 0);
        debug_assert!(line_thickness > 0.);
        let half_thickness = line_thickness / 2.;
        let inv_dx2 = (x_supersampling as XPixels).powi(2);
        y_samples.windows(2)
                 .map(|y_win| y_win[1] - y_win[0])
                 .map(|dy| half_thickness * (1. + dy.powi(2) * inv_dx2).sqrt())
                 .collect::<Vec<_>>()
                 .into_boxed_slice()
    }

    // Compute a triangle strip from function position and line height samples
    //
    // Note that since function samples lie on subpixel edges and line height
    // samples lie on pixel centers, there is one more function position sample
    // than there are line height samples.
    //
    fn compute_function_strip_vertices(
        &self,
        y_positions: &[YPixels],
        line_heights: &[YPixels]
    ) -> Box<[Vertex]> {
        // Check that the input slices match our expectations
        debug_assert_eq!(y_positions.len(), line_heights.len() + 1);

        // Interpolate line heights on pixel edges, clamping to edge value
        let mut interp_heights = Vec::with_capacity(y_positions.len());
        interp_heights.push(line_heights[0]);
        for heights in line_heights.windows(2) {
            interp_heights.push((heights[0] + heights[1]) / 2.);
        }
        interp_heights.push(line_heights[line_heights.len() - 1]);
        let interp_heights = interp_heights.into_boxed_slice();

        // Prepare coordinate conversions from pixel coordinates to Vulkan ones
        let num_x_subpixels_us = line_heights.len();
        let num_x_subpixels = num_x_subpixels_us as IntPixels;
        let x_subpixels = PixelCoordinates1D::new(num_x_subpixels);
        let x_subpixel_to_vulkan = x_subpixels.to(&VulkanCoordinates1D);
        let y_pixel_to_vulkan = self.y_pixels.to(&VulkanCoordinates1D);

        // Compute the final triangle strip
        let mut strip_vertices = Vec::with_capacity(2 * num_x_subpixels_us);
        for (i, (y, h)) in y_positions.iter().zip(interp_heights.iter())
                                             .enumerate()
        {
            let x_vulkan = x_subpixel_to_vulkan.apply(i as FloatCoord);
            let y_top = y_pixel_to_vulkan.apply(y - h);
            strip_vertices.push(Vertex { position: [x_vulkan, y_top] });
            let y_bottom = y_pixel_to_vulkan.apply(y + h);
            strip_vertices.push(Vertex { position: [x_vulkan, y_bottom] });
        }
        strip_vertices.into_boxed_slice()
    }
}


/// Vertex shader used for 2D plots
mod vertex_shader {
    #![allow(unused)]
    #[derive(VulkanoShader)]
    #[ty = "vertex"]
    #[src = "
#version 460

layout(location = 0) in vec2 position;

void main() {
    gl_Position = vec4(position, 0.0, 1.0);
}
    "]
    struct _Dummy;
}

/// Fragment shader used for 2D plots
/// TODO: Let the user pick the draw color using push constants
mod fragment_shader {
    #![allow(unused)]
    #[derive(VulkanoShader)]
    #[ty = "fragment"]
    #[src = "
#version 460

layout(push_constant) uniform PushConstants {
    vec4 trace_color;
} u_pushConstants;

layout(location = 0) out vec4 f_color;

void main() {
    f_color = u_pushConstants.trace_color;
}
    "]
    struct _Dummy;
}

// Long-lived Vulkan context associated with Plot2D
pub(crate) struct Context {
    // Reference to the common Vulkan context shared by all plot types
    common: Arc<CommonContext>,

    // Render pass describing rendering targets
    render_pass: Arc<RenderPassAbstract + Send + Sync>,

    // Graphics pipeline description
    pipeline: Arc<GraphicsPipeline<SingleBufferDefinition<Vertex>,
                                   Box<PipelineLayoutAbstract + Send + Sync>,
                                   Arc<RenderPassAbstract + Send + Sync>>>,
}
//
impl Context {
    /// Build the Plot2D-specific context
    pub(crate) fn new(common_context: Arc<CommonContext>) -> Result<Self> {
        // Load the vertex shader and fragment shader
        let device = common_context.device.clone();
        let vs = vertex_shader::Shader::load(device.clone())?;
        let fs = fragment_shader::Shader::load(device.clone())?;

        // Build the render pass
        let render_pass: Arc<RenderPassAbstract + Send + Sync> = Arc::new(
            single_pass_renderpass!(
                device.clone(),
                attachments: {
                    intermediary: {
                        load: Clear,
                        store: DontCare,
                        format: Format::R8G8B8A8Unorm,
                        samples: common_context.multisampling_factor,
                    },
                    color: {
                        load: DontCare,
                        store: Store,
                        format: Format::R8G8B8A8Unorm,
                        samples: 1,
                    }
                },
                pass: {
                    color: [intermediary],
                    depth_stencil: {},
                    resolve: [color],
                }
            )?
        );

        // Build the graphics pipeline
        let pipeline = Arc::new(
            GraphicsPipeline::start()
                             // - We have only one source of vertex inputs
                             .vertex_input_single_buffer::<Vertex>()
                             // - We use a triangle strip
                             .triangle_strip()
                             // - This is our vertex shader
                             .vertex_shader(vs.main_entry_point(), ())
                             // - This sets up the target image region
                             .viewports_dynamic_scissors_irrelevant(1)
                             // - This is our fragment shader
                             .fragment_shader(fs.main_entry_point(), ())
                             // - The pipeline is used in this render pass
                             .render_pass(
                                Subpass::from(render_pass.clone(), 0)
                                        .ok_or(failure::err_msg("Bad subpass"))?
                             )
                             // - Enable alpha blending
                             .blend_alpha_blending()
                             // - Here is the target device: build the pipeline!
                             .build(device.clone())?
        );

        // Return the 2D plotting context
        Ok(Self {
            common: common_context,
            render_pass,
            pipeline,
        })
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use ::{
        context,
        coordinates::float_coord,
    };

    use env_logger;

    // TODO: Build better tests later on
    #[test]
    fn it_works() {
        // Enable the env_logger logging implementation
        env_logger::init();

        // Set up a drawing context
        let context = context::Context::new(8).unwrap();

        // Create a plot
        const WIDTH: IntPixels = 8192;
        const HEIGHT: IntPixels = 4320;
        let x_range = AxisRange::new(-3.14, 3.14).unwrap();
        let y_range = AxisRange::new(-1.2, 1.2).unwrap();
        let mut plot =
            context.new_plot_2d(Bidi { x: WIDTH, y: HEIGHT },
                                Bidi { x: x_range, y: y_range },
                                [0.2, 0.2, 1., 0.8]).unwrap();

        // Add two function traces
        const X_SUPERSAMPLING: u8 = 64;
        const LINE_THICKNESS: FracPixels = 42.;
        plot.add_function(|x| (10.*x).sin(),
                           X_SUPERSAMPLING,
                           LINE_THICKNESS,
                           [1., 0., 0., 0.6]).unwrap();
        plot.add_function(|x| (10.*x).cos(),
                           X_SUPERSAMPLING,
                           LINE_THICKNESS,
                           [0., 1., 0., 0.6]).unwrap();

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
            assert!((sum_squares - 1.).abs() <= 2.*float_coord::EPSILON);
        }

        // Render the function traces
        plot.render("plot.png").unwrap();

        // Prepare to check the traces
        let y_to_pixel = plot.y_axis.to(&plot.y_pixels);
        let plus_one_pixel = y_to_pixel.apply(1.);
        let minus_one_pixel = y_to_pixel.apply(-1.);

        // Check the recorded function traces
        assert_eq!(plot.traces.len(), 2);
        for trace in &plot.traces[..] {
            // Check the position samples
            assert_eq!(trace.y_positions.len(), X_SUBPIXELS_US + 1);
            for &y_pixel in &trace.y_positions[..] {
                assert!(y_pixel >= plus_one_pixel);
                assert!(y_pixel <= minus_one_pixel);
            }

            // Check the line heights
            assert_eq!(trace.line_heights.len(), X_SUBPIXELS_US);
            for &height in &trace.line_heights[..] {
                assert!(height >= LINE_THICKNESS / 2.);
            }

            // Check the triangle strip
            let strip_vertices = &trace.strip_vertices;
            // - Two vertices per subpixel edge (top and bottom)
            assert_eq!(strip_vertices.len(), 2*trace.y_positions.len());
            // - First vertex is on the left edge of the graph
            assert_eq!(strip_vertices[0].position[0], -1.);
            // - Vertices go by pairs of identical x coordinates
            let mut last_x = float_coord::NEG_INFINITY;
            for vx_pair in strip_vertices.chunks(2) {
                assert_eq!(vx_pair[0].position[0], vx_pair[1].position[0]);
                // - x coordinate is strictly growing along the strip
                assert!(vx_pair[0].position[0] > last_x);
                last_x = vx_pair[0].position[0];
                // - Top vertex goes first, followed by bottom vertex
                assert!(vx_pair[0].position[1] < vx_pair[1].position[1]);
                for vertex in vx_pair {
                    for &coord in &vertex.position {
                        // - Every vertex is inside of the Vulkan viewport
                        assert!((coord >= -1.) && (coord <= 1.));
                    }
                }
            }
            // - Last vertex is on the right edge of the graph
            assert_eq!(strip_vertices[strip_vertices.len()-1].position[0], 1.);
        }

        // TODO: Test the actual rendering
    }
}
