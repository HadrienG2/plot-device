//! Long-lived API context object, which caches work that does not need to be
//! re-done every time a plot is drawn.

use ::{
    Bidi,
    Color,
    coordinates::AxisRange,
    IntPixels,
    plot2d::{self, Plot2D},
};

use failure;

use std::{
    cmp::Ordering,
    sync::Arc,
};

use vulkano::{
    device::{Device, DeviceExtensions, Queue},
    instance::{
        Features,
        InstanceExtensions,
        PhysicalDevice,
        PhysicalDeviceType,
        QueueFamily,
    },
};

use vulkanoob::{
    instance::EasyInstance,
    Result,
    self,
};


/// Persistent Vulkan setup, shared across plots
pub struct Context {
    // A vulkano instance with some goodies, see the vulkanoob docs for more
    //
    // TODO: Move away from vulkanoob
    //
    _instance: EasyInstance,

    /// Context elements which are used by the plot2d module
    pub(crate) plot2d: plot2d::Context,
}
//
impl Context {
    /// Setup the Vulkan context
    ///
    /// TODO: Make this more customizable
    ///
    pub fn new(multisampling_factor: u32) -> Result<Self> {
        // Set up a Vulkan instance
        let instance = EasyInstance::new(
            Some(&app_info_from_cargo_toml!()),
            &InstanceExtensions::none(),
            vec!["VK_LAYER_LUNARG_standard_validation"]
        )?;

        // Build the common part of the Vulkan context
        let common_context =
            Arc::new(CommonContext::new(&instance,
                                        multisampling_factor)?);

        // Build plot-specific context objects
        let result = Self {
            _instance: instance,
            plot2d: plot2d::Context::new(common_context)?,
        };

        // Return the context object
        Ok(result)
    }

    /// Create a 2D plot
    pub fn new_plot_2d(
        &self,
        size: Bidi<IntPixels>,
        axis_ranges: Bidi<AxisRange>,
        background_color: Color
    ) -> Result<Plot2D> {
        Plot2D::new(&self.plot2d, size, axis_ranges, background_color)
    }
}


/// Parts of the Vulkan context which are shared by all plot types
pub(crate) struct CommonContext {
    // Handle to the Vulkan device in use
    pub(crate) device: Arc<Device>,

    // Handle to a graphics + transfer queue
    //
    // TODO: Study multi-queue workflows
    //
    pub(crate) queue: Arc<Queue>,

    // Degree of multisampling to be used
    pub(crate) multisampling_factor: u32,
}
//
impl CommonContext {
    // Setup the Vulkan context
    //
    // TODO: Make this more customizable
    //
    fn new(instance: &EasyInstance, multisampling_factor: u32) -> Result<Self> {        
        // Setup a Vulkan device and command queue
        let (device, queue) = {
            // Select which physical device we are going to use
            let features = Features {
                robust_buffer_access: true,
                .. Features::none()
            };
            let extensions = DeviceExtensions::none();
            let device_filter = vulkanoob::easy_device_filter(
                &features,
                &extensions,
                self::queue_filter,
                self::device_filter
            );
            let phys_device = instance.select_physical_device(
                device_filter,
                device_preference
            )?.ok_or(failure::err_msg("No suitable Vulkan device found!"))?;

            // Set up our logical device and command queue
            phys_device.setup_single_queue_device(
                &features,
                &extensions,
                queue_filter,
                queue_preference
            )?.expect("Device selection should have prevented this error")
        };

        // Check if the multisampling factor makes sense for that device
        ensure!(multisampling_factor.is_power_of_two(),
                "Invalid multisampling factor");
        ensure!(multisampling_factor &
                device.physical_device()
                      .limits()
                      .framebuffer_color_sample_counts() != 0,
                "Unsupported multisampling factor");

        // Return the context object
        Ok(Self { device, queue, multisampling_factor })
    }
}


// Tells whether we can use a certain physical device or not
fn device_filter(_dev: PhysicalDevice) -> bool {
    // We need no filtering above vulkanoob's default one at this point in time
    true
}

// Tells whether we can use a certain queue family or not
fn queue_filter(family: &QueueFamily) -> bool {
    // For this learning exercise, we want at least a hybrid graphics + compute
    // queue (this implies data transfer support)
    family.supports_graphics() && family.supports_compute()
}

// Tells how acceptable device "dev1" compares to alternative "dev2"
fn device_preference(dev1: PhysicalDevice,
                     dev2: PhysicalDevice) -> Ordering {
    // Device type preference should suffice most of the time
    use self::PhysicalDeviceType::*;
    let type_pref = match (dev1.ty(), dev2.ty()) {
        // If both devices have the same type, this doesn't play a role
        (same1, same2) if same1 == same2 => Ordering::Equal,

        // Discrete GPU goes first
        (DiscreteGpu, _) => Ordering::Greater,
        (_, DiscreteGpu) => Ordering::Less,

        // Then comes integrated GPU
        (IntegratedGpu, _) => Ordering::Greater,
        (_, IntegratedGpu) => Ordering::Less,

        // Then comes virtual GPU
        (VirtualGpu, _) => Ordering::Greater,
        (_, VirtualGpu) => Ordering::Less,

        // Then comes "other" (can't be worse than CPU?)
        (Other, _) => Ordering::Greater,
        (_, Other) => Ordering::Less,

        // We have actually covered all cases, but Rust can't see it :(
        _ => unreachable!(),
    };
    if type_pref != Ordering::Equal { return type_pref; }

    // Figure out which queue family we would pick on each device
    fn target_family(dev: PhysicalDevice) -> QueueFamily {
        dev.queue_families()
           .filter(queue_filter)
           .max_by(queue_preference)
           .expect("Device filtering failed")
    }
    let (fam1, fam2) = (target_family(dev1), target_family(dev2));
    let queue_pref = queue_preference(&fam1, &fam2);
    if queue_pref != Ordering::Equal { return queue_pref; }

    // If control reaches this point, we like both devices equally
    Ordering::Equal
}

// Tells whether we like a certain queue family or not
fn queue_preference(_: &QueueFamily, _: &QueueFamily) -> Ordering {
    // Right now, we only intend to do graphics and compute, on a single queue,
    // without sparse binding magic, so any graphics- and compute-capable queue
    // family is the same by our standards.
    Ordering::Equal
}