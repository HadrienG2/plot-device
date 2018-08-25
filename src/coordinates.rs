//! This module is responsible for managing coordinates and coordinate systems


/// Standard floating-point coordinate type
///
/// This is the standard floating-point type for coordinate manipulation. 32-bit
/// IEEE-754 floating point provides a bit more than 6 decimal digits of
/// precision, which is adequate for graphics where we deal with at most a few
/// thousand pixels on each side of the screen.
///
pub type FloatCoord = f32;
pub mod float_coord {
    pub use std::f32::*;
}

/// Standard integer coordinate type
///
/// This is the standard integer type for integral coordinates like pixel
/// indices. It was chosen using a similar rationale as Float: 32 bits seem to
/// be enough for current screens, even with supersampling, while 16 bits would
/// be a bit too little.
///
pub type IntCoord = u32;


/// Coordinate system abstraction
///
/// As a plotting application, we need to manipulate data in multiple coordinate
/// systems, including but perhaps not limited to:
///
/// - Axis coordinates (user data, user function input and output)
/// - Pixel coordinates (index of a pixel or pixel edge on a side of the plot)
/// - Distances/angular diameters (for DPI-aware plotting)
/// - API coordinates (what the graphics API ingests for e.g. vertices)
///
/// Since this set is open, a trait sounds like the correct way to abstract
/// this notion and the ways to convert between coordinate systems.
///
/// To save each coordinate system from the trouble of knowing about all other
/// coordinate systems in existence, we introduce a common "normalized"
/// system in which coordinates range from 0.0 to 1.0, where 0.0 represents the
/// beginning of an axis, the upper-left corner of a 2D image, and the front
/// of a 3D frustrum.
///
/// Since Rust does not have const generics yet, it is best to have one trait
/// per coordinate system dimensionality. After all, we'll only need 1D, 2D and
/// 3D on the CPU side, so the duplication is not too bad.
///
pub trait CoordinatesSystem1D {
    /// Transformation from normalized coordinates to this coordinate system
    fn from_normalized(&self) -> CoordinatesTransform1D;

    /// Transformation from this coordinate system back to normalized coords
    fn to_normalized(&self) -> CoordinatesTransform1D {
        self.from_normalized().invert()
    }

    /// Transformation from this coordinate system to another
    fn from(&self, other: &impl CoordinatesSystem1D) -> CoordinatesTransform1D {
        other.to_normalized().then(self.from_normalized())
    }

    /// Transformation from this coordinate system to another
    fn to(&self, other: &impl CoordinatesSystem1D) -> CoordinatesTransform1D {
        self.to_normalized().then(other.from_normalized())
    }
}

/// Transform mapping from one coordinate system to another
///
/// This opaque object is able to map floating-point coordinates from one
/// coordinate system to another.
///
/// At the moment, all our coordinate systems of interest of interest admit
/// affine transforms between each other, so we only handle this use case. But
/// if we need non-linear transforms like spherical coordinates in the future,
/// we will need to review this code accordingly.
///
#[derive(Clone, Copy)]
pub struct CoordinatesTransform1D {
    /// Linear part of the transform
    multiplier: FloatCoord,

    /// Affine part of the transform
    offset: FloatCoord,
}
//
impl CoordinatesTransform1D {
    /// Construct an affine transform
    pub fn affine(multiplier: FloatCoord, offset: FloatCoord) -> Self {
        Self {
            multiplier,
            offset,
        }
    }

    /// Apply the transformation to a coordinate
    pub fn apply(&self, x: FloatCoord) -> FloatCoord {
        x * self.multiplier + self.offset
    }

    /// Compute the inverse of this transform
    fn invert(&self) -> CoordinatesTransform1D {
        CoordinatesTransform1D {
            offset: -self.offset / self.multiplier,
            multiplier: 1. / self.multiplier,
        }
    }

    /// Compute a transform which is equivalent to applying this transform,
    /// followed by another transform (but more efficient).
    fn then(self, other: CoordinatesTransform1D) -> CoordinatesTransform1D {
        CoordinatesTransform1D {
            offset: other.apply(self.offset),
            multiplier: self.multiplier * other.multiplier
        }
    }
}

// TODO: Support higher-dimensional coordinate systems


/// User data coordinate system, used by plot axes
pub struct PlotCoordinates1D {
    /// The inner affine coordinate transform
    ///
    /// We could also compute this quantity lazily in the highly unlikely event
    /// where computing it eagerly in the constructor would turn out to cause
    /// performance issues.
    ///
    transform: CoordinatesTransform1D,
}
//
impl PlotCoordinates1D {
    /// Build the coordinate system of a plot axis
    ///
    /// TODO: Should probably take an AxisRange instead
    ///
    pub fn new(start: FloatCoord, stop: FloatCoord) -> Self {
        Self {
            transform: CoordinatesTransform1D::affine(stop-start, start),
        }
    }
}
//
impl CoordinatesSystem1D for PlotCoordinates1D {
    fn from_normalized(&self) -> CoordinatesTransform1D {
        self.transform
    }
}

/// Pixel coordinates, used for computations that are sensitive to pixel edges
pub struct PixelCoordinates1D {
    /// Number of pixels on this edge of the bitmap
    num_pixels: IntCoord,
}
//
impl PixelCoordinates1D {
    /// Build a pixel-based coordinate system
    pub fn new(num_pixels: IntCoord) -> Self {
        Self {
            num_pixels,
        }
    }

    /// Tell how many pixels there are on this axis
    pub fn num_pixels(&self) -> IntCoord {
        self.num_pixels
    }
}
//
impl CoordinatesSystem1D for PixelCoordinates1D {
    fn from_normalized(&self) -> CoordinatesTransform1D {
        CoordinatesTransform1D {
            multiplier: self.num_pixels as FloatCoord,
            offset: 0.,
        }
    }
}

/// Vulkan's coordinate system, used for vertex positions
#[derive(Default)]
pub struct VulkanCoordinates1D();
//
impl VulkanCoordinates1D {
    /// Build Vulkan's coordinate system
    pub fn new() -> Self { VulkanCoordinates1D() }
}
//
impl CoordinatesSystem1D for VulkanCoordinates1D {
    fn from_normalized(&self) -> CoordinatesTransform1D {
        // Vulkan expects vertex positions from -1 to 1
        CoordinatesTransform1D {
            multiplier: 2.,
            offset: -1.,
        }
    }
}


// TODO: Add tests