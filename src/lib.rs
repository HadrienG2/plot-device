#[macro_use] extern crate failure;
#[macro_use] extern crate vulkano;
#[macro_use] extern crate vulkano_shader_derive;

extern crate env_logger;
extern crate image;
extern crate rayon;
extern crate vulkanoob;

mod context;
mod coordinates;
mod plot2d;

use coordinates::{FloatCoord, IntCoord};


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