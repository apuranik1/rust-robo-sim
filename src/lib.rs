pub mod a_star;
pub mod graph;
pub mod grid_world;
pub mod kalman;
pub mod particle;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Color {
    r: u8,
    g: u8,
    b: u8,
}

pub mod colors {
    use super::Color;

    pub const BLACK: Color = Color { r: 0, g: 0, b: 0 };
    pub const WHITE: Color = Color {
        r: 0xff,
        g: 0xff,
        b: 0xff,
    };
}

impl Default for Color {
    fn default() -> Self {
        colors::BLACK
    }
}

#[derive(Debug)]
pub struct FuzzyLocation {
    mean: (f64, f64),
    variance: (f64, f64),
}

pub trait GPSSensor {
    fn sense(&self) -> FuzzyLocation;
}

pub fn main() {}
