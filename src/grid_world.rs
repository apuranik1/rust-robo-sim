/*
 * Discretized toroidal grid world for histogram localization practice
 */
use super::Color;
use ndarray::{Array, Array2};

pub trait ColorSensor {
    fn sense(&self) -> Color;
}

pub struct Vec2D {
    x: i64,
    y: i64,
}

#[derive(Debug, Clone, Copy)]
pub enum Direction {
    Up,
    Down,
    Left,
    Right,
}

impl From<&Direction> for Vec2D {
    fn from(dir: &Direction) -> Vec2D {
        match dir {
            Direction::Up => Vec2D { x: 0, y: 1 },
            Direction::Down => Vec2D { x: 0, y: -1 },
            Direction::Left => Vec2D { x: -1, y: 0 },
            Direction::Right => Vec2D { x: 1, y: 0 },
        }
    }
}

/* Part 1: a car that localizes itself based on a map of colors.
 * Infers a histogram distribution on a 1D grid.
 */

#[derive(Clone, Copy, Debug)]
pub struct MotionSpec {
    p_move_fail: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct ColorSenseSpec {
    p_inaccurate: f64,
}

#[derive(Debug)]
pub struct ColorLocalizer<'a> {
    world_map: &'a Array2<Color>,
    probabilities: Array2<f64>,
}

fn move_update(
    probabilities: &Array2<f64>,
    dir: Direction,
    motion_spec: MotionSpec,
) -> Array2<f64> {
    let Vec2D { x: dx, y: dy } = (&dir).into();
    // convolve self.probabilities with the appropriate filter
    let (width, height) = probabilities.dim();
    let mut new_probabilities = Array::zeros((width, height));
    let p_fail = motion_spec.p_move_fail;
    let p_move = 1. - p_fail;
    for x in 0..width {
        for y in 0..height {
            // assign p[x][y] to
            let p = probabilities[(x, y)];
            new_probabilities[(x, y)] += p_fail * p;
            let new_x = ((x as i64) + dx).rem_euclid(width as i64) as usize;
            let new_y = ((y as i64) + dy).rem_euclid(height as i64) as usize;
            new_probabilities[(new_x, new_y)] += p_move * p;
        }
    }
    new_probabilities
}

fn sense_update(
    probabilities: &Array2<f64>,
    world_map: &Array2<Color>,
    obs: &Color,
    color_spec: ColorSenseSpec,
) -> Array2<f64> {
    // pointwise multiply and renormalize
    let p_diff = color_spec.p_inaccurate;
    let p_same = 1. - p_diff;
    let filter = world_map.map(|x| if x == obs { p_same } else { p_diff });
    let unnormalized = filter * probabilities;
    let norm = unnormalized.sum();
    unnormalized / norm
}

impl<'a> ColorLocalizer<'a> {
    pub fn new(world_map: &'a Array2<Color>) -> Self {
        let probabilities = Array::ones(world_map.raw_dim());
        Self {
            world_map,
            probabilities,
        }
    }

    pub fn do_move(&mut self, dir: Direction, motion_spec: MotionSpec) {
        self.probabilities = move_update(&self.probabilities, dir, motion_spec);
    }

    pub fn do_sense(&mut self, obs: &Color, sense_spec: ColorSenseSpec) {
        self.probabilities = sense_update(&self.probabilities, &self.world_map, obs, sense_spec);
    }

    pub fn set_position(&mut self, x: usize, y: usize) {
        self.probabilities = Array2::zeros(self.world_map.raw_dim());
        self.probabilities[(x, y)] = 1.;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::colors;
    use expect_test::expect;
    use ndarray::Array2;

    fn array2_to_grid(arr: &Array2<f64>) -> String {
        // print an (x, y) indexed Array2 as a grid
        let mut output = String::new();
        let (width, height) = arr.dim();
        for y_offset in 0..height {
            for x in 0..width {
                let y = (height - y_offset - 1) as usize;
                if x > 0 {
                    output += ", ";
                }
                output += &format!("{:.5}", arr[(x, y)]);
            }
            output += "\n";
        }
        output
    }

    #[test]
    fn test_move_indicator() {
        let mut prob_array = Array2::zeros((3, 3));
        prob_array[(0, 0)] = 1.;
        let motion_spec = MotionSpec { p_move_fail: 0.2 };
        let update1 = move_update(&prob_array, Direction::Right, motion_spec);
        let expected1 = expect![[r#"
            0.00000, 0.00000, 0.00000
            0.00000, 0.00000, 0.00000
            0.20000, 0.80000, 0.00000
        "#]];
        expected1.assert_eq(&array2_to_grid(&update1));
        let update2 = move_update(&update1, Direction::Down, motion_spec);
        let expected2 = expect![[r#"
            0.16000, 0.64000, 0.00000
            0.00000, 0.00000, 0.00000
            0.04000, 0.16000, 0.00000
        "#]];
        expected2.assert_eq(&array2_to_grid(&update2));
    }

    #[test]
    fn test_sense_update() {
        let color_spec = ColorSenseSpec { p_inaccurate: 0.2 };
        let mut world_map = Array2::from_elem((4, 4), colors::BLACK);
        for i in 0..3 {
            world_map[(i, i)] = colors::WHITE;
        }
        let prior: Array2<f64> = Array::ones(world_map.dim());
        let posterior = sense_update(&prior, &world_map, &colors::WHITE, color_spec);
        let expected = expect![[r#"
            0.04000, 0.04000, 0.04000, 0.04000
            0.04000, 0.04000, 0.16000, 0.04000
            0.04000, 0.16000, 0.04000, 0.04000
            0.16000, 0.04000, 0.04000, 0.04000
        "#]];
        expected.assert_eq(&array2_to_grid(&posterior));
    }

    #[test]
    fn test_localize_e2e() {
        use colors::*;
        use Direction::*;
        let mut world_map = Array2::from_elem((5, 4), BLACK);
        let white_squares = vec![(1, 3), (2, 1), (2, 2), (2, 3), (3, 1)];
        for coords in white_squares.iter() {
            world_map[*coords] = WHITE;
        }
        let mut localizer = ColorLocalizer::new(&world_map);
        let motions = vec![None, Some(Right), Some(Down), Some(Down), Some(Right)];
        let observations = vec![WHITE, WHITE, WHITE, WHITE, WHITE];
        let motion_spec = MotionSpec { p_move_fail: 0.2 };
        let sense_spec = ColorSenseSpec { p_inaccurate: 0.3 };
        for (motion, obs) in motions.iter().zip(observations.iter()) {
            if let Some(m) = motion {
                localizer.do_move(*m, motion_spec);
            }
            localizer.do_sense(obs, sense_spec);
        }
        let expected = expect![[r#"
            0.01106, 0.02464, 0.06800, 0.04472, 0.02465
            0.00715, 0.01017, 0.08697, 0.07988, 0.00935
            0.00740, 0.00894, 0.11273, 0.35351, 0.04066
            0.00911, 0.00715, 0.01435, 0.04313, 0.03643
        "#]];
        expected.assert_eq(&array2_to_grid(&localizer.probabilities));
    }
}
