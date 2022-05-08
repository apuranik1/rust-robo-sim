use ordered_float::OrderedFloat;
use priority_queue::PriorityQueue;
use std::{
    cmp::Reverse,
    collections::{HashMap, HashSet},
    hash::Hash,
};

use crate::graph::{DirectedEdge, Graph, WithCost};

/// Online implementation of A*

fn trace_prev<'a, T>(prev_map: HashMap<T, T>, from: T) -> Vec<T>
where
    T: Hash + Eq + Copy,
{
    let mut path = vec![from];
    let mut current = from;
    while let Some(prev) = prev_map.get(&current) {
        path.push(*prev);
        current = *prev;
    }
    path.reverse();
    path
}

// first implement basic offline A*
// what the fuck are these type bounds?
pub fn a_star_offline<'a, G, N, E, F>(
    graph: &'a G,
    start: &'a N,
    goal: &'a N,
    mut heuristic: F,
) -> Option<Vec<&'a N>>
where
    N: Hash + Eq,
    E: WithCost + DirectedEdge<N> + 'a,
    G: Graph<N, E>,
    F: FnMut(&N) -> f64,
{
    type Float = OrderedFloat<f64>;
    let mut closed = HashSet::new();
    // priority queue takes max prio, but we want min estimated distance
    let mut to_search: PriorityQueue<&N, Reverse<(Float, Float)>> = PriorityQueue::new();
    let mut prev_map = HashMap::new();
    let h = heuristic(&start).into();
    let g = (0.).into();
    to_search.push(start, Reverse((h, g)));
    while let Some((current, prio)) = to_search.pop() {
        let (_, cost_to) = prio.0;
        if current == goal {
            return Some(trace_prev(prev_map, goal));
        }
        for edge in graph.edges(current) {
            let next = edge.to_node();
            if closed.contains(next) {
                continue;
            }
            let g_next = cost_to + edge.cost();
            let h_next = g_next + heuristic(next);
            let discarded = to_search.push_increase(next, Reverse((h_next, g_next)));
            let new_best = discarded.map_or(true, |discarded| {
                let (g, _h) = discarded.0;
                // if we discarded a g that isn't g_next, g_next is the new best
                g != g_next
            });
            if new_best {
                prev_map.insert(next, current);
            }
        }
        closed.insert(current);
    }
    None
}
// A* outline:
// - take a starting state
// - initialize a set of "open" states to visit, with heuristic
// - initialize a set of "closed" states to visit
pub struct AStarState {}

#[cfg(test)]
mod tests {
    use expect_test::expect;

    #[test]
    fn noop_test() {
        let text = "Hello World";
        let expected = expect![[r#"
            "Hello World"
        "#]];
        expected.assert_debug_eq(&text);
    }
}
