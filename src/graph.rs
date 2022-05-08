use std::{
    cmp::Eq,
    collections::{HashMap, HashSet},
    hash::Hash,
    rc::Rc,
};

pub trait WithCost {
    fn cost(&self) -> f64;
}
/// Explicit finite graph with enumerable nodes
pub trait Graph<N, E> {
    fn nodes(&self) -> Vec<&N>;
    fn edges(&self, node: &N) -> Vec<&E>;
}

pub trait DirectedEdge<N> {
    fn from_node(&self) -> &N;
    fn to_node(&self) -> &N;
}

/// Implicit graph, potentially with infinite nodes
pub trait ImplicitGraph<N, E> {
    fn edges(&self, node: &N) -> Vec<&E>;
    fn endpoints(&self, edge: &E) -> (&N, &N);
}

/// Basic graph with adjacency vector representation
pub struct SimpleGraph<A> {
    nodes: Vec<Rc<A>>,
    node_index: HashMap<Rc<A>, usize>,
    edges: Vec<HashSet<usize>>,
}

impl<A> SimpleGraph<A>
where
    A: Hash + Eq,
{
    pub fn new(nodes: Vec<A>) -> Self {
        let nodes: Vec<Rc<A>> = nodes.into_iter().map(Rc::new).collect();
        let node_index: HashMap<_, _> = nodes
            .iter()
            .enumerate()
            .map(|(a, b)| (b.clone(), a))
            .collect();
        let edges = { 0..nodes.len() }.map(|_| HashSet::new()).collect();
        Self {
            nodes,
            node_index,
            edges,
        }
    }
}
