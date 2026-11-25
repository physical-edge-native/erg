//! The virtual network between the legs (Issue #34), on Dyne's seeded `VirtNet`.
//!
//! Which legs talk to which is a setting ([`Links`]). For each message, Dyne's virtual network decides whether it is
//! lost and when it arrives; a message sent at control step `k` is delivered at the first control step at or after
//! its arrival time, and never before `k + 1`. The order of the calls fixes the random numbers drawn, so the same
//! seed and the same calls give the same lost and delayed messages on every run and platform.
//!
//! Legs are numbered in the order of erg's `robot.LEGS`: 0 front-left, 1 front-right, 2 hind-left, 3 hind-right.

use dyne_core::net::VirtNet;

/// Number of legs.
pub const N_LEGS: usize = 4;

/// Which legs talk to which.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Links {
    /// No messages between legs.
    None = 0,
    /// Each leg with its two neighbours in the ring front-left, front-right, hind-right, hind-left.
    Neighbours = 1,
    /// Every leg with every other leg.
    All = 2,
}

impl Links {
    /// From the integer used across the WebAssembly interface.
    pub fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Links::None),
            1 => Some(Links::Neighbours),
            2 => Some(Links::All),
            _ => None,
        }
    }

    /// The legs that `leg` sends to, in increasing order.
    pub fn targets(self, leg: usize) -> &'static [usize] {
        const RING: [[usize; 2]; N_LEGS] = [[1, 2], [0, 3], [0, 3], [1, 2]];
        const ALL: [[usize; 3]; N_LEGS] = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]];
        match self {
            Links::None => &[],
            Links::Neighbours => &RING[leg],
            Links::All => &ALL[leg],
        }
    }
}

/// The network between the legs.
pub struct LegNet {
    /// Which legs talk to which.
    pub links: Links,
    /// Dyne's virtual network.
    pub net: VirtNet,
    /// Control period in microseconds.
    pub period_us: u64,
}

impl LegNet {
    /// A network with message loss probability `loss`, base delay `lat_us` and jitter `jitter_us` (a uniform
    /// integer in 0..=2·jitter is added, as in Dyne), drawn from `seed`.
    pub fn new(links: Links, loss: f64, lat_us: u64, jitter_us: u64, seed: u64, period_us: u64) -> Self {
        LegNet { links, net: VirtNet::new(loss, lat_us, jitter_us, Vec::new(), seed), period_us }
    }

    /// Fate of a message from `from` to `to` sent at control step `step`: `None` if lost, otherwise the control step
    /// at which it is delivered (at least `step + 1`).
    pub fn route(&mut self, from: usize, to: usize, step: u64) -> Option<u64> {
        let arrival = self.net.transit(from, to, step * self.period_us)?;
        let at = arrival.div_ceil(self.period_us);
        Some(at.max(step + 1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_is_symmetric_and_neighbour_only() {
        for a in 0..N_LEGS {
            for &b in Links::Neighbours.targets(a) {
                assert!(Links::Neighbours.targets(b).contains(&a));
            }
            assert_eq!(Links::Neighbours.targets(a).len(), 2);
            assert!(!Links::Neighbours.targets(a).contains(&a));
        }
        // front-left and hind-right are diagonal: not neighbours
        assert!(!Links::Neighbours.targets(0).contains(&3));
    }

    #[test]
    fn no_delay_arrives_next_step() {
        let mut n = LegNet::new(Links::All, 0.0, 0, 0, 1, 20_000);
        assert_eq!(n.route(0, 1, 7), Some(8));
    }

    #[test]
    fn one_period_delay_arrives_one_step_later() {
        let mut n = LegNet::new(Links::All, 0.0, 20_000, 0, 1, 20_000);
        assert_eq!(n.route(0, 1, 7), Some(8));
        let mut n = LegNet::new(Links::All, 0.0, 20_001, 0, 1, 20_000);
        assert_eq!(n.route(0, 1, 7), Some(9));
    }
}
