"""Standing up after a fall (Issue #59): a fixed sequence of joint targets, the same for both forms of control.

Requirement (docs/design/2026-11-27-redesign.md): the robot returns to standing from lying on either side; first as a
hand-made motion, learning it comes later. A robot of body L1 that falls on its side rolls onto its back once its
legs fold, so the sequence starts by folding the legs and then gets up from the back:

1. fold all legs (FOLD) and wait for the body to settle;
2. the legs of one side ("push side") swing out by abduction and push against the floor (PUSH), while the legs of
   the other side move to OTHER_1: the body rolls up on its side;
3. the other side's legs move to OTHER_2, which brings the body over onto its belly;
4. fold all legs again, then stand up to robot.STAND.

The three leg poses were found by a random search in MuJoCo over the joint ranges (2026-11-27, 300 candidates; this
one got up in every one of 20 trials from the back, from either side and from standing, with random heading, tilt and
joint offsets, at most 0.47 of the stall torque), not designed by hand; they are fixed here. The push side does not
matter on the back; both are tested. Each move is a straight line in joint space over its duration.
"""

import numpy as np

from erg import robot

FOLD_LEG = np.array([0.0, 1.3, -2.5])  # abduction, hip flexion, knee
PUSH = np.array([2.59, -0.55, -2.08])  # legs of the push side; the abduction sign is the side's (left +, right -)
OTHER_1 = np.array([-1.98, 1.12, -0.96])
OTHER_2 = np.array([-2.38, 1.53, -0.78])
MOVE_SECONDS = 2.0  # each of the three big moves
SETTLE_SECONDS = 1.0


def _pose(push_side: int, push: np.ndarray, other: np.ndarray) -> np.ndarray:
    """Joint targets with the push side's legs at `push` and the others at `other` (abduction mirrored)."""
    out = []
    for leg in robot.LEGS:
        side = 1 if leg[1] == "L" else -1
        p = push if side == push_side else other
        out += [push_side * p[0], p[1], p[2]]
    return np.array(out)


def sequence(start: np.ndarray, push_side: int = 1) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """The moves (from targets, to targets, seconds), starting from the current targets `start`."""
    fold = np.tile(FOLD_LEG, len(robot.LEGS))
    a = _pose(push_side, PUSH, OTHER_1)
    b = _pose(push_side, PUSH, OTHER_2)
    return [
        (start, fold, SETTLE_SECONDS),
        (fold, fold, SETTLE_SECONDS),
        (fold, a, MOVE_SECONDS),
        (a, a, 0.3),
        (a, b, MOVE_SECONDS),
        (b, b, 0.3),
        (b, fold, MOVE_SECONDS),
        (fold, fold, 0.5),
        (fold, robot.STAND, 1.5),
        (robot.STAND, robot.STAND, 1.0),
    ]


def duration(moves) -> float:
    return sum(s for _, _, s in moves)


def targets_at(moves, t: float) -> np.ndarray:
    """Joint targets at time t (s) from the start of the sequence; the last targets after its end."""
    for a, b, s in moves:
        if t < s:
            return a + (b - a) * (t / s)
        t -= s
    return moves[-1][1].copy()
