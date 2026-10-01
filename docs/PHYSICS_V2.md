# Flag Arena physics v2

v1 failed review on the seed 165 picture. Balls sank to the bottom of a top-down floor and barely moved. Most exits had no visible cause. The sweeper and the meteors did not decide fights. The cameo was a flat flag. Text sat on the arena, and a lot of the frame was empty. There was no cheap picture to look at before a full render.

v2 deletes those systems. Cast files, guards, flag rasters, the meme loader, the announcer script, the encode pipe, the shared loudness loop, and the verify and CLI skeletons stay.

## Design law

Every elimination has a visible cause, and the viewer sees it coming: telegraph, then impact, then consequence. Nothing pushes a ball unless the log names the source. The only impulses are:

- `ball` — circle-circle contact between country balls
- `dash` — a ball's own dash
- `boss` — the cameo entering, charging, or hitting
- `clash` — the seeded opening pair

Survival braking and linear damping are accelerations, not impulses. Spin is visual and does not feed back into the solver. There is no gravity. Config validation rejects a `gravity` key. There is no wander steering, no sweeper, and no meteors.

## Solver

Numpy only. Fixed dt is 1/240. `PCG64` streams for spawn, AI, and effects are independent. Country balls are radius 44, mass 1, restitution 0.92. Each step applies `v *= exp(-2.2 * dt)` unless a tuned damping value replaces 2.2. Speed is clamped at 1800 px/s, so a step never moves a center more than 7.5 px. The solver runs up to four iterations plus positional correction.

A ball is out when its center's signed distance to the rounded platform is greater than 0. The platform is centered at world (540, 940). The spec table starts at half-width 410 and half-height 460. Corner radius is `0.35 * min(hw, hh)`.

## Tuned knobs

Damping stays 2.2. A lower value moves the opening clash earlier than 0.4 s, and a higher value drops the 2 s mean speed under 140. Dash speed is 1200 rather than 1500 so a single hit does not cross the whole floor. Aggression scale is 1.2. The platform keyframes start at 1020 x 1150 and shrink at about 12 px/s, under the 28 px/s cap, and stay wide through the final duel so one clean hit does not end it in the same tenth of a second. The spec 410 x 460 table wiped the cast before 12 s on every probed seed. Camera zoom is `min(410 / hw, 460 / hh)`, so the larger floor still occupies the same on-screen box and the balls grow as it shrinks. The opening punch is clamped by that same limit, so it only appears after the floor has shrunk.

## What a passing seed must show

Alive counts, the winner window, a final duel of at least 4 s, a first knockout in [1.5, 4] s, and a first impact of at least 700 px/s in [0.4, 1.3] s. At least 80% of eliminations are ball hits, at most 20% are the storm, and at most 10% are unaided falls. The last knockout is a ball hit. The final two cannot fall within 0.2 s of each other. Before the duel, no 1.5 s gap lacks an impact of at least 600 px/s. Alive balls keep a mean speed of at least 140 px/s on every 2 s window until four seconds before the win. On average, at most 45% of alive balls sit in the bottom third, and a ball touches at most 1.2 neighbors.

Search draws 40 seeds. The pass rate must be at least 30%. The only knobs that may move are aggression, damping, dash speed, and the platform keyframes. Gates are not relaxed.

## Budgets

A short seed stays under 3 s. A match that runs out to the horizon is slower than that on this machine. The debug reel stays under 90 s. Preview stays under 5 min. A full render is not started by the agent. If one profiled heavy frame implies more than 25 min, report that and the options (half-resolution bloom, fewer particles) without applying them.
