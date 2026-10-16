"""A* around accumulated range hits, inside the known corridor. No obstacle poses."""
import heapq
import math

import numpy as np

from humanoid_avoidance.perception.clustering import scan_to_world_points


class ScanRoute:
    def __init__(self, cfg, radius):
        w = cfg["world"]
        self.spacing = 0.2
        self.padding = radius + 0.3
        self.lower = np.array([w["corridor_x_min_m"], -w["corridor_half_width_m"]])
        upper = self.lower + [w["corridor_length_m"], 2 * w["corridor_half_width_m"]]
        self.shape = tuple((np.floor((upper - self.lower) / self.spacing).astype(int) + 1))
        self.grid = np.indices(self.shape).transpose(1, 2, 0) * self.spacing + self.lower
        self.blocked = np.any((self.grid < self.lower + self.padding) |
                              (self.grid > upper - self.padding), axis=2)
        self.waypoint = None
        # ponytail: static scenes only; moving obstacles need an expiring occupancy map.

    def update(self, scan):
        if scan is None:
            return
        for point in scan_to_world_points(scan):
            self.blocked |= np.linalg.norm(self.grid - point, axis=2) <= self.padding + 0.15

    def path(self, start, goal):
        if not np.all(np.isfinite(start)) or not np.all(np.isfinite(goal)):
            return []
        def cell(xy):
            return tuple(np.rint((np.asarray(xy) - self.lower) / self.spacing).astype(int))

        a, b = cell(start), cell(goal)

        def free(c):
            return all(0 <= c[i] < self.shape[i] for i in (0, 1)) and not self.blocked[c]

        if not free(a) or not free(b):
            return []
        queue = [(math.dist(a, b), 0.0, a)]
        costs, parent = {a: 0.0}, {}
        while queue:
            _, cost, here = heapq.heappop(queue)
            if cost > costs[here]:
                continue
            if here == b:
                cells = [b]
                while cells[-1] != a:
                    cells.append(parent[cells[-1]])
                return [self.lower + np.asarray(c) * self.spacing for c in reversed(cells)]
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1),
                           (1, 1), (1, -1), (-1, 1), (-1, -1)):
                nxt = (here[0] + dx, here[1] + dy)
                if not free(nxt) or not free((here[0] + dx, here[1])) \
                        or not free((here[0], here[1] + dy)):
                    continue
                new_cost = cost + math.hypot(dx, dy)
                if new_cost < costs.get(nxt, math.inf):
                    costs[nxt], parent[nxt] = new_cost, here
                    heapq.heappush(queue, (new_cost + math.dist(nxt, b), new_cost, nxt))
        return []

    def target(self, start, goal):
        def clear(point):
            samples = np.linspace(start, point, max(2, int(np.linalg.norm(point - start) / 0.05) + 1))
            cells = np.rint((samples - self.lower) / self.spacing).astype(int)
            if np.any(cells < 0) or np.any(cells >= self.shape):
                return False
            return not np.any(self.blocked[cells[:, 0], cells[:, 1]])

        # Keep the chosen side until reaching the waypoint; noisy scans must not flip turns.
        if self.waypoint is not None and np.linalg.norm(self.waypoint - start) > 0.25 \
                and clear(self.waypoint):
            return self.waypoint
        path = self.path(start, goal)
        if not path:
            self.waypoint = None
            return np.asarray(start)  # no free route: ask heading control to stop
        # Follow a short, collision-checked segment; never cut an A* corner.
        target = path[0]
        for point in path[1:]:
            if not clear(point):
                break
            target = point
            if np.linalg.norm(point - start) >= 0.8:
                break
        self.waypoint = np.asarray(goal) if len(path) < 3 else target
        return self.waypoint
