import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import KDTree, Voronoi
from numpy.linalg import norm


def track(file):
    data = np.loadtxt(file, delimiter=",", skiprows=1, dtype=str)
    x = data[:, 0].astype(float)
    y = data[:, 1].astype(float)
    types = data[:, 2]
    cones = np.column_stack((x[types == "cone"], y[types == "cone"]))
    start = np.array([x[types == "start"][0], y[types == "start"][0]])
    return cones, start


def preprocess(cones, start, dmax=7.0, min_neighbors=2, start_radius=40.0):

    pts = cones.copy()

    for _ in range(10):
        if len(pts) < 4:
            break
        tree = KDTree(pts)
        counts = np.array([len(tree.query_ball_point(c, dmax)) - 1 for c in pts])
        keep = counts >= min_neighbors
        if keep.sum() == len(pts):
            break
        pts = pts[keep]


    connect_radius = dmax * 1.5
    tree = KDTree(pts)
    _, seed_idx = tree.query(start)

    visited = set()
    queue = [seed_idx]
    visited.add(seed_idx)
    while queue:
        cur = queue.pop()
        neighbors = tree.query_ball_point(pts[cur], connect_radius)
        for nb in neighbors:
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)

    keep_mask = np.array([i in visited for i in range(len(pts))])
    pts = pts[keep_mask]

    return pts


def Voronoi1(cones, dmin=1.5, dmax=18.0):

    vor = Voronoi(cones)
    cmin = cones.min(axis=0) - 1
    cmax = cones.max(axis=0) + 1
    midpoints = []
    for p1, p2 in vor.ridge_points:
        c1 = cones[p1]
        c2 = cones[p2]
        d = norm(c1 - c2)
        if dmin < d < dmax:
            mid = (c1 + c2) / 2
            if np.all(mid >= cmin) and np.all(mid <= cmax):
                midpoints.append(mid)
    return np.array(midpoints)


def filter(midpoints, cones, min_cone_dist=1.0):
    if len(midpoints) == 0:
        return midpoints
    tree = KDTree(cones)
    dists, _ = tree.query(midpoints)
    return midpoints[dists >= min_cone_dist]


def centerline1(midpoints, start, max_step=16.0, k=15, dot_thresh=-0.2):

    tree = KDTree(midpoints)
    _, seed = tree.query(start)
    path = [seed]
    used = {seed}
    heading = None

    while True:
        cur = path[-1]
        dists, idxs = tree.query(midpoints[cur], k=k + 1)
        best, best_score = None, -np.inf

        for dist, idx in zip(dists[1:], idxs[1:]):
            if idx in used or dist > max_step:
                continue
            vec = midpoints[idx] - midpoints[cur]
            vec_norm = norm(vec)
            if vec_norm < 1e-9:
                continue
            vec = vec / vec_norm

            if heading is not None:
                dot = float(np.dot(vec, heading))
                if dot < dot_thresh:
                    continue
                score=dot-0.04*dist
            else:
                score =-dist

            if score > best_score:
                best_score, best = score, idx

        if best is None:
            break

        new_heading = midpoints[best] - midpoints[cur]
        h_norm = norm(new_heading)
        if h_norm > 1e-9:
            heading = new_heading / h_norm

        path.append(best)
        used.add(best)

    return midpoints[np.array(path)]


def smooth_centerline(centerline, window=7):
    if len(centerline) <= window:
        return centerline
    pad = window // 2
    smoothed = np.copy(centerline)
    for i in range(pad, len(centerline) - pad):
        smoothed[i] = centerline[i - pad:i + pad + 1].mean(axis=0)
    return smoothed


def classify(cones, centerline):
    tree = KDTree(centerline)
    left, right = [], []
    for c in cones:
        k = min(5, len(centerline))
        _, idxs = tree.query(c, k=k)
        cross_sum = 0.0
        for idx in idxs:
            if idx < len(centerline) - 1:
                seg = centerline[idx + 1] - centerline[idx]
            else:
                seg = centerline[idx] - centerline[idx - 1]
            cross = seg[0] * (c[1] - centerline[idx][1]) - seg[1] * (c[0] - centerline[idx][0])
            cross_sum += cross
        (left if cross_sum > 0 else right).append(c)
    return np.array(left), np.array(right)


def order(points, start_ref, maxhop=9.0):
    if len(points) == 0:
        return points
    tree = KDTree(points)
    _, start_idx = tree.query(start_ref)
    ordered = [start_idx]
    used = {start_idx}

    while len(used) < len(points):
        cur = ordered[-1]
        dists, idxs = tree.query(points[cur], k=min(10, len(points)))
        moved = False
        for d, i in zip(dists[1:], idxs[1:]):
            if i not in used and d < maxhop:
                ordered.append(i)
                used.add(i)
                moved = True
                break
        if not moved:
            break

    return points[np.array(ordered)]


def resample(pts, n):
    dists = np.concatenate([[0], np.cumsum(norm(np.diff(pts, axis=0), axis=1))])
    if dists[-1] < 1e-9:
        return pts
    t = np.linspace(0, dists[-1], n)
    x = np.interp(t, dists, pts[:, 0])
    y = np.interp(t, dists, pts[:, 1])
    return np.column_stack([x, y])


def recenter(left, right, n=300):
    if len(left) < 2 or len(right) < 2:
        return None
    l = resample(left, n) 
    r = resample(right,n)

    tree = KDTree(r)
    _, idxs = tree.query(l)
    center = (l + r[idxs]) / 2
    return center


def plot_track(cones, centerline, left, right, start):

    plt.figure(figsize=(14,10))

    plt.scatter(cones[:,0], cones[:,1], c='black', s=20, zorder=3, label="cones")
    plt.scatter(start[0], start[1], c='orange', s=150, zorder=6, label="start")

    if centerline is not None and len(centerline) > 0:
        closed_centerline = np.vstack([centerline, centerline[0]])
        plt.plot(closed_centerline[:,0], closed_centerline[:,1],
                 color="green", linewidth=3, zorder=2, label="centerline")

    if len(left) > 0:
        plt.plot(left[:,0], left[:,1],
                 color="blue", linewidth=2.5, zorder=3, label="left boundary")

    if len(right) > 0:
        plt.plot(right[:,0], right[:,1],
                 color="yellow", linewidth=2.5, zorder=4, label="right boundary")

    plt.legend()
    plt.gca().set_aspect("equal")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def build(file, show_plot=False):
    cones, start = track(file)

    cones = preprocess(cones, start, dmax=7.0, min_neighbors=2)
    midpoints = Voronoi1(cones, dmin=1.5, dmax=9.0)
    midpoints = filter(midpoints, cones, min_cone_dist=0.8)
    centerline = centerline1(midpoints, start, max_step=8.0, k=15, dot_thresh=-0.2)
    centerline = smooth_centerline(centerline, window=7)

    # Classify & order boundaries
    left, right = classify(cones, centerline)
    left = order(left, start, maxhop=9.0)
    right = order(right, start, maxhop=9.0)

    # Recompute clean centerline from boundaries
    refined = recenter(left, right, n=300)
    if refined is not None:
        centerline = refined

    if show_plot:
        plot_track(cones, centerline, left, right, start)

    return left, right, centerline


if __name__ == "__main__":
    build("cones.txt",show_plot=True)