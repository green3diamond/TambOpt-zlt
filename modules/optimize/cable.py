"""Cable cost of a layout: the length of the 3-D minimum spanning tree (MST) over the detectors, Up from the surface.

The tree is chosen by Prim's algorithm on detached positions; the length is the sum of its edge lengths with the
positions attached, so it backpropagates. The MST length is continuous in the positions (a tree switch happens where
the old and new trees tie); only its gradient jumps there. `spacing` is a hinge on pairs closer than d_min, so the
cable term cannot pull detectors on top of each other.

    L, h = cable_length(x_east, y_north, surface, d_min)     # metres, m^2
"""
import torch

EPS = 1e-3                       # metres: keeps the edge-length gradient finite at zero distance


def mst_edges(p):
    """(i, j) index tensors of the N - 1 MST edges of the (N, D) points p (Prim, O(N^2)), on CPU, no gradient."""
    p = p.detach().double().cpu()
    d = torch.cdist(p, p)
    n = len(p)
    in_tree = torch.zeros(n, dtype=torch.bool)
    in_tree[0] = True
    best, parent = d[0].clone(), torch.zeros(n, dtype=torch.long)
    ii, jj = [], []
    for _ in range(n - 1):
        j = int(torch.where(in_tree, torch.inf, best).argmin())
        ii.append(int(parent[j])); jj.append(j)
        in_tree[j] = True
        closer = d[j] < best
        best, parent = torch.where(closer, d[j], best), torch.where(closer, j, parent)
    return torch.tensor(ii), torch.tensor(jj)


def points_3d(x, y, surface):
    """(N, 3) East, North, Up; surface(north, east) is SurfaceUpMap (differentiable)."""
    return torch.stack([x, y, surface(y, x)], -1)


def cable_length(x, y, surface, d_min=5.0):
    """MST length [m] and the spacing hinge sum_{i<j} relu(d_min - d_ij)^2 [m^2], both differentiable in (x, y)."""
    p = points_3d(x, y, surface)
    i, j = (t.to(p.device) for t in mst_edges(p))
    L = (((p[i] - p[j]) ** 2).sum(-1) + EPS ** 2).sqrt().sum()
    iu = torch.triu_indices(len(p), len(p), 1, device=p.device)
    d = (((p[iu[0]] - p[iu[1]]) ** 2).sum(-1) + EPS ** 2).sqrt()
    return L, torch.relu(d_min - d).pow(2).sum()
