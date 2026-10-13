"""Two first-order objective guards, an optimization ablation, not novelty.

Project a proposed parameter step onto two homogeneous descent halfspaces.
The guarantee is about local directional derivatives, never finite loss or
generalization. This does not partition facial channels or decoder routes.
"""
import torch


def project_descent_step(step, first_gradient, second_gradient):
    if step.ndim != 1 or step.shape != first_gradient.shape or step.shape != second_gradient.shape:
        raise ValueError('expected three flat vectors with equal shape')
    if not all(torch.isfinite(x).all() for x in (step, first_gradient, second_gradient)):
        raise ValueError('nonfinite update')
    # Double precision makes nearly opposing objectives and tiny optimizer
    # steps safe. Unit normals make constraints invariant to loss weighting.
    d = step.double()
    normals = []
    for gradient in (first_gradient, second_gradient):
        g = gradient.double()
        norm = g.norm()
        if norm > 1e-15:
            normals.append(g / norm)
    if not normals:
        return step.clone()
    a = torch.stack(normals)
    tolerance = 1e-12 * max(float(d.norm()), 1e-12)
    def feasible(x):
        return bool(((a @ x) <= tolerance).all())
    if feasible(d):
        return step.clone()
    # Active-set candidates: either boundary or both boundaries. Zero is
    # always feasible, including degenerate opposing-gradient cases.
    candidates = [torch.zeros_like(d)]
    for n in normals:
        candidate = d - torch.dot(n, d).clamp_min(0.) * n
        if feasible(candidate):
            candidates.append(candidate)
    if len(normals) == 2:
        gram = a @ a.T
        if torch.linalg.det(gram) > 1e-12:
            multiplier = torch.linalg.solve(gram, a @ d)
            candidate = d - a.T @ multiplier
            if bool((multiplier >= -tolerance).all()) and feasible(candidate):
                candidates.append(candidate)
        else:
            # For exactly opposite normals the shared boundary is feasible.
            candidate = d - torch.dot(normals[0], d) * normals[0]
            if feasible(candidate):
                candidates.append(candidate)
    costs = torch.stack([(x-d).square().sum() for x in candidates])
    return candidates[int(costs.argmin())].to(step.dtype)


def flat_grad(loss, parameters, *, retain_graph=True):
    gradients = torch.autograd.grad(loss, parameters, retain_graph=retain_graph, allow_unused=True)
    return torch.cat([torch.zeros_like(p).flatten() if g is None else g.detach().flatten()
                      for p, g in zip(parameters, gradients)])


@torch.no_grad()
def flat_parameters(parameters):
    return torch.cat([p.detach().flatten() for p in parameters])


@torch.no_grad()
def assign_parameters(parameters, vector):
    position = 0
    for p in parameters:
        p.copy_(vector[position:position+p.numel()].view_as(p))
        position += p.numel()
    if position != vector.numel():
        raise ValueError('parameter vector size mismatch')
