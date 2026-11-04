"""Export the fixed command schedule for the toy MoE hardware path."""


def export_schedule(n_layers, n_experts, top_k):
    """Return a JSON-serializable static schedule for routing one layer at a time."""
    if n_layers < 1 or n_experts < 1 or not 1 <= top_k <= n_experts:
        raise ValueError("require n_layers >= 1 and 1 <= top_k <= n_experts")

    commands = []
    for layer in range(n_layers):
        commands.append({"op": "ROUTE_TOPK", "layer": layer})
        commands.append({"op": "DISPATCH", "layer": layer})
        for expert in range(n_experts):
            commands.append({"op": "EXPERT", "layer": layer, "expert": expert})
        commands.append({"op": "COMBINE", "layer": layer})
    return {"version": 1, "n_layers": n_layers, "n_experts": n_experts,
            "top_k": top_k, "commands": commands}
