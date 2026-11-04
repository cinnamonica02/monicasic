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


def validate_schedule(schedule):
    """Reject malformed or reordered command streams at the ISS boundary."""
    if set(schedule) != {"version", "n_layers", "n_experts", "top_k", "commands"}:
        raise ValueError("unsupported schedule fields")
    if schedule["version"] != 1 or schedule["n_layers"] < 1 or schedule["n_experts"] < 1:
        raise ValueError("unsupported schedule version or dimensions")
    if not 1 <= schedule["top_k"] <= schedule["n_experts"]:
        raise ValueError("invalid schedule top_k")
    want = export_schedule(schedule["n_layers"], schedule["n_experts"], schedule["top_k"])["commands"]
    if schedule["commands"] != want:
        raise ValueError("invalid command stream")
