def color_scale(name):
    if name is None:
        name = "Hi-C"
    scales = {
        "Hi-C": [(0, "white"), (0.2, "white"), (1, "red")],
    }

    return scales[name]
