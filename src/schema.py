def build_schema(cfg):
    x_cols = cfg["design_variables"]["names"]
    y_cols = cfg["objectives"]["names"]
    return x_cols, y_cols
