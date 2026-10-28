"""Removed 2026-10-19: oracle row-sharing diagnostic is not used by the main method."""
class FedCAPOracleClassShareServer:
    implementation = "none"
    head_aggregation = "none"
    def __init__(self, *args, **kwargs):
        raise RuntimeError("oracle class sharing has been removed")

def oracle_label_maps_for_round(*args, **kwargs):
    raise RuntimeError("oracle class sharing has been removed")
