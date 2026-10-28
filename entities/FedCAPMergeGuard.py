"""Removed 2026-10-19: merge guards are not used by the main method."""
class MergeLossGuard:
    def __init__(self, mode="none", tolerance=0.05, min_witnesses=2):
        if mode not in (None, "none"):
            raise ValueError("merge guard has been removed from the main branch")
        self.mode = "none"
        self.implementation = ""
        self.policy = "none"
        self.tolerance = tolerance
        self.min_witnesses = min_witnesses

    def _disabled(self, status="disabled"):
        return {"merge_guard_passed": True, "merge_guard_checked": False,
                "merge_guard_reason": status, "merge_guard_evaluated_clients": 0}

    def result(self, status="disabled"):
        return self._disabled(status)

    def check(self, *args, **kwargs):
        return self._disabled("disabled")

    def checkpoint(self):
        return {"mode": "none"}
