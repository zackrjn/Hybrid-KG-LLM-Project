from dataclasses import dataclass
from typing import List, Optional

from .providers import ClientRouter, load_provider_config


@dataclass
class ChangeBudget:
    max_files: int = 20
    max_total_lines: int = 1500


class ExecutionAgent:
    """Composer-aware execution agent facade.

    This does not call Composer directly; instead it centralizes decision logic
    so Cursor can wire its Composer model behind this interface.
    """

    def __init__(self, change_budget: Optional[ChangeBudget] = None) -> None:
        self.router = ClientRouter(load_provider_config())
        self.change_budget = change_budget or ChangeBudget()

    def should_use_composer(self, files_to_touch: List[str], estimated_total_lines: int) -> bool:
        if not self.router.use_composer():
            return False
        if len(files_to_touch) > self.change_budget.max_files:
            return True
        if estimated_total_lines > self.change_budget.max_total_lines:
            return True
        # Default to Composer for multi-file refactors
        return len(files_to_touch) >= 3


