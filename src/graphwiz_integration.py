from typing import List, Tuple


Triple = Tuple[str, str, str]


def select_diverse_paths(paths: List[List[Triple]], max_paths: int = 2) -> List[List[Triple]]:
    # Placeholder: select first N paths; extension point to call third_party GraphWiz methods
    return paths[:max_paths]


