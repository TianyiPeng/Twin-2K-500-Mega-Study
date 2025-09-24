from typing import Any, Dict


def parse_randomizer_element(flow_elem: Dict[str, Any]) -> Dict[str, Any]:
    """Parse a randomizer element from the survey flow.

    Args:
        flow_elem: Dictionary containing the randomizer element data

    Returns:
        Dictionary containing the parsed randomizer element
    """
    element = {
        "ElementType": "Randomizer",
        "FlowID": flow_elem.get("FlowID", ""),
        "SelectElements": flow_elem.get("SubSet", 1),
    }
    return element
