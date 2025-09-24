from typing import Any, Dict


def parse_group_element(flow_elem: Dict[str, Any]) -> Dict[str, Any]:
    """Parse a group element from the survey flow.

    Args:
        flow_elem: Dictionary containing the group element data

    Returns:
        Dictionary containing the parsed group element
    """
    element = {
        "ElementType": "Group",
        "FlowID": flow_elem.get("FlowID", ""),
        "Description": flow_elem.get("Description", ""),
    }

    return element
