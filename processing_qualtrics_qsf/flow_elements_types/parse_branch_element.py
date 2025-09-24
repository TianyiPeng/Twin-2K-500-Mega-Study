from typing import Any, Dict


def parse_branch_element(flow_elem: Dict[str, Any]) -> Dict[str, Any]:
    """Parse a branch element from the survey flow.

    Args:
        flow_elem: Dictionary containing the branch element data

    Returns:
        Dictionary containing the parsed branch element
    """
    element = {
        "ElementType": "Branch",
        "FlowID": flow_elem.get("FlowID", ""),
    }
    if "BranchLogic" in flow_elem:
        element["BranchLogic"] = flow_elem["BranchLogic"]

    return element
