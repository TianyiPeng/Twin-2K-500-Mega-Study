from typing import Any, Dict


def parse_embedded_data_element(flow_elem: Dict[str, Any]) -> Dict[str, Any]:
    """Parse an embedded data element from the survey flow.

    Args:
        flow_elem: Dictionary containing the embedded data element data

    Returns:
        Dictionary containing the parsed embedded data element
    """
    element = {
        "ElementType": "EmbeddedData",
        "FlowID": flow_elem.get("FlowID", ""),
        "EmbeddedData": [],
    }

    if flow_elem.get("EmbeddedData"):
        for data in flow_elem.get("EmbeddedData"):
            if data.get("Type") == "Recipient":
                continue
            element["EmbeddedData"].append(
                {"Field": data.get("Field", ""), "Value": data.get("Value", "")}
            )

    return element
