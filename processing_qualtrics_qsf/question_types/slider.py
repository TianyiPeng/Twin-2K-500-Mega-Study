from typing import Any, Dict

from .base import default_order, get_common_settings, strip_html


def extract_slider_question_data(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Extract data specific to Slider questions"""
    question_data = {
        "Statements": [],
        "StatementsID": [],
        "Range": {
            "Min": payload.get("Configuration", {}).get("CSSliderMin", 0),
            "Max": payload.get("Configuration", {}).get("CSSliderMax", 100),
            "Ticks": payload.get("Configuration", {}).get("GridLines", 10),
        },
        "Settings": {
            "Selector": payload.get("Selector", ""),
            "ForceResponse": payload.get("Validation", {})
            .get("Settings", {})
            .get("ForceResponse", ""),
        },
    }

    # Some sliders have labeled choices
    if "Choices" in payload:
        choices = payload.get("Choices", {})
        choice_order = payload.get("ChoiceOrder", [])

        if choice_order:
            for choice_id in choice_order:
                if str(choice_id) in choices:
                    choice_info = choices[str(choice_id)]
                    if isinstance(choice_info, dict):
                        choice_text = choice_info.get("Display", "")
                        choice_text = strip_html(choice_text)
                        question_data["Statements"].append(choice_text)
                        question_data["StatementsID"].append(str(choice_id))

        if default_order(question_data["StatementsID"]):
            question_data.pop("StatementsID")  # if not present, the options are in default order

    return question_data


def extract_slider_question_data(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Extract data specific to Slider questions"""
    question_data = {
        "Statements": [],
        "StatementsID": [],
        "Range": {
            "Min": payload.get("Configuration", {}).get("CSSliderMin", 0),
            "Max": payload.get("Configuration", {}).get("CSSliderMax", 100),
            "Ticks": payload.get("Configuration", {}).get("GridLines", 10),
        },
        "Settings": get_common_settings(payload),
    }

    # Some sliders have labeled choices
    if "Choices" in payload:
        choices = payload.get("Choices", {})
        choice_order = payload.get("ChoiceOrder", [])

        if choice_order:
            for choice_id in choice_order:
                if str(choice_id) in choices:
                    choice_info = choices[str(choice_id)]
                    if isinstance(choice_info, dict):
                        choice_text = choice_info.get("Display", "")
                        choice_text = strip_html(choice_text)
                        question_data["Statements"].append(choice_text)
                        question_data["StatementsID"].append(str(choice_id))

        if default_order(question_data["StatementsID"]):
            question_data.pop("StatementsID")  # if not present, the options are in default order

    return question_data
