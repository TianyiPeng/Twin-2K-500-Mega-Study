from typing import Any, Dict

from .base import default_order, get_common_settings, strip_html


def extract_choice_question_data(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Extract data specific to Multiple Choice questions"""
    question_data = {"Options": [], "OptionsID": [], "Settings": get_common_settings(payload)}

    if "Choices" in payload:
        choices = payload.get("Choices", {})
        choice_order = payload.get("ChoiceOrder", [])

        for choice_id in choice_order:
            if str(choice_id) in choices:
                choice_info = choices[str(choice_id)]
                if isinstance(choice_info, dict):
                    choice_text = choice_info.get("Display", "")
                    choice_text = strip_html(choice_text)
                    question_data["Options"].append(choice_text)
                    question_data["OptionsID"].append(str(choice_id))

        if "RecodeValues" in payload:
            recode_values = payload.get("RecodeValues", {})
            for key, value in recode_values.items():
                recode_values[str(key)] = value
            for i in range(len(question_data["OptionsID"])):
                if str(question_data["OptionsID"][i]) in recode_values:
                    question_data["OptionsID"][i] = recode_values[question_data["OptionsID"][i]]

        if default_order(question_data["OptionsID"]):
            question_data.pop("OptionsID")  # if not present, the options are in default order
    return question_data
