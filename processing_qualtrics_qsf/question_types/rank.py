from typing import Any, Dict

from .base import get_common_settings, strip_html


def extract_rank_question_data(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Extract data specific to Ranking questions"""
    question_data = {"Options": [], "OptionsID": [], "Settings": get_common_settings(payload)}

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
                        question_data["Options"].append(choice_text)
                        question_data["OptionsID"].append(choice_id)
    return question_data
