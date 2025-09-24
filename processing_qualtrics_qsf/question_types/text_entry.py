from typing import Any, Dict

from .base import default_order, strip_html


def extract_text_question_data(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Extract data specific to Text Entry questions"""
    selector = payload.get("Selector", "")
    if selector == "SL" or selector == "ML" or selector == "ESTB":
        ## SL: Single Line Text Entry
        ## ML: Multi Line Text Entry
        ## ESTB: Essay Text Box
        question_data = {
            "Settings": {
                "Selector": selector,
                "ForceResponse": payload.get("Validation", {})
                .get("Settings", {})
                .get("ForceResponse", ""),
                "ContentType": payload.get("Validation", {})
                .get("Settings", {})
                .get("ContentType", ""),
            }
        }
    elif selector == "FORM":
        question_data = {
            "Rows": [],
            "RowsID": [],
            "Settings": {
                "Selector": selector,
                "ForceResponse": payload.get("Validation", {})
                .get("Settings", {})
                .get("ForceResponse", ""),
                "ContentType": payload.get("Validation", {})
                .get("Settings", {})
                .get("ContentType", ""),
            },
        }
        if "Choices" in payload:
            choices = payload.get("Choices", {})
            choice_order = payload.get("ChoiceOrder", [])
            for choice_id in choice_order:
                if str(choice_id) in choices:
                    choice_info = choices[str(choice_id)]
                    if isinstance(choice_info, dict):
                        choice_text = choice_info.get("Display", "")
                        choice_text = strip_html(choice_text)
                        question_data["Rows"].append(choice_text)
                        question_data["RowsID"].append(str(choice_id))

            if default_order(question_data["RowsID"]):
                question_data.pop("RowsID")  # if not present, the options are in default order

            if "DefaultChoices" in payload and isinstance(payload.get("DefaultChoices"), dict):
                question_data["DefaultChoices"] = []
                default_choices = payload.get("DefaultChoices", [])
                for choice_id in choice_order:
                    if str(choice_id) in choices:
                        question_data["DefaultChoices"].append(None)
                        if str(choice_id) in default_choices:
                            question_data["DefaultChoices"][-1] = default_choices[str(choice_id)][
                                "TEXT"
                            ]["Text"]
                if all(item is None for item in question_data["DefaultChoices"]):
                    question_data.pop("DefaultChoices")
    else:
        raise ValueError(f"Unknown selector for text question: {selector}")
    return question_data
