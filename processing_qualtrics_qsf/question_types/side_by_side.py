from typing import Any, Dict

from .base import strip_html


def extract_sbs_question_data(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Extract data for Side By Side questions"""
    question_data = {"sub_questions": []}

    if "Questions" in payload:
        for sub_q_id, sub_q_data in payload.get("Questions", {}).items():
            if not isinstance(sub_q_data, dict):
                continue

            sub_question = {
                "id": sub_q_id,
                "text": strip_html(sub_q_data.get("QuestionText", "")),
                "question_type": sub_q_data.get("QuestionType", ""),
                "Options": [],
                "OptionsID": [],
            }

            # Extract choices for the sub-question
            if "Choices" in sub_q_data:
                choices = sub_q_data.get("Choices", {})
                choice_order = sub_q_data.get("ChoiceOrder", [])

                if choice_order:
                    for choice_id in choice_order:
                        if str(choice_id) in choices:
                            choice_info = choices[str(choice_id)]
                            if isinstance(choice_info, dict):
                                choice_text = choice_info.get("Display", "")
                                choice_text = strip_html(choice_text)
                                sub_question["choices"].append(
                                    {"id": choice_id, "text": choice_text}
                                )

            question_data["sub_questions"].append(sub_question)

    return question_data
