from typing import Any, Dict

from .base import default_order, get_common_settings, strip_html


def extract_matrix_question_data(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Extract data specific to Matrix type questions"""
    question_data = {
        "Rows": [],
        "Columns": [],
        "RowsID": [],
        "ColumnsID": [],
        "Settings": get_common_settings(payload),
    }

    # Extract the statements (rows)
    if "Choices" in payload:
        choices = payload.get("Choices", {})
        choice_order = payload.get("ChoiceOrder", [])
        for choice_id in choice_order:
            if str(choice_id) in choices:
                choice_data = choices[str(choice_id)]
                if isinstance(choice_data, dict):
                    choice_text = choice_data.get("Display", "")
                    choice_text = strip_html(choice_text)
                    question_data["Rows"].append(choice_text)
                    question_data["RowsID"].append(str(choice_id))

    # Extract the answer scale (columns)
    if "Answers" in payload:
        answers = payload.get("Answers", {})
        answer_order = payload.get("AnswerOrder", [])

        if answer_order:
            for answer_id in answer_order:
                if str(answer_id) in answers:
                    answer_data = answers[str(answer_id)]
                    if isinstance(answer_data, dict):
                        answer_text = answer_data.get("Display", "")
                        answer_text = strip_html(answer_text)
                        question_data["Columns"].append(answer_text)
                        question_data["ColumnsID"].append(str(answer_id))

        if "RecodeValues" in payload:
            recode_values = {str(k): v for k, v in payload.get("RecodeValues", {}).items()}
            question_data["ColumnsID"] = [
                recode_values[cid] if cid in recode_values else cid
                for cid in question_data["ColumnsID"]
            ]

        if "Selector" in payload and payload["Selector"] == "Bipolar":
            # Warning: What if the matrix is transposed and need to deal with vertical polar
            # choice_text = "From one end:( more " + choice_text.split(':')[0] + " ) to the other end:( more " + choice_text.split(':')[1]+")"
            choice_text = question_data["Rows"][0]
            if len(question_data["Rows"]) > 1:
                raise ValueError("Bipolar matrix is not supported for multiple rows")
            one_end = choice_text.split(":")[0]
            another_end = choice_text.split(":")[1]
            question_data["Rows"][0] = (
                "From one end:( more " + one_end + " ) to the other end:( more " + another_end + ")"
            )
            question_data["Columns"][0] = question_data["Columns"][0] + " - " + one_end
            question_data["Columns"][-1] = question_data["Columns"][-1] + " - " + another_end

    if default_order(question_data["ColumnsID"]):
        question_data.pop("ColumnsID")  # if not present, the options are in default order

    if default_order(question_data["RowsID"]):
        question_data.pop("RowsID")  # if not present, the options are in default order

    return question_data
