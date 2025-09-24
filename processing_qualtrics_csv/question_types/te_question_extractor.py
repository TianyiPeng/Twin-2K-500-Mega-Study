from typing import Dict, Tuple


def replace_embedded_data(question: Dict, embedded_data_dict: Dict) -> Dict:
    """Replace embedded data in the question."""
    if "QuestionText" in question:
        for key, value in embedded_data_dict.items():
            question["QuestionText"] = question["QuestionText"].replace(
                f"${{e://Field/{key}}}", value
            )
    return question


def extract_responses_to_te_question(
    question: Dict, import_qid_to_response: Dict, embedded_data_dict: Dict
) -> Tuple[Dict, bool]:
    """Extract responses from the import_qid_to_response dictionary and add them to the text entry question."""
    question = replace_embedded_data(question, embedded_data_dict)
    question_id = question["QuestionID"]
    selector = question.get("Settings", {}).get(
        "Selector", "SL"
    )  # Default to SL if Settings or Selector not found
    if "Settings" not in question:
        raise ValueError(f"Settings not found for question: {question_id}")
    if "Selector" not in question["Settings"]:
        raise ValueError(f"Selector not found for question: {question_id}")
    if selector == "SL" or selector == "ML" or selector == "ESTB":
        question["Answers"] = {
            "Text": "",
        }
        question_lookup_id = f"{question_id}_TEXT"
        has_response = False
        if question_lookup_id in import_qid_to_response:
            question["Answers"]["Text"] = import_qid_to_response[question_lookup_id]
            has_response = True
        return question, has_response
    elif selector == "FORM":
        question["Answers"] = {
            "Text": [],
        }
        has_response = False
        if "RowsID" in question:
            RowsID = question["RowsID"]
        else:
            RowsID = list(str(i) for i in range(1, len(question["Rows"]) + 1))
        for i, row_id in enumerate(RowsID):
            row_lookup_id = f"{question_id}_{row_id}"
            if row_lookup_id in import_qid_to_response:
                question["Answers"]["Text"].append(import_qid_to_response[row_lookup_id])
                has_response = True
            else:
                question["Answers"]["Text"].append(None)
        return question, has_response
    raise ValueError(f"Unknown selector for text question: {question_id} {selector}")
