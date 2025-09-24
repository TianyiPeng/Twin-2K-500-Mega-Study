from typing import Dict, Tuple


def replace_embedded_data(question: Dict, embedded_data_dict: Dict) -> Dict:
    """Replace embedded data in the question."""
    if "QuestionText" in question:
        for key, value in embedded_data_dict.items():
            question["QuestionText"] = question["QuestionText"].replace(
                f"${{e://Field/{key}}}", value
            )

    if "Statements" in question:
        for i, statement in enumerate(question["Statements"]):
            for key, value in embedded_data_dict.items():
                question["Statements"][i] = question["Statements"][i].replace(
                    f"${{e://Field/{key}}}", value
                )

    return question


def extract_responses_to_slider_question(
    question: Dict, import_qid_to_response: Dict, embedded_data_dict: Dict
) -> Tuple[Dict, bool]:
    """Extract responses from the import_qid_to_response dictionary and add them to the slider question."""
    question["Answers"] = {
        "Values": [],
    }

    question = replace_embedded_data(question, embedded_data_dict)

    if "StatementsID" in question:
        StatementsID = question["StatementsID"]
    else:
        StatementsID = list(str(i) for i in range(1, len(question["Statements"]) + 1))

    has_response = False
    for i, statement_id in enumerate(StatementsID):
        statement_lookup_id = f"{question['QuestionID']}_{statement_id}"
        if statement_lookup_id in import_qid_to_response:
            question["Answers"]["Values"].append(import_qid_to_response[statement_lookup_id])
            has_response = True
    return question, has_response
