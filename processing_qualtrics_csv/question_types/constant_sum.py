from typing import Dict, Tuple


def replace_embedded_data(question: Dict, embedded_data_dict: Dict) -> Dict:
    """Replace embedded data in the question."""
    if "QuestionText" in question:
        for key, value in embedded_data_dict.items():
            question["QuestionText"] = question["QuestionText"].replace(
                f"${{e://Field/{key}}}", value
            )

    if "Rows" in question:
        for i, row in enumerate(question["Rows"]):
            for key, value in embedded_data_dict.items():
                question["Rows"][i] = question["Rows"][i].replace(f"${{e://Field/{key}}}", value)

    return question


def extract_responses_to_cs_question(
    question: Dict, import_qid_to_response: Dict, embedded_data_dict: Dict
) -> Tuple[Dict, bool]:
    """Extract responses from the import_qid_to_response dictionary and add them to the constant sum question.

    Constant Sum questions allow participants to allocate values (typically percentages)
    across multiple choices that sum to a specified total (usually 100).
    """
    question["Answers"] = {
        "Values": [],
    }

    question = replace_embedded_data(question, embedded_data_dict)

    # Get row IDs - either from RowsID if available, or generate sequential IDs
    if "RowsID" in question:
        RowsID = question["RowsID"]
    else:
        # Generate sequential row IDs starting from 1
        RowsID = list(str(i) for i in range(1, len(question["Rows"]) + 1))

    has_response = False

    # Extract responses for each choice in the constant sum question
    for i, row_id in enumerate(RowsID):
        row_lookup_id = f"{question['QuestionID']}_{row_id}"
        if row_lookup_id in import_qid_to_response:
            # Store the allocated value (e.g., percentage)
            allocated_value = import_qid_to_response[row_lookup_id]
            question["Answers"]["Values"].append(allocated_value)

            has_response = True
        else:
            # If no response for this choice, add None/empty values to maintain alignment
            question["Answers"]["Values"].append(None)

    return question, has_response
