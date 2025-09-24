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

    if "Columns" in question:
        for i, column in enumerate(question["Columns"]):
            for key, value in embedded_data_dict.items():
                question["Columns"][i] = question["Columns"][i].replace(
                    f"${{e://Field/{key}}}", value
                )

    return question


def extract_responses_to_matrix_question(
    question: Dict, import_qid_to_response: Dict, embedded_data_dict: Dict
) -> Tuple[Dict, bool]:
    """Extract responses from the import_qid_to_response dictionary and add them to the matrix question."""
    question["Answers"] = {"SelectedByPosition": [], "SelectedText": []}
    if "RowsID" in question:
        RowsID = question["RowsID"]
    else:
        RowsID = list(str(i) for i in range(1, len(question["Rows"]) + 1))

    question = replace_embedded_data(question, embedded_data_dict)

    if "ColumnsID" in question:
        ColumnsID = question["ColumnsID"]
    else:
        ColumnsID = list(str(i) for i in range(1, len(question["Columns"]) + 1))

    has_response = False

    for i, row_id in enumerate(RowsID):
        row_lookup_id = f"{question['QuestionID']}_{row_id}"
        if row_lookup_id in import_qid_to_response:
            column_select_id = str(import_qid_to_response[row_lookup_id])
            for j, column_id in enumerate(ColumnsID):
                if int(float(column_id)) == int(float(column_select_id)):
                    question["Answers"]["SelectedByPosition"].append(j + 1)
                    question["Answers"]["SelectedText"].append(question["Columns"][j])
                    has_response = True
    return question, has_response
