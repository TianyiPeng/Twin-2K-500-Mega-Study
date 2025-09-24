from typing import Dict, Tuple


def replace_embedded_data(question: Dict, embedded_data_dict: Dict) -> Dict:
    """Replace embedded data in the question."""
    if "QuestionText" in question:
        for key, value in embedded_data_dict.items():
            question["QuestionText"] = question["QuestionText"].replace(
                f"${{e://Field/{key}}}", value
            )

    if "Options" in question:
        for i, option in enumerate(question["Options"]):
            for key, value in embedded_data_dict.items():
                question["Options"][i] = question["Options"][i].replace(
                    f"${{e://Field/{key}}}", value
                )

    return question


def extract_responses_to_mc_question(
    question: Dict, import_qid_to_response: Dict, embedded_data_dict: Dict
) -> Tuple[Dict, bool]:
    """Extract responses from the import_qid_to_response dictionary and add them to the multiple choice question."""
    question["Answers"] = {}
    if "OptionsID" in question:
        OptionsID = question["OptionsID"]
    else:
        OptionsID = list(str(i) for i in range(1, len(question["Options"]) + 1))

    question = replace_embedded_data(question, embedded_data_dict)

    if (
        question["Settings"]["Selector"] == "SAVR"
        or question["Settings"]["Selector"] == "SAHR"
        or question["Settings"]["Selector"] == "DL"
    ):
        ## SAVR: Single Answer Vertical Response
        ## SAHR: Single Answer Horizontal Response
        ## DL: Dropdown
        question_id = question["QuestionID"]
        has_response = False
        if question_id in import_qid_to_response:
            selected_ID = str(import_qid_to_response[question_id])  ## identify the selected option
            for i, option_id in enumerate(OptionsID):
                if int(float(option_id)) == int(float(selected_ID)):
                    question["Answers"]["SelectedByPosition"] = i + 1
                    question["Answers"]["SelectedText"] = question["Options"][i]
                    has_response = True
        return question, has_response

    if question["Settings"]["Selector"] == "MAVR" or question["Settings"]["Selector"] == "MAHR":
        question_id = question["QuestionID"]
        question["Answers"] = {"SelectedByPosition": [], "SelectedText": []}
        has_response = False
        for i, option_id in enumerate(OptionsID):
            question_sub_id = f"{question_id}_{option_id}"
            if question_sub_id in import_qid_to_response:
                question["Answers"]["SelectedByPosition"].append(i + 1)
                question["Answers"]["SelectedText"].append(question["Options"][i])
                has_response = True
        return question, has_response

    raise ValueError(
        f"Selector {question['Settings']['Selector']} not supported for multiple choice question {question['QuestionID']}"
    )
