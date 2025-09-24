import copy
import random
from typing import Any, Dict, List

"""
Example of a block that enables looping and merging
{
        "ElementType": "Block",
        "BlockName": "Min dist main",
        "BlockType": "Standard",
        "BlockID": "BL_dcJcgQZuh7Ib2v4",
        "Questions": [
            {
                "QuestionID": "QID34",
                "QuestionText": "Round #${lm://CurrentLoopNumber} of ${lm://TotalLoops} Find a way to connect these two words: ${lm://Field/1} and ${lm://Field/2}. Each word in the sequence below should be as closely related as possible to the word before it. The first word in the sequence is already set to ${lm://Field/1} . The last word is already set to ${lm://Field/2} . That is, you only need to add 3 words that connect the first word to the last word.",
                "QuestionType": "TE",
                "QuestionName": "Q30",
                "Rows": [
                    "Word 1:",
                    "Word 2:",
                    "Word 3:",
                    "Word 4:",
                    "Word 5:"
                ],
                "Settings": {
                    "Selector": "FORM",
                    "ForceResponse": "ON",
                    "ContentType": ""
                },
                "DefaultChoices": [
                    "${lm://Field/1}",
                    null,
                    null,
                    null,
                    "${lm://Field/2}"
                ]
            }
        ],
        "Options": {
            "BlockLocking": "false",
            "RandomizeQuestions": "false",
            "Looping": "Static",
            "LoopingOptions": {
                "Static": {
                    "1": {
                        "1": "ETERNITY",
                        "2": "CURIOSITY"
                    },
                    "2": {
                        "1": "ELEPHANT",
                        "2": "GALAXY"
                    },
                    "4": {
                        "1": "PERSEVERANCE",
                        "2": "ELOQUENCE"
                    },
                    "5": {
                        "1": "EUPHORIA",
                        "2": "TULIP"
                    },
                    "7": {
                        "1": "TANGERINE",
                        "2": "PENGUIN"
                    }
                },
                "Randomization": "None"
            },
            "BlockVisibility": "Expanded"
        }
    }
"""


def loop_and_merge_questions(element: Dict) -> Dict:
    """Process questions with looping options and merge them appropriately.

    Args:
        element: Dictionary containing the block element with looping options

    Returns:
        new element with loop questions merged
    """
    if "Options" not in element or "Looping" not in element["Options"]:
        return element

    looping_options = element["Options"]["Looping"]

    if looping_options == "None":
        return element

    if looping_options != "Static" and looping_options != "Question":
        raise ValueError(f"Looping options {looping_options} not supported")

    questions = element.get("Questions", [])
    merged_questions = []

    # Get the static loop data
    static_loops = element["Options"]["LoopingOptions"]["Static"]
    total_loops = len(static_loops)

    ## extract the looped fields
    static_loops = [(k, v) for k, v in sorted(static_loops.items(), key=lambda item: int(item[0]))]

    if "Randomization" in element["Options"]["LoopingOptions"]:
        if (element["Options"]["LoopingOptions"]["Randomization"] == "All") or (
            element["Options"]["LoopingOptions"]["Randomization"] == "Subset"
        ):
            ## random permutation of the static loops
            random.shuffle(static_loops)
        elif element["Options"]["LoopingOptions"]["Randomization"] == "None":  ## no randomization
            pass
        else:
            raise ValueError(
                f"Looping Randomization {element['Options']['LoopingOptions']['Randomization']} not supported"
            )

    element["Options"]["LoopingOptions"]["RealizedLoopingOrder"] = static_loops

    # Process each loop iteration
    for loop_num, (loop_label, loop_data) in enumerate(static_loops):
        for question in questions:
            # Create a deep copy of the question to avoid modifying the original
            question_copy = copy.deepcopy(question)

            # Update question ID with loop number
            if "QuestionID" in question_copy:
                question_copy["QuestionID"] = f"{loop_label}_{question_copy['QuestionID']}"

            # Replace loop variables in question text
            if "QuestionText" in question_copy:
                text = question_copy["QuestionText"]
                # Replace field values
                for field_num, field_value in loop_data.items():
                    text = text.replace(f"${{lm://Field/{field_num}}}", field_value)

                question_copy["QuestionText"] = text

            if "Options" in question_copy:
                for i, option in enumerate(question_copy["Options"]):
                    if isinstance(option, str):
                        for field_num, field_value in loop_data.items():
                            option = option.replace(f"${{lm://Field/{field_num}}}", field_value)
                        question_copy["Options"][i] = option

            # Replace field values in default choices if they exist
            if "DefaultChoices" in question_copy:
                for i, choice in enumerate(question_copy["DefaultChoices"]):
                    if isinstance(choice, str):
                        for field_num, field_value in loop_data.items():
                            choice = choice.replace(f"${{lm://Field/{field_num}}}", field_value)
                        question_copy["DefaultChoices"][i] = choice

            merged_questions.append(question_copy)

    element["Questions"] = merged_questions
    return element


def fillin_loop_number(element: Dict) -> str:
    """Fill in the loop number in the question text.

    Args:
        element: The element to adjust

    Returns:
        The adjusted element
    """
    if "Options" not in element or "Looping" not in element["Options"]:
        return element

    looping_options = element["Options"]["Looping"]

    if looping_options == "None":
        return element

    if looping_options != "Static" and looping_options != "Question":
        raise ValueError(f"Looping options {looping_options} not supported")

    questions = element.get("Questions", [])
    merged_questions = []
    total_loops = len(questions)

    for loop_num, question in enumerate(questions):
        # Create a deep copy of the question to avoid modifying the original
        question_copy = copy.deepcopy(question)

        if "QuestionText" in question_copy:
            text = question_copy["QuestionText"]
            text = text.replace("${lm://CurrentLoopNumber}", str(loop_num + 1))
            text = text.replace("${lm://TotalLoops}", str(total_loops))

            question_copy["QuestionText"] = text

        merged_questions.append(question_copy)

    element["Questions"] = merged_questions
    return element
