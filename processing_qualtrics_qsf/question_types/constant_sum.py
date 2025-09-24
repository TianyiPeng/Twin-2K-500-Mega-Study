from typing import Any, Dict

from .base import default_order, strip_html

"""
Example of a constant sum question:
{
      "SurveyID": "SV_dcHrU5epSdGBcLc",
      "Element": "SQ",
      "PrimaryAttribute": "QID1218539715",
      "SecondaryAttribute": "Based on the chapter you just read, what do you think is the likelihood that the text of the next...",
      "TertiaryAttribute": null,
      "Payload": {
        "QuestionText": "Based on the chapter you just read, what do you think is the likelihood that the text of the next chapter will be very negative or very positive?\nPlease assign a percentage to each. The total must sum to 100%.<br><br>If you think that there is a 50% chance that the next chapter is very positive and 50% chance that the next chapter is very negative, then you should assign 50% to very positive and 50% to very negative. If you are 100% sure that the next chapter is going to be neutral, then you should assign 100% to neutral.<br>",
        "DefaultChoices": false,
        "QuestionType": "CS",
        "Selector": "VRTL",
        "DataVisibility": {
          "Private": false,
          "Hidden": false
        },
        "Configuration": {
          "QuestionDescriptionOption": "UseText",
          "CSSliderMin": 0,
          "CSSliderMax": 100,
          "GridLines": 10,
          "NumDecimals": "0",
          "SliderStartPositions": {
            "1": 0.6104294478527608,
            "2": 0.5598159509202454,
            "3": 0.6104294478527608,
            "4": 0.6104294478527608,
            "5": 0.8803680981595092
          }
        },
        "QuestionDescription": "Based on the chapter you just read, what do you think is the likelihood that the text of the next...",
        "Choices": {
          "1": {
            "Display": "Very negative<br />\n<span style=\"font-size:11px;\">(e.g., torture, mourn)</span>"
          },
          "2": {
            "Display": "Somewhat negative"
          },
          "3": {
            "Display": "Neutral<br />\n<span style=\"font-size:11px;\">(e.g., vertical, beaver)</span>"
          },
          "4": {
            "Display": "Somewhat positive"
          },
          "5": {
            "Display": "Very positive<br />\n<span style=\"font-size:11px;\">(e.g., vacation, happiness)</span>"
          }
        },
        "ChoiceOrder": [
          1,
          2,
          3,
          4,
          5
        ],
        "Validation": {
          "Settings": {
            "EnforceRange": "OFF",
            "Type": "ChoicesTotal",
            "ForceResponse": "OFF",
            "ChoiceTotal": "100"
          }
        },
        "GradingData": [],
        "Language": [],
        "NextChoiceId": 6,
        "NextAnswerId": 1,
        "Labels": [],
        "ClarifyingSymbolType": "After",
        "SubSelector": "TX",
        "QuestionText_Unsafe": "Now try to think about the whole range of possible next chapters and how negative or positive the words are in these possible chapters. For each of the following buckets, please estimate what percentage of next chapters fall into each of the buckets based on the words used.<br><br>The Total must equal 100.<br>",
        "DataExportTag": "Achap1_val",
        "QuestionID": "QID1218539715",
        "ClarifyingSymbol": "%"
      }
}
"""


def extract_constant_sum_question_data(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Extract data specific to Constant Sum questions

    Constant Sum questions allow participants to allocate values (typically percentages)
    across multiple choices that sum to a specified total (usually 100).

    Args:
        payload: The question payload from the QSF file

    Returns:
        Dictionary containing structured constant sum question data
    """
    question_data = {
        "Rows": [],
        "RowsID": [],
        "Settings": {
            "Selector": payload.get("Selector", ""),
            "ForceResponse": payload.get("Validation", {})
            .get("Settings", {})
            .get("ForceResponse", ""),
            "Type": payload.get("Validation", {}).get("Settings", {}).get("Type", ""),
        },
    }

    # Extract total that choices must sum to
    validation_settings = payload.get("Validation", {}).get("Settings", {})
    if "ChoiceTotal" in validation_settings:
        question_data["Settings"]["ChoiceTotal"] = validation_settings["ChoiceTotal"]

    # Extract choices
    if "Choices" in payload:
        choices = payload.get("Choices", {})
        choice_order = payload.get("ChoiceOrder", [])

        # Process choices in the specified order
        for choice_id in choice_order:
            if str(choice_id) in choices:
                choice_info = choices[str(choice_id)]
                if isinstance(choice_info, dict):
                    choice_text = choice_info.get("Display", "")
                    choice_text = strip_html(choice_text)
                    question_data["Rows"].append(choice_text)
                    question_data["RowsID"].append(str(choice_id))

        # Remove ChoicesID if they are in default order (1, 2, 3, ...)
        if default_order(question_data["RowsID"]):
            question_data.pop("RowsID")

    return question_data
