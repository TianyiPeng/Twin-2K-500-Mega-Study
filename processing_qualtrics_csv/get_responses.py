import copy
import html
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
import yaml
from bs4 import BeautifulSoup
from tqdm import tqdm

try:
    # Try relative imports first (when imported as a module)
    from .question_types import (
        extract_responses_to_cs_question,
        extract_responses_to_matrix_question,
        extract_responses_to_mc_question,
        extract_responses_to_slider_question,
        extract_responses_to_te_question,
    )
    from .question_types.looping import fillin_loop_number, loop_and_merge_questions
    from .util.logic_eval import eval_display_logic
except ImportError:
    # Fall back to absolute imports (when run directly as a script)
    from question_types import (
        extract_responses_to_cs_question,
        extract_responses_to_matrix_question,
        extract_responses_to_mc_question,
        extract_responses_to_slider_question,
        extract_responses_to_te_question,
    )
    from question_types.looping import fillin_loop_number, loop_and_merge_questions
    from util.logic_eval import eval_display_logic

PID_COLUMN = "TWIN_ID"


# Function to extract ImportId from JSON string in CSV cells
def extract_json_value(json_string: str, key: str) -> Optional[str]:
    """Extract a value from a JSON string."""
    try:
        # Handle cases where the string might be a raw JSON object
        if isinstance(json_string, str) and json_string.strip().startswith("{"):
            data = json.loads(json_string)
            return data.get(key)
    except (json.JSONDecodeError, TypeError, AttributeError):
        pass
    return None


def load_template(json_path: str) -> Dict:
    """Load the JSON template file."""
    try:
        with open(json_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: Template file not found at {json_path}")
        return {}
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON template: {e}")
        return {}
    except Exception as e:
        print(f"Error loading template: {e}")
        return {}


def load_csv(csv_path: str) -> pd.DataFrame:
    """Load the CSV file."""

    def extract_integer(val):
        try:
            f = float(val)
            return f.is_integer()
        except (ValueError, TypeError):
            return False

    try:
        print(f"Loading CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        if "TWIN_ID" in df.columns:
            try:
                df["TWIN_ID"] = df["TWIN_ID"].apply(
                    lambda x: str(int(x)) if (extract_integer(x)) else x
                )
            except:
                raise Exception("TWIN_ID column conversion failed")

        print(f"CSV loaded with {len(df)} rows and {len(df.columns)} columns")
        return df
    except Exception as e:
        print(f"Error loading CSV: {e!s}")
        return pd.DataFrame()


def get_import_ids_from_csv(df: pd.DataFrame) -> Dict[str, str]:
    """Extract ImportIds from the first data row of the CSV.

    Args:
        df: DataFrame containing survey data

    Returns:
        Dictionary mapping column names to ImportIds
    """
    if len(df) < 2:
        print("CSV has insufficient rows to extract ImportIds")
        return {}

    # Get the first data row where the ImportIds are stored
    first_data_row = df.iloc[
        1
    ]  # Usually, the first row contains column names, second row has ImportIds

    import_id_map = {}
    for col_name, value in first_data_row.items():
        if pd.isna(value):
            continue

        import_id = extract_json_value(value, "ImportId")
        choice_id = extract_json_value(value, "choiceId")
        if choice_id:
            import_id_map[col_name] = f"{import_id}_{choice_id}"
        elif import_id:
            import_id_map[col_name] = import_id

    print(f"Extracted {len(import_id_map)} ImportIds from CSV")
    return import_id_map


def extract_responses_to_question(
    question: Dict, import_qid_to_response: Dict, embedded_data_dict: Dict
) -> Tuple[Dict, bool]:
    """Extract responses from the import_qid_to_response dictionary and add them to the question."""
    if "QuestionID" not in question:
        return question, False

    if "DisplayLogic" in question:
        if not eval_display_logic(question["DisplayLogic"], embedded_data_dict):
            return question, False

    if question["QuestionType"] == "Matrix":
        return extract_responses_to_matrix_question(
            question, import_qid_to_response, embedded_data_dict
        )

    if question["QuestionType"] == "Slider":
        return extract_responses_to_slider_question(
            question, import_qid_to_response, embedded_data_dict
        )

    if question["QuestionType"] == "MC":
        return extract_responses_to_mc_question(
            question, import_qid_to_response, embedded_data_dict
        )

    if question["QuestionType"] == "TE":
        return extract_responses_to_te_question(
            question, import_qid_to_response, embedded_data_dict
        )

    if question["QuestionType"] == "CS":
        return extract_responses_to_cs_question(
            question, import_qid_to_response, embedded_data_dict
        )

    if question["QuestionType"] == "DB":
        return question, True

    return question, False


### replace ${e://Field/StoryA_Chapter1} -> embedded_data_dict['StoryA_Chapter1'] in question text
def replace_embedded_data(question: Dict, embedded_data_dict: Dict) -> Dict:
    """Replace embedded data in the question."""
    if "QuestionText" in question:
        for key, value in embedded_data_dict.items():
            question["QuestionText"] = question["QuestionText"].replace(
                f"${{e://Field/{key}}}", value
            )

    return question


def generate_randomizer_list(element: Dict, import_qid_to_response: Dict) -> List[Dict]:
    """Generate a list of elements for a randomizer."""
    flow_ID = element["FlowID"]
    k = element["SelectElements"]
    if isinstance(k, str):
        k = int(k)
    randomizer_list = [None for _ in range(k)]
    for sub_element in element["Elements"]:
        if sub_element["ElementType"] == "Block":
            subflow_ID = sub_element["BlockName"].replace(" ", "")
        else:
            subflow_ID = sub_element["FlowID"]
        lookup_ID = f"{flow_ID}_DO_{subflow_ID}"
        if lookup_ID in import_qid_to_response:
            randomizer_list[int(import_qid_to_response[lookup_ID]) - 1] = sub_element
    return_randomizer_list = []
    for i in range(len(randomizer_list)):
        if randomizer_list[i] is None:
            print(f"Randomizer {element['FlowID']} list is not full. Missing element {i + 1}")
        else:
            return_randomizer_list.append(randomizer_list[i])
    return return_randomizer_list


def process_branch_element(
    element: Dict, import_qid_to_response: Dict, embedded_data_dict: Dict = None
) -> List[Dict]:
    """Process a branch element."""
    if "BranchLogic" in element:
        if embedded_data_dict is None:
            embedded_data_dict = {}
        if eval_display_logic(element["BranchLogic"], embedded_data_dict, import_qid_to_response):
            return element["Elements"]
        else:
            return []
    return element["Elements"]


def process_randomizer_element(element: Dict, import_qid_to_response: Dict) -> List[Dict]:
    """Process a randomizer element."""
    return generate_randomizer_list(element, import_qid_to_response)


"""
options = {
    "BlockLocking": False,
    "RandomizeQuestions": "Advanced",
    "BlockVisibility": "Expanded",
    "Randomization": {
        "Advanced": {
            "FixedOrder": [
                "QID17",
                "QID18",
                "{~Randomized~}",
                "{~Randomized~}"
            ],
            "RandomizeAll": [
                "QID19",
                "QID20"
            ],
            "RandomSubSet": [],
            "Undisplayed": [],
            "TotalRandSubset": 0,
            "QuestionsPerPage": 0,
        },
        "EvenPresentation": False
    }
}
"""


def randomize_questions(element: Dict, import_qid_to_response: Dict) -> List[Dict]:
    """Randomize the questions in the block element."""
    options = element["Options"]
    new_questions = [None for _ in range(len(element["Questions"]))]
    if options["RandomizeQuestions"] == "Advanced":
        block_id = element["BlockID"]
        for question in element["Questions"]:
            quest_name = question["QuestionName"]
            lookup_ID = f"{block_id}_DO_{quest_name}"
            if lookup_ID in import_qid_to_response:
                if import_qid_to_response[lookup_ID] != None:
                    position = int(import_qid_to_response[lookup_ID])
                    new_questions[position - 1] = question

        new_questions = [question for question in new_questions if question is not None]
        return new_questions

    else:
        return element["Questions"]


def process_block_element(
    element: Dict,
    import_qid_to_response: Dict,
    embedded_data_dict: Dict,
    keep_questions_list: List[str] = [],
) -> List[Dict]:
    """Process a block element."""
    if "Options" in element and "Looping" in element["Options"]:
        element = loop_and_merge_questions(element)
        # element.pop('Options') ## remove looping options
    if "Options" in element and "Randomization" in element["Options"]:
        element["Questions"] = randomize_questions(element, import_qid_to_response)
        # element.pop("Options")  ## remove randomization options

    new_questions = []
    for question in element["Questions"]:
        question_with_response, success = extract_responses_to_question(
            question, import_qid_to_response, embedded_data_dict
        )
        if success or question_with_response["QuestionID"] in keep_questions_list:
            new_questions.append(question_with_response)
    element["Questions"] = new_questions

    for question in element["Questions"]:
        question = replace_embedded_data(question, embedded_data_dict)

    if "Options" in element and "Looping" in element["Options"]:
        element = fillin_loop_number(element)

    if "Options" in element and (
        "Randomization" in element["Options"] or "Looping" in element["Options"]
    ):
        element.pop("Options")

    return element


def load_pre_defined_embedded_data(
    import_qid_to_response: Dict, load_embedded_data: List[Dict]
) -> Dict:
    """Load embedded data from the CSV file."""
    embedded_data_dict = {}
    if load_embedded_data is not None:
        for embedded_data in load_embedded_data:
            if len(embedded_data) > 1:
                raise Exception(f"Embedded data {embedded_data} has more than one key")
            lookup_name = list(embedded_data.keys())[0]
            if lookup_name not in import_qid_to_response:
                # print(f"Embedded data {lookup_name} not found in import_qid_to_response")
                continue

            embedded_name = (
                lookup_name
                if "rename" not in embedded_data[lookup_name]
                else embedded_data[lookup_name]["rename"]
            )
            embedded_data_dict[embedded_name] = import_qid_to_response[lookup_name]

    return embedded_data_dict


def process_survey(
    template_path: str,
    csv_path: str,
    output_dir: str,
    max_participants: int = None,
    load_embedded_data: List[Dict] = None,
    keep_questions_list: List[str] = [],
) -> None:
    """Main processing function.

    Args:
        template_path: Path to the JSON template file
        csv_path: Path to the CSV data file
        output_dir: Directory to save output files
        max_participants: Maximum number of participants to process (None for all)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load template
    template = load_template(template_path)
    if not template or "Elements" not in template:
        print("Error: Template empty/invalid. Aborting.")
        return

    # Load CSV data
    df = load_csv(csv_path)
    if df.empty:
        print("Error: CSV data is empty or could not be loaded. Aborting.")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    ### create a mapping from column name to ImportId
    import_id_map = get_import_ids_from_csv(df)

    ### enumerate each row in the csv (each row is a participant)

    df = df.iloc[2:, :].reset_index(drop=True)

    pid_processed = {}
    df.sort_values(by=PID_COLUMN, inplace=True)
    total_length = min(len(df), max_participants) if max_participants else len(df)
    for i, row in tqdm(df.iterrows(), total=total_length):
        ### max_participants limit
        if max_participants and i >= max_participants:
            break

        # Find participant row
        pid = row[PID_COLUMN]

        if pid in pid_processed:
            print(f"Duplicate PID Detected: {pid}")
            continue

        pid_processed[pid] = True

        # copy template and add responses for this participant
        participant_responses = copy.deepcopy(template)  ## deep copy

        import_qid_to_response = {}
        for col, value in row.items():
            if pd.isna(value) or value == "":
                continue

            # Get the ImportId for this column
            import_id = import_id_map.get(col)
            if import_id:
                import_qid_to_response[import_id] = value

        # add responses for this participant
        embedded_data_dict = load_pre_defined_embedded_data(
            import_qid_to_response, load_embedded_data
        )

        def add_responses(element, import_qid_to_response):
            if "ElementType" not in element:
                return element, False

            if element["ElementType"] == "Block":
                element = process_block_element(
                    element, import_qid_to_response, embedded_data_dict, keep_questions_list
                )
                return element, len(element["Questions"]) > 0

            if element["ElementType"] == "EmbeddedData":
                for embedded_data in element["EmbeddedData"]:
                    embedded_data_dict[embedded_data["Field"]] = embedded_data["Value"]

            new_elements = []
            if "Elements" in element:
                element_list = element["Elements"]
                if element["ElementType"] == "Branch":
                    element_list = process_branch_element(
                        element, import_qid_to_response, embedded_data_dict
                    )
                if element["ElementType"] == "Randomizer":
                    element_list = process_randomizer_element(element, import_qid_to_response)

                for sub_element in element_list:
                    new_sub_element, success = add_responses(sub_element, import_qid_to_response)
                    if success:
                        new_elements.append(new_sub_element)
                element["Elements"] = new_elements
            return element, len(new_elements) > 0

        new_elements = []
        for element in participant_responses["Elements"]:
            return_element, success = add_responses(element, import_qid_to_response)

            if success:
                new_elements.append(return_element)
        participant_responses["Elements"] = new_elements
        ### add embedded data
        participant_responses["EmbeddedData"] = embedded_data_dict

        # Save template-based output
        template_file_path = os.path.join(output_dir, f"pid_{pid}_response.json")
        try:
            with open(template_file_path, "w", encoding="utf-8") as f:
                json.dump(participant_responses, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving template data for participant {pid}: {e}")

    print(f"Processing complete. Successfully processed {len(pid_processed)} participants.")
    print(f"Output saved in {output_dir}")


def read_config(config_path: str) -> Dict[str, Any]:
    """Read and parse the YAML configuration file"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main function"""
    # Parse command line arguments
    if len(sys.argv) < 3:
        print("Usage: python get_responses.py --config <config_file_path>")
        sys.exit(1)
    elif sys.argv[1] == "--config":
        config_file_path = sys.argv[2]
    else:
        print("Usage: python get_responses.py --config <config_file_path>")
        sys.exit(1)

    # Read configuration from YAML
    config = read_config(config_file_path)
    csv_config = config["csv_to_fill_answers"]
    template_path = csv_config["input_survey_template"]
    csv_path = csv_config["input_csv_file"]
    output_dir = csv_config["output_dir"]
    load_embedded_data = csv_config.get("load_embedded_data", None)
    keep_questions_list = csv_config.get("keep_questions_when_answer_missing", [])
    max_participants = csv_config.get("limit", None)
    if max_participants == -1:
        max_participants = None

    # Run the main processing function
    process_survey(
        template_path,
        csv_path,
        output_dir,
        max_participants,
        load_embedded_data,
        keep_questions_list,
    )


if __name__ == "__main__":
    main()
