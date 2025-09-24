from typing import Any, Dict


def eval_display_logic(
    display_logic: dict, embedded_data_dict: Dict, response_data: Dict = None
) -> bool:
    """Evaluate the display logic for Qualtrics QSF files.

    Args:
        display_logic: The display logic dictionary from QSF (could be DisplayLogic or BranchLogic)
        embedded_data_dict: Dictionary of embedded data values
        response_data: Dictionary of question responses (optional)

    Returns:
        bool: True if logic conditions are met, False otherwise
    """
    if not display_logic:
        return True

    # Handle different QSF logic structures:
    # 1. DisplayLogic: nested structure with numbered groups
    # 2. BranchLogic: array-like structure with Type at top level

    return _evaluate_logic_group(display_logic, embedded_data_dict, response_data)


def _evaluate_logic_group(
    logic_group: dict, embedded_data_dict: Dict, response_data: Dict = None
) -> bool:
    """Evaluate a group of logic conditions following QSF structure.
    Handles both DisplayLogic and BranchLogic formats.
    """
    results = []

    # Check if this is a BranchLogic structure (has Type at root and numbered conditions)
    logic_type = logic_group.get("Type", "If")  # Default to 'If' (AND logic)

    # Process numbered condition groups (0, 1, 2, etc.)
    for key, value in logic_group.items():
        if key in ["Type", "Conjunction"]:  # Skip meta fields
            continue

        # Skip non-numeric keys that aren't logic-related
        if not key.isdigit():
            continue

        if isinstance(value, dict):
            # Check if this is a direct condition or a nested group
            if "LogicType" in value and "Operator" in value:
                # This is a direct condition (like in BranchLogic)
                result = _evaluate_condition(value, embedded_data_dict, response_data)
                results.append(result)
            else:
                # This is a nested group (like in DisplayLogic) - process sub-conditions
                group_results = []
                group_conjunction = value.get(
                    "Conjunction", "And"
                )  # Check for conjunction at group level
                detected_conjunction = None  # To detect conjunction from conditions

                for sub_key, sub_value in value.items():
                    if sub_key in ["Conjunction", "Type"]:  # Skip meta fields
                        continue

                    if sub_key.isdigit() and isinstance(sub_value, dict):
                        if "LogicType" in sub_value:
                            result = _evaluate_condition(
                                sub_value, embedded_data_dict, response_data
                            )

                            # Check for conjunction within the condition (handle typo too)
                            condition_conjunction = sub_value.get(
                                "Conjunction", sub_value.get("Conjuction")
                            )  # Handle typo

                            # If any condition specifies a conjunction, use it for the group
                            if condition_conjunction:
                                detected_conjunction = condition_conjunction

                            group_results.append(result)
                        else:
                            # Deeper nesting
                            result = _evaluate_logic_group(
                                sub_value, embedded_data_dict, response_data
                            )
                            group_results.append(result)

                # Apply conjunction within the group - prefer detected conjunction from conditions
                if group_results:
                    final_conjunction = (
                        detected_conjunction if detected_conjunction else group_conjunction
                    )
                    if final_conjunction == "Or":
                        group_result = any(group_results)
                    else:  # Default to 'And'
                        group_result = all(group_results)
                    results.append(group_result)

    if not results:
        return True

    # Apply main conjunction logic
    if logic_type == "BooleanExpression":
        # For BooleanExpression, check if there's a Conjunction field at the condition level
        # Look for Conjunction field in the logic structure
        conjunction = None
        for key, value in logic_group.items():
            if key.isdigit() and isinstance(value, dict) and "Conjunction" in value:
                conjunction = value["Conjunction"]
                break

        if conjunction == "Or":
            return any(results)
        else:
            return all(results)  # Default to AND
    elif logic_type == "Or":
        return any(results)
    else:  # 'If' or default
        return all(results)


def _evaluate_condition(
    condition: dict, embedded_data_dict: Dict, response_data: Dict = None
) -> bool:
    """Evaluate a single condition from QSF logic."""
    logic_type = condition.get("LogicType", "")
    operator = condition.get("Operator", "")
    left_operand = condition.get("LeftOperand", "")
    right_operand = condition.get("RightOperand", "")

    # Handle different logic types
    if logic_type == "EmbeddedField":
        actual_value = embedded_data_dict.get(left_operand)
    elif logic_type == "Question":
        if response_data is None:
            return False
        actual_value = response_data.get(left_operand)
    else:
        # Unsupported logic type
        return False

    # Apply the operator
    return _apply_operator(actual_value, right_operand, operator)


def _apply_operator(actual_value: Any, expected_value: Any, operator: str) -> bool:
    """Apply the specified operator to compare values."""
    # Handle None values
    if actual_value is None:
        actual_str = ""
        actual_num = None
    else:
        actual_str = str(actual_value)
        try:
            actual_num = (
                float(actual_str)
                if actual_str.replace(".", "", 1).replace("-", "", 1).isdigit()
                else None
            )
        except (ValueError, AttributeError):
            actual_num = None

    if expected_value is None:
        expected_str = ""
        expected_num = None
    else:
        expected_str = str(expected_value)
        try:
            expected_num = (
                float(expected_str)
                if expected_str.replace(".", "", 1).replace("-", "", 1).isdigit()
                else None
            )
        except (ValueError, AttributeError):
            expected_num = None

    # QSF operator mappings
    if operator == "EqualTo":
        return actual_value == expected_value
    elif operator == "NotEqualTo":
        return actual_value != expected_value
    elif operator == "GreaterThan":
        if actual_num is not None and expected_num is not None:
            return actual_num > expected_num
        return actual_str > expected_str
    elif operator == "LessThan":
        if actual_num is not None and expected_num is not None:
            return actual_num < expected_num
        return actual_str < expected_str
    elif operator == "GreaterThanOrEqualTo":
        if actual_num is not None and expected_num is not None:
            return actual_num >= expected_num
        return actual_str >= expected_str
    elif operator == "LessThanOrEqualTo":
        if actual_num is not None and expected_num is not None:
            return actual_num <= expected_num
        return actual_str <= expected_str
    elif operator == "Contains":
        return expected_str.lower() in actual_str.lower()
    elif operator == "DoesNotContain":
        return expected_str.lower() not in actual_str.lower()
    elif operator == "IsEmpty":
        return actual_value is None or actual_str == ""
    elif operator == "IsNotEmpty":
        return actual_value is not None and actual_str != ""
    elif operator == "Selected":
        # For multiple choice questions - check if choice is selected
        if isinstance(actual_value, list):
            return expected_value in actual_value
        return actual_value == expected_value
    elif operator == "NotSelected":
        # For multiple choice questions - check if choice is not selected
        if isinstance(actual_value, list):
            return expected_value not in actual_value
        return actual_value != expected_value
    else:
        # Unknown operator, default to equality check
        return actual_value == expected_value


# Test function to validate both structures
def test_qsf_logic_evaluation():
    """Test with both DisplayLogic and BranchLogic QSF structures."""
    # Test data
    embedded_data = {"Condition": "2", "user_type": "premium", "age": "25"}  # Based on your example

    test_cases = [
        # Original DisplayLogic structure
        {
            "logic": {
                "0": {
                    "0": {
                        "LogicType": "EmbeddedField",
                        "Operator": "EqualTo",
                        "LeftOperand": "user_type",
                        "RightOperand": "premium",
                        "Type": "Expression",
                    },
                    "Type": "If",
                },
                "Type": "BooleanExpression",
            },
            "expected": True,
            "description": "Original DisplayLogic structure",
        },
        # Real-world example from Accuracy_Nudges_parsed.json
        {
            "logic": {
                "0": {
                    "0": {
                        "LogicType": "EmbeddedField",
                        "LeftOperand": "Condition",
                        "Operator": "EqualTo",
                        "RightOperand": "2",
                        "_HiddenExpression": False,
                        "Type": "Expression",
                        "Description": '<span class="ConjDesc">If</span>  <span class="LeftOpDesc">Condition</span> <span class="OpDesc">Is Equal to</span> <span class="RightOpDesc"> 2 </span>',
                    },
                    "1": {
                        "LogicType": "EmbeddedField",
                        "LeftOperand": "Condition",
                        "Operator": "EqualTo",
                        "RightOperand": "3",
                        "_HiddenExpression": False,
                        "Type": "Expression",
                        "Description": '<span class="ConjDesc">Or</span>  <span class="LeftOpDesc">Condition</span> <span class="OpDesc">Is Equal to</span> <span class="RightOpDesc"> 3 </span>',
                        "Conjuction": "Or",  # Note: typo in original JSON
                    },
                    "Type": "If",
                },
                "Type": "BooleanExpression",
            },
            "expected": True,  # Should be True because Condition=2 matches first condition
            "description": "Real-world nested BranchLogic from Accuracy_Nudges_parsed.json",
        },
        # Enhanced DisplayLogic with multiple nested conditions and OR logic
        {
            "logic": {
                "0": {
                    "0": {
                        "LogicType": "EmbeddedField",
                        "Operator": "EqualTo",
                        "LeftOperand": "user_type",
                        "RightOperand": "basic",  # This will be False
                        "Type": "Expression",
                    },
                    "1": {
                        "LogicType": "EmbeddedField",
                        "Operator": "EqualTo",
                        "LeftOperand": "user_type",
                        "RightOperand": "premium",  # This will be True
                        "Type": "Expression",
                        "Conjunction": "Or",
                    },
                    "Type": "If",
                },
                "Type": "BooleanExpression",
            },
            "expected": True,
            "description": "Enhanced DisplayLogic with nested OR conditions",
        },
        # Enhanced DisplayLogic with nested AND conditions (both must be true)
        {
            "logic": {
                "0": {
                    "0": {
                        "LogicType": "EmbeddedField",
                        "Operator": "EqualTo",
                        "LeftOperand": "user_type",
                        "RightOperand": "premium",  # This will be True
                        "Type": "Expression",
                    },
                    "1": {
                        "LogicType": "EmbeddedField",
                        "Operator": "EqualTo",
                        "LeftOperand": "age",
                        "RightOperand": "25",  # This will be True
                        "Type": "Expression",
                        "Conjunction": "And",
                    },
                    "Type": "If",
                },
                "Type": "BooleanExpression",
            },
            "expected": True,
            "description": "Enhanced DisplayLogic with nested AND conditions",
        },
        # Enhanced DisplayLogic with nested AND conditions (one fails)
        {
            "logic": {
                "0": {
                    "0": {
                        "LogicType": "EmbeddedField",
                        "Operator": "EqualTo",
                        "LeftOperand": "user_type",
                        "RightOperand": "premium",  # This will be True
                        "Type": "Expression",
                    },
                    "1": {
                        "LogicType": "EmbeddedField",
                        "Operator": "EqualTo",
                        "LeftOperand": "age",
                        "RightOperand": "30",  # This will be False
                        "Type": "Expression",
                        "Conjunction": "And",
                    },
                    "Type": "If",
                },
                "Type": "BooleanExpression",
            },
            "expected": False,
            "description": "Enhanced DisplayLogic with nested AND conditions (one fails)",
        },
        # Nested structure with different conjunction styles (testing typo handling)
        {
            "logic": {
                "0": {
                    "0": {
                        "LogicType": "EmbeddedField",
                        "LeftOperand": "Condition",
                        "Operator": "EqualTo",
                        "RightOperand": "5",  # This will be False
                        "_HiddenExpression": False,
                        "Type": "Expression",
                    },
                    "1": {
                        "LogicType": "EmbeddedField",
                        "LeftOperand": "user_type",
                        "Operator": "EqualTo",
                        "RightOperand": "premium",  # This will be True
                        "_HiddenExpression": False,
                        "Type": "Expression",
                        "Conjuction": "Or",  # Intentional typo to test handling
                    },
                    "Type": "If",
                },
                "Type": "BooleanExpression",
            },
            "expected": True,  # Should be True because second condition matches (OR logic)
            "description": "Nested structure testing typo handling (Conjuction vs Conjunction)",
        },
    ]

    # Run tests
    print("Testing QSF Logic Evaluation (DisplayLogic & BranchLogic):")
    print("=" * 60)

    for i, test_case in enumerate(test_cases):
        result = eval_display_logic(test_case["logic"], embedded_data)
        status = "PASS" if result == test_case["expected"] else "FAIL"
        print(f"Test {i + 1}: {test_case['description']}")
        print(f"Expected: {test_case['expected']}, Got: {result}")
        print(f"Status: {status}")
        print("-" * 60)


if __name__ == "__main__":
    test_qsf_logic_evaluation()
