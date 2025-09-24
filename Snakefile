configfile: "configs/infotainment.yaml"

import glob
import os
import yaml
import json

def get_qsf_processing_python_dependencies():
    """Get all Python files in the processing_qualtrics_qsf directory."""
    main_py_files = glob.glob("processing_qualtrics_qsf/*.py")
    subdir_py_files = glob.glob("processing_qualtrics_qsf/**/*.py", recursive=True)
    return list(set(main_py_files + subdir_py_files))

def get_csv_processing_dependencies():
    """Get all Python files in the processing_qualtrics_csv directory."""
    main_py_files = glob.glob("processing_qualtrics_csv/*.py")
    subdir_py_files = glob.glob("processing_qualtrics_csv/**/*.py", recursive=True)
    return list(set(main_py_files + subdir_py_files))

def get_personas_to_text_dependencies():
    """Get Python files needed for converting personas to text."""
    return ["text_simulation/convert_personas_to_text.py"]

def get_questions_to_text_dependencies():
    """Get Python files needed for converting questions to text."""
    main_files = ["text_simulation/convert_question_json_to_text.py"]
    formatter_files = glob.glob("text_simulation/question_formatters/*.py")
    return main_files + formatter_files

def get_simulation_input_dependencies():
    """Get Python files needed for creating simulation input."""
    return ["text_simulation/create_text_simulation_input.py"]

def get_llm_simulation_dependencies():
    """Get Python files needed for running LLM simulations."""
    main_py_files = ["text_simulation/run_LLM_simulations.py", "text_simulation/postprocess_responses.py"]
    helper_py_files = glob.glob("text_simulation/llm_batch_helper/*.py")
    return list(set(main_py_files + helper_py_files))

def get_convert_to_csv_dependencies():
    """Get Python files needed for converting responses to CSV."""
    return ["text_simulation/convert_responses_to_csv.py"]

# ============================================================================
# MAIN TARGETS
# ============================================================================

rule all:
    input:
        config["qsf_to_json"]["output_file"],
        config["csv_to_fill_answers"]["output_dir"],
        config["text_simulation"]["LLM_simulation"]["output_dir"],
        config["text_simulation"]["LLM_simulation"]["output_updated_questions_dir"],
        config["text_simulation"]["convert_to_csv"]["output_dir"]

rule qualtrics_only:
    input:
        config["qsf_to_json"]["output_file"],
        config["csv_to_fill_answers"]["output_dir"]

rule text_simulation_only:
    input:
        config["text_simulation"]["LLM_simulation"]["output_dir"],
        config["text_simulation"]["convert_to_csv"]["output_dir"]

# ============================================================================
# QUALTRICS PROCESSING RULES
# ============================================================================

rule process_qsf:
    input:
        qsf=config["qsf_to_json"]["input_file"],
        python_files=get_qsf_processing_python_dependencies(),
    output:
        config["qsf_to_json"]["output_file"]
    shell:
        "poetry run python processing_qualtrics_qsf/parse_qsf.py --config {workflow.configfiles[0]}"

rule process_csv:
    input:
        template=config["qsf_to_json"]["output_file"],
        csv=config["csv_to_fill_answers"]["input_csv_file"],
        python_files=get_csv_processing_dependencies(),
    output:
        directory(config["csv_to_fill_answers"]["output_dir"])
    shell:
        "poetry run python processing_qualtrics_csv/get_responses.py --config {workflow.configfiles[0]}"

# ============================================================================
# TEXT SIMULATION RULES
# ============================================================================

# rule convert_personas:
#     input:
#         persona_dir=config["text_simulation"]["personas_to_texts"]["persona_json_dir"]
#     output:
#         directory(config["text_simulation"]["personas_to_texts"]["output_text_dir"])
#     shell:
#         "poetry run python text_simulation/convert_personas_to_text.py --config {workflow.configfiles[0]}"

rule convert_questions:
    input:
        input_path=config["text_simulation"]["questions_to_texts"]["input_path"],
        python_files=get_questions_to_text_dependencies(),
    output:
        directory(config["text_simulation"]["questions_to_texts"]["output_dir"])
    shell:
        "poetry run python text_simulation/convert_question_json_to_text.py --config {workflow.configfiles[0]}"

rule create_simulation_input:
    input:
        personas=config["text_simulation"]["create_text_simulation_input"]["persona_text_dir"],
        questions=config["text_simulation"]["create_text_simulation_input"]["question_prompts_dir"]
    output:
        directory(config["text_simulation"]["create_text_simulation_input"]["output_combined_prompts_dir"])
    shell:
        "poetry run python text_simulation/create_text_simulation_input.py --config {workflow.configfiles[0]}"

rule run_llm_simulation:
    input:
        simulation_input=config["text_simulation"]["create_text_simulation_input"]["output_combined_prompts_dir"],
        question_json_base_dir=config["text_simulation"]["LLM_simulation"]["question_json_base_dir"],
        python_files=get_llm_simulation_dependencies(),
    output:
        directory(config["text_simulation"]["LLM_simulation"]["output_dir"]),
        directory(config["text_simulation"]["LLM_simulation"]["output_updated_questions_dir"]),
    shell:
        "poetry run python text_simulation/run_LLM_simulations.py --config {workflow.configfiles[0]}"

rule convert_to_csv:
    input:
        json_dir=config["text_simulation"]["convert_to_csv"]["input_json_dir"],
        reference_csv=config["text_simulation"]["convert_to_csv"]["reference_csv_path"],
        llm_output=config["text_simulation"]["LLM_simulation"]["output_dir"],
        python_files=get_convert_to_csv_dependencies(),
    output:
        directory(config["text_simulation"]["convert_to_csv"]["output_dir"])
    shell:
        "poetry run python text_simulation/convert_responses_to_csv.py --config {workflow.configfiles[0]}"