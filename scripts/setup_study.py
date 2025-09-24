"""This script sets up the directory structure and configuration file for a new study.

It is based on a Jinja2 YAML template located in `configs/study_template.yaml.j2`.

It:
- Creates folders under `data/<study_name>/` for raw data, parsed QSF, and JSON responses
- Copies the provided QSF and CSV files into the appropriate location
- Renders a custom YAML config file under `configs/<study_name>.yaml`

-----------
Usage:
-----------

From the project root, run:

    poetry run python scripts/setup_study.py <study_name> --qsf /path/to/file.qsf --csv /path/to/responses.csv

Optional arguments (with defaults):

    --include_reasoning       Whether to include reasoning in prompts (default: True)
    --temperature             LLM temperature (default: 1.0)
    --max_personas            Max personas to simulate (default: -1 for all)
    --force_regenerate        Whether to overwrite previous outputs (default: False)
    --system_instruction      Custom system instruction string for the LLM
    --llm_provider            LLM provider (default: "openai")
    --model_name              LLM model name (default: "gpt-4.1")
    --max_tokens              Max tokens in LLM response (default: 16384)
    --max_retries             Retry attempts for failed LLM calls (default: 10)
    --num_workers             Number of parallel workers (default: 300)
    --dry_run                 Preview actions without writing anything
    --persona_text_dir        Directory containing persona text files (default: data/full_persona_text)
    --combine_prompt_end      End of the combined prompt (default: "")
-----------

Example:
-----------
    poetry run python scripts/setup_study.py idea_generation \
        --qsf ~/Downloads/Measures_of_Creativity_Digital_Twins_Toubia.qsf \
        --csv ~/Downloads/response.csv \
        --temperature 0.7 \
        --include_reasoning False \
        --dry_run

"""

import argparse
import json
import shutil
from io import StringIO
from pathlib import Path
from typing import Optional

from jinja2 import Template
from ruamel.yaml import YAML

DEFAULT_SYSTEM_INSTRUCTION = """You are an AI assistant. Your task is to answer the 'New Survey Question' as if you are the person described in the 'Persona Profile' (which consists of their past survey responses). 
Remain consistent with the persona's previous answers and stated traits. Simulate their responses to new questions while accounting for human cognitive limitations, uncertainty, and biases.
Follow all instructions provided for the new question carefully regarding the format of your answer."""

PERSONA_PROMPT_HEADER = "## Persona Profile (This individual's past survey responses):"

PERSONA_QUESTION_PROMPT_SEPARATOR = (
    "## New Survey Question & Instructions (Please respond as the persona described above):"
)


def setup_study(
    study: str,
    output_filename: Optional[str],
    qsf: Optional[str],
    csv: Optional[str],
    include_reasoning: str,
    temperature: float,
    max_personas: int,
    force_regenerate: str,
    system_instruction: str,
    llm_provider: str,
    model_name: str,
    max_tokens: int,
    max_retries: int,
    num_workers: int,
    persona_prompt_header: str,
    persona_question_prompt_separator: str,
    empty_persona_prompt: str,
    base_config: Optional[str],
    dry_run: bool,
    persona_text_dir: str,
    combine_prompt_end: str,
) -> None:
    """Sets up folder structure and configuration for a new study.

    Args:
        study (str): Name of the study.
        output_filename (Optional[str]): Custom output filename for the study.
        qsf (Optional[str]): Path to the QSF file. If not provided, defaults to a standard location.
        csv (Optional[str]): Path to the CSV file. If not provided, defaults to a standard location.
        include_reasoning (str): Whether to include reasoning in prompts, as a string ("true" or "false").
        temperature (float): LLM temperature setting.
        max_personas (int): Maximum number of personas to simulate (-1 for all).
        force_regenerate (str): Whether to overwrite previous outputs, as a string ("true" or "false").
        system_instruction (str): Custom system instruction for the LLM.
        llm_provider (str): LLM provider (e.g., "openai").
        model_name (str): LLM model name (e.g., "gpt-4.1").
        max_tokens (int): Maximum tokens in LLM response.
        max_retries (int): Number of retry attempts for failed LLM calls.
        num_workers (int): Number of parallel workers for processing.
        persona_prompt_header (str): Header for the persona prompt.
        persona_question_prompt_separator (str): Separator for the new question prompt.
        empty_persona_prompt (str): Whether to use an empty persona prompt, as a string ("true" or "false").
        base_config (Optional[str]): Path to a base config file to merge with the generated config.
        dry_run (bool): If True, only prints actions without writing files.
        persona_text_dir (str): Directory containing persona text files.
        combine_prompt_end (str): End of the combined prompt.
    Raises:
        FileNotFoundError: If the provided QSF or CSV files do not exist.
        OSError: If there are issues creating directories or copying files.
    """
    output_filename = output_filename if output_filename else study

    base = Path(__file__).resolve().parents[1]
    study_dir = base / "data" / output_filename
    raw_data = study_dir / "raw_data"
    qsf_json = study_dir / "wave_qsf_json"
    response_json = study_dir / "response_json"

    text_simulation_dir = base / "text_simulation" / output_filename
    text_simulation_questions = text_simulation_dir / "text_questions"
    text_simulation_input = text_simulation_dir / "text_simulation_input"
    text_simulation_output = text_simulation_dir / "text_simulation_output"
    text_simulation_updated = text_simulation_dir / "response_json_llm_imputed"
    text_simulation_csv = text_simulation_dir / "csv_output"

    config_template = base / "configs" / "study_template.yaml.j2"
    config_output_dir = base / "configs" / study
    config_output = config_output_dir / f"{output_filename}.yaml"

    if qsf:
        qsf_src = Path(qsf).expanduser().resolve()
    else:
        qsf_src = base / ".dat" / study / "raw_data" / "survey.qsf"

    if csv:
        csv_src = Path(csv).expanduser().resolve()
    else:
        csv_src = base / ".dat" / study / "raw_data" / "response.csv"

    qsf_dest = raw_data / qsf_src.name
    csv_dest = raw_data / csv_src.name

    with open(config_template) as f:
        template = Template(f.read())

    config_text = template.render(
        output_filename=output_filename,
        qsf_filename=qsf_dest.name,
        qsf_basename=qsf_dest.stem,
        csv_filename=csv_dest.name,
        include_reasoning=include_reasoning,
        system_instruction=system_instruction,
        llm_provider=llm_provider,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=max_retries,
        num_workers=num_workers,
        force_regenerate=force_regenerate,
        max_personas=max_personas,
        persona_prompt_header=persona_prompt_header,
        persona_question_prompt_separator=persona_question_prompt_separator,
        empty_persona_prompt=empty_persona_prompt,
        persona_text_dir=persona_text_dir,
        combine_prompt_end=combine_prompt_end,
    )

    if base_config:
        yaml_parser = YAML()
        yaml_parser.preserve_quotes = True
        yaml_parser.Representer.add_representer(
            type(None), lambda self, data: self.represent_scalar("tag:yaml.org,2002:null", "null")
        )
        yaml_parser.indent(mapping=2, sequence=4, offset=2)
        yaml_parser.width = 4096

        config_dict = yaml_parser.load(StringIO(config_text))
        base_config_dict = yaml_parser.load(Path(config_output_dir / base_config).read_text())

        config_stack = [(base_config_dict, config_dict)]

        while config_stack:
            base_level, config_level = config_stack.pop()
            for base_key, base_val in base_level.items():
                if base_key not in config_level:
                    config_level[base_key] = base_val
                    config_level.yaml_set_comment_before_after_key(
                        base_key, before=f"Added key: {base_key}", after=None
                    )
                    print(f"Added or filled null key: {base_key}")
                elif isinstance(base_val, dict) and isinstance(config_level[base_key], dict):
                    config_stack.append((base_val, config_level[base_key]))

        config_output_stream = StringIO()
        yaml_parser.dump(config_dict, config_output_stream)
        config_text = config_output_stream.getvalue()

    if dry_run:
        print("Dry-run: No files written.")
        print(f"Would create:\n  - {raw_data}\n  - {qsf_json}\n  - {response_json}")
        print(f"Would copy:\n  - {qsf_src} → {qsf_dest}\n  - {csv_src} → {csv_dest}")
        print(f"Would write config to: {config_output}")
        print("----- YAML PREVIEW -----")
        print(config_text)
        print("------------------------")
        return

    for folder in [
        raw_data,
        qsf_json,
        response_json,
        text_simulation_questions,
        text_simulation_input,
        text_simulation_output,
        text_simulation_updated,
        text_simulation_csv,
        config_output_dir,
    ]:
        folder.mkdir(parents=True, exist_ok=True)

    if qsf_src != qsf_dest:
        shutil.copy(qsf_src, qsf_dest)
    if csv_src != csv_dest:
        shutil.copy(csv_src, csv_dest)

    config_output.write_text(config_text)

    print(f"Setup complete: {config_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", required=True, type=str, help="Name of the study")
    parser.add_argument("--output_filename", required=False, type=str)
    parser.add_argument("--qsf", required=False, type=str)
    parser.add_argument("--csv", required=False, type=str)
    parser.add_argument(
        "--include_reasoning",
        required=False,
        type=str,
        default="false",
    )
    parser.add_argument("--temperature", required=False, type=float, default=0.7)
    parser.add_argument("--max_personas", required=False, type=int, default=-1)
    parser.add_argument("--force_regenerate", required=False, type=str, default="false")
    parser.add_argument(
        "--system_instruction", required=False, type=str, default=DEFAULT_SYSTEM_INSTRUCTION
    )
    parser.add_argument("--llm_provider", required=False, type=str, default="openai")
    parser.add_argument("--model_name", required=False, type=str, default="gpt-4.1")
    parser.add_argument("--max_tokens", required=False, type=int, default=16384)
    parser.add_argument("--max_retries", required=False, type=int, default=10)
    parser.add_argument("--num_workers", required=False, type=int, default=300)
    parser.add_argument(
        "--persona_prompt_header", required=False, type=str, default=PERSONA_PROMPT_HEADER
    )
    parser.add_argument(
        "--persona_question_prompt_separator",
        required=False,
        type=str,
        default=PERSONA_QUESTION_PROMPT_SEPARATOR,
    )
    parser.add_argument("--empty_persona_prompt", required=False, type=str, default="false")
    parser.add_argument("--base_config", required=False, type=str, default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--persona_text_dir", required=False, type=str, default="data/full_persona_text"
    )
    parser.add_argument("--combine_prompt_end", required=False, type=str, default="")
    args = parser.parse_args()
    setup_study(
        study=args.study,
        output_filename=args.output_filename,
        qsf=args.qsf,
        csv=args.csv,
        include_reasoning=args.include_reasoning,
        temperature=args.temperature,
        max_personas=args.max_personas,
        force_regenerate=args.force_regenerate,
        system_instruction=args.system_instruction,
        llm_provider=args.llm_provider,
        model_name=args.model_name,
        max_tokens=args.max_tokens,
        max_retries=args.max_retries,
        num_workers=args.num_workers,
        persona_prompt_header=args.persona_prompt_header,
        persona_question_prompt_separator=args.persona_question_prompt_separator,
        empty_persona_prompt=args.empty_persona_prompt,
        base_config=args.base_config,
        dry_run=args.dry_run,
        persona_text_dir=args.persona_text_dir,
        combine_prompt_end=args.combine_prompt_end,
    )
