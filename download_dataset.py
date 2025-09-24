#!/usr/bin/env python3
"""Download and process the Twin-2K-500 dataset from Hugging Face.

This script downloads the dataset and organizes it into appropriate directories
with proper file naming conventions.
"""

import json
import os

import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from tqdm.auto import tqdm


def create_directories():
    """Create necessary directories for storing the dataset."""
    directories = [
        "data/mega_persona_json/mega_persona",
        "data/mega_persona_json/answer_blocks",
        "data/mega_persona_summary_text",
        "data/full_persona_json",
        "data/full_persona_text",
        "data/wave_csv",
        "data/mega_persona_summary_csv",
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def download_wave_split_data():
    """Download and process the wave split data from the dataset.

    This includes mega persona data and answer blocks.
    """
    print("Downloading wave split data...")
    dataset = load_dataset("LLM-Digital-Twin/Twin-2K-500", "wave_split", cache_dir="./cache")

    # Extract data from the dataset
    pid = dataset["data"]["pid"]
    mega_persona = dataset["data"]["wave1_3_persona_json"]
    wave4_Q_wave1_3_A_answer_blocks = dataset["data"]["wave4_Q_wave1_3_A"]
    wave4_Q_wave4_A_answer_blocks = dataset["data"]["wave4_Q_wave4_A"]

    # Save mega_persona data
    print("Saving mega persona data...")
    for idx, persona in zip(pid, mega_persona, strict=False):
        with open(f"data/mega_persona_json/mega_persona/pid_{idx}_mega_persona.json", "w") as f:
            f.write(persona)

    # Save answer blocks
    print("Saving answer blocks...")
    for idx, answer_block in zip(pid, wave4_Q_wave1_3_A_answer_blocks, strict=False):
        with open(
            f"data/mega_persona_json/answer_blocks/pid_{idx}_wave4_Q_wave1_3_A.json", "w"
        ) as f:
            f.write(answer_block)

    for idx, answer_block in zip(pid, wave4_Q_wave4_A_answer_blocks, strict=False):
        with open(f"data/mega_persona_json/answer_blocks/pid_{idx}_wave4_Q_wave4_A.json", "w") as f:
            f.write(answer_block)


def download_full_persona_data():
    """Download and process the full persona data from the dataset.

    This includes persona summaries.
    """
    print("Downloading full persona data...")
    dataset = load_dataset("LLM-Digital-Twin/Twin-2K-500", "full_persona", cache_dir="./cache")

    # Extract data from the dataset
    pid = dataset["data"]["pid"]
    persona_summary = dataset["data"]["persona_summary"]

    # Save persona summaries
    print("Saving persona summaries...")
    for idx, summary in zip(pid, persona_summary, strict=False):
        with open(f"data/mega_persona_summary_text/pid_{idx}_mega_persona.txt", "w") as f:
            if summary is not None:
                f.write(summary)  # Write the summary directly without json.dump
            else:
                # print(f"No summary available for pid {idx}")
                pass

    # Save full_persona_json
    print("Saving full persona json...")
    persona_json = dataset["data"]["persona_json"]
    for idx, persona in zip(pid, persona_json, strict=False):
        with open(f"data/full_persona_json/pid_{idx}_full_persona.json", "w") as f:
            f.write(json.dumps(persona))

    # Save full_persona_text
    print("Saving full persona text...")
    persona_text = dataset["data"]["persona_text"]
    for idx, text in zip(pid, persona_text, strict=False):
        with open(f"data/full_persona_text/pid_{idx}_full_persona.txt", "w") as f:
            f.write(text)


def download_raw_data():
    """Download raw CSV files from the dataset repository."""
    print("Downloading raw data...")
    raw_files = [
        "wave_1_labels_anonymized.csv",
        "wave_1_numbers_anonymized.csv",
        "wave_2_labels_anonymized.csv",
        "wave_2_numbers_anonymized.csv",
        "wave_3_labels_anonymized.csv",
        "wave_3_numbers_anonymized.csv",
        "wave_4_labels_anonymized.csv",
        "wave_4_numbers_anonymized.csv",
    ]

    for file_name in tqdm(raw_files, desc="Downloading raw files"):
        downloaded_file_path = hf_hub_download(
            repo_id="LLM-Digital-Twin/Twin-2K-500",
            filename=f"raw_data/{file_name}",
            repo_type="dataset",
            cache_dir="./cache",
        )
        df = pd.read_csv(downloaded_file_path, low_memory=False)
        output_path = f"data/wave_csv/{file_name}"
        df.to_csv(output_path, index=False)


def download_wave_scores():
    """Download wave score CSV files for idea generation analysis."""
    print("Downloading wave score files...")
    wave_score_files = [
        "wave 1 scores.csv",
        "wave 2 scores.csv",
        "wave 3 scores.csv",
    ]

    for file_name in tqdm(wave_score_files, desc="Downloading wave score files"):
        try:
            downloaded_file_path = hf_hub_download(
                repo_id="LLM-Digital-Twin/Twin-2K-500",
                filename=f"raw_data/{file_name}",
                repo_type="dataset",
                cache_dir="./cache",
            )
            df = pd.read_csv(downloaded_file_path, low_memory=False)
            output_path = f"data/mega_persona_summary_csv/{file_name}"
            df.to_csv(output_path, index=False)
            print(f"Successfully downloaded: {file_name}")
        except Exception as e:
            print(f"Error downloading {file_name}: {e}")


def main():
    """Main function to orchestrate the download and processing of the dataset."""
    try:
        # Create necessary directories
        create_directories()

        # Download and process wave split data
        download_wave_split_data()

        # Download and process full persona data
        download_full_persona_data()

        # Download raw data
        download_raw_data()

        # Download wave score files
        download_wave_scores()

        print("Dataset download and processing completed successfully!")

    except Exception as e:
        print(f"An error occurred: {e!s}")


if __name__ == "__main__":
    main()
