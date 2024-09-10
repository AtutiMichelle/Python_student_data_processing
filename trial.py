import os
from functions import (
    process_students, split_by_gender, find_special_characters,
    shuffle_and_save, save_as_jsonl, setup_logging, log_gender_counts,
    compute_name_similarity, save_name_similarity
)
from google_drive_utils import backup_files_to_drive

def main():
    # Get the directory of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)

    setup_logging()

    # Process students
    input_file = os.path.join(project_dir, 'data', 'Test Files.xlsx')
    df = process_students(input_file)

    # Split by gender
    male_df, female_df = split_by_gender(df)

    # Create output directory if it doesn't exist
    output_dir = os.path.join(project_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)

    # Save as CSV
    male_df.to_csv(os.path.join(output_dir, 'male_students.csv'), index=False)
    female_df.to_csv(os.path.join(output_dir, 'female_students.csv'), index=False)

    # Save as TSV
    df.to_csv(os.path.join(output_dir, 'all_students.tsv'), sep='\t', index=False)

    # Log gender counts
    log_gender_counts(male_df, female_df)

    # Find students with special characters
    special_chars_df = find_special_characters(df)
    print("Students with special characters in their names:")
    print(special_chars_df['Student Name'].tolist())

    # Shuffle and save as JSON
    shuffle_and_save(df, os.path.join(output_dir, 'shuffled_students.json'))

    # Save as JSONL
    save_as_jsonl(df, os.path.join(output_dir, 'students.jsonl'))

    # Compute name similarity
    similar_pairs = compute_name_similarity(male_df, female_df)
    save_name_similarity(similar_pairs, os.path.join(output_dir, 'name_similarity.json'))

    # Backup files to Google Drive
    output_files = [
        os.path.join(output_dir, 'male_students.csv'),
        os.path.join(output_dir, 'female_students.csv'),
        os.path.join(output_dir, 'all_students.tsv'),
        os.path.join(output_dir, 'shuffled_students.json'),
        os.path.join(output_dir, 'students.jsonl'),
        os.path.join(output_dir, 'name_similarity.json')
    ]
    backup_files_to_drive(output_files)

    print("Processing complete. Check the output folder for results.")

if __name__ == "__main__":
    main()