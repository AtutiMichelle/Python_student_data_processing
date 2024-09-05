import pandas as pd
import logging
import re
from functions import load_data, generate_email, save_to_csv, save_to_tsv
from constraints import validate_email
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import json

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_and_merge_files(files):
    dfs = [pd.read_csv(file) for file in files]
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df


def main():
    try:
        logging.info('Starting the process...')

        # Load the data
        logging.info('Loading data from Test Files.xlsx...')
        df = load_data('F:\\PycharmProjects\\Codelabs1\\Test Files.xlsx')
        logging.info(f'Data loaded successfully with {len(df)} records.')

        # Generate email addresses
        logging.info('Generating email addresses...')
        df['Email Address'] = df['Student Name'].apply(generate_email)
        logging.info('Email addresses generated successfully.')

        # Separate Male and Female students
        male_students = df[df['Gender'] == 'M']
        female_students = df[df['Gender'] == 'F']

        # Save lists to files
        male_students.to_csv('male_students.csv', index=False)
        female_students.to_csv('female_students.csv', index=False)
        logging.info(f'Separated lists saved: {len(male_students)} Male and {len(female_students)} Female students.')

        # Find names with special characters
        special_characters = re.compile(r'[^a-zA-Z\s]')
        names_with_special_chars = df[df['Student Name'].apply(lambda x: bool(special_characters.search(x)))]
        logging.info('Names with special characters:')
        logging.info(names_with_special_chars['Student Name'].tolist())

        # Merge all documents
        files = ['male_students.csv', 'female_students.csv']
        merged_df = load_and_merge_files(files)

        # Prepare JSON data
        json_data = []
        for idx, row in merged_df.iterrows():
            dob = row['DoB']
            gender = row['Gender'].lower()
            special_character = 'yes' if special_characters.search(row['Student Name']) else 'no'
            name_similar = 'no'  # Placeholder for similarity checking, not implemented
            json_data.append({
                'id': str(idx),
                'student_number': row['Student Number'],
                'additional_details': [{
                    'dob': dob,
                    'gender': gender,
                    'special_character': [special_character],
                    'name_similar': [name_similar]
                }]
            })

        # Shuffle and Save Data
        df_shuffled = pd.DataFrame(json_data).sample(frac=1).reset_index(drop=True)
        df_shuffled.to_json('shuffled_data.json', orient='records', lines=False, indent=4)

        # Save as JSONL
        with open('shuffled_data.jsonl', 'w') as f:
            for record in json_data:
                json.dump(record, f, indent=4)
                # f.write('\n')

        logging.info('Data processed and saved successfully.')

    except Exception as e:
        logging.error(f'An error occurred: {e}')

    finally:
        logging.info('Process completed.')



if __name__ == "__main__":
    main()
