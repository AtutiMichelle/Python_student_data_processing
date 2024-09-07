import pandas as pd
import logging
import re
from functions import load_data, generate_email, save_to_csv, save_to_tsv
from constraints import validate_email
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import json
import os.path
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the scopes for Google Drive API
SCOPES = ['https://www.googleapis.com/auth/drive.file']

def authenticate_google_drive():
    creds = None

    # Check if there is an existing token
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)

    # If there are no valid credentials, prompt the user to log in
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)

        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json(indent=4))

    return creds


def upload_file_to_drive(file_path, file_name):
    creds = authenticate_google_drive()
    service = build('drive', 'v3', credentials=creds)

    file_metadata = {'name': file_name}
    media = MediaFileUpload(file_path, mimetype='application/json')
    file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()

    print(f'File ID: {file.get("id")}')


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
        special_characters = re.compile(r'[^a-zA-Z\s,]')
        names_with_special_chars = df[df['Student Name'].apply(lambda x: bool(special_characters.search(x)))]
        logging.info('Names with special characters:')
        logging.info(names_with_special_chars['Student Name'].tolist())

        model = SentenceTransformer('LaBSE')
        male_names = male_students['Student Name'].tolist()
        female_names = female_students['Student Name'].tolist()
        male_embeddings = model.encode(male_names)
        female_embeddings = model.encode(female_names)
        similarities = cosine_similarity(male_embeddings, female_embeddings)
        similarity_results = []
        for i, male_name in enumerate(male_names):
            for j, female_name in enumerate(female_names):
                if similarities[i, j] >= 0.5:
                    similarity_results.append({
                        'Male Name': male_name,
                        'Female Name': female_name,
                        'Similarity': float(similarities[i, j])
                    })
        with open('similarity_results.json', 'w') as f:
            json.dump(similarity_results, f, indent=4)

        # Merge all documents
        files = ['male_students.csv', 'female_students.csv']
        merged_df = load_and_merge_files(files)

        # Prepare JSON data
        df_shuffled = merged_df.sample(frac=1).reset_index(drop=True)

        # Save the original JSON file
        df_shuffled.to_json('shuffled_student_data.json', orient='records', lines=False, indent=4)

        # Create a new list to store data in the desired format for JSONL
        formatted_data = []

        for index, row in df_shuffled.iterrows():
            formatted_entry = {
                "id": str(index),
                "student_number": str(row['Student Number']),
                "additional_details": [
                    {
                        "dob": row['DoB'],
                        "gender": row['Gender'].lower(),
                        "special_character": ["yes" if re.search(r'[^a-zA-Z\s]', row['Student Name']) else "no"],
                        "name_similar": ["yes" if any(result['Similarity'] >= 0.5 for result in similarity_results if
                                                      result['Male Name'] == row['Student Name'] or result[
                                                          'Female Name'] == row['Student Name']) else "no"]
                    }
                ]
            }
            formatted_data.append(formatted_entry)

        # Save the formatted data to the JSONL file
        with open('student_data_copy.jsonl', 'w') as jsonl_file:
            for entry in formatted_data:
                json.dump(entry, jsonl_file, indent=4)
                jsonl_file.write('\n')

        logging.info('Data processed and saved successfully.')

        # Upload files to Google Drive
        upload_file_to_drive('shuffled_student_data.json', 'shuffled_student_data.json')
        upload_file_to_drive('student_data_copy.jsonl', 'student_data_copy.jsonl')

    except Exception as e:
        logging.error(f'An error occurred: {e}')

    finally:
        logging.info('Process completed.')


if __name__ == "__main__":
    main()
