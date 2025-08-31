import json
from datasets import Dataset, DatasetDict
from huggingface_hub import login # Import login for interactive login

# --- Step 1: Define your file paths and dataset name ---
# IMPORTANT: Replace these with the actual paths to your JSON files
# and your desired dataset name on the Hugging Face Hub.
FILE_PATHS = {
    "forget_standard": "",
    "forget_train_para": "",
    "forget_eval_para": "",
    "retain_standard": "",
    "retain_train_para": "",
    "retain_eval_para": "",
    "relearn_standard": "",

}

# The name your dataset will have on the Hugging Face Hub (e.g., "my-qa-dataset")
# DATASET_NAME_ON_HUB  # Replace with your Hugging Face username and desired dataset name
# The base name for your datasets on the Hugging Face Hub.
# Each subset will be uploaded as 'BASE_DATASET_NAME-subset_name'.
# Replace with your Hugging Face username and desired base dataset name.
BASE_DATASET_NAME_ON_HUB = ""



# --- Step 3: Load each JSON data and push as a separate Dataset ---
uploaded_dataset_links = []
for subset_name, file_path in FILE_PATHS.items():
    print(f"\n--- Processing subset: '{subset_name}' ---")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Basic validation: Ensure data is a list of dictionaries
            if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
                print(f"Warning: {file_path} does not seem to contain a list of dictionaries. "
                      "Please ensure your JSON is formatted as a list of Q/A objects. Skipping this file.")
                continue

            print(f"Loaded {len(data)} items from {file_path}.")

            # Create a Hugging Face Dataset from the loaded data
            # No harmonization needed here, as each will be independent.
            dataset = Dataset.from_list(data)
            print(f"Created Hugging Face Dataset for '{subset_name}'.")

            # Define the unique name for this specific dataset on the Hub
            dataset_name_on_hub = f"{BASE_DATASET_NAME_ON_HUB}-{subset_name}"

            # Push the individual Dataset to the Hugging Face Hub
            print(f"Attempting to push dataset to Hugging Face Hub: {dataset_name_on_hub}")
            dataset.push_to_hub(dataset_name_on_hub)
            print(f"Successfully pushed '{subset_name}' to the Hugging Face Hub!")
            link = f"https://huggingface.co/datasets/{dataset_name_on_hub}"
            print(f"You can view it at: {link}")
            uploaded_dataset_links.append(link)

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}. Please check the path and try again. Skipping this subset.")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}. Please check the JSON format. Skipping this subset.")
    except Exception as e:
        print(f"An unexpected error occurred while processing {file_path}: {e}. Skipping this subset.")

print("\n--- All processing complete ---")
if uploaded_dataset_links:
    print("\nAll datasets successfully pushed to the Hugging Face Hub:")
    for link in uploaded_dataset_links:
        print(f"- {link}")
else:
    print("No datasets were successfully pushed to the Hugging Face Hub.")