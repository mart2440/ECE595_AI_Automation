#!/opt/homebrew/bin/python3
# ^ Script ran using homebrew

# ECE595 Final Project LLM Automation
# December 12, 2025
# Group 6

import argparse     # user-friendly CLI 
import csv          # for reading/writing the .csv file
import os           # used to access environment API keys
import time         # machine sleeps in between API calls
import requests     # sends HTTP POST reqs to LLM API

# 1. CALL LLM FUNCTION
# sends the prompt to the LLM model and returns the response
def call_llm(
    base_url,           # API server (Purdue GenAI)
    api_key,            # authentication method
    model,              # decides which model to call
    user_content,       # prompts
    system_prompt=None, 
    temperature=0.2,
    max_tokens=512,
):
    # Used to build the AP URL - standard convention
    url = base_url.rstrip("/") + "/chat/completions"
    # Used to send personalized API key
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Create the format for the messages used by OpenAI
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})

    # Payload for full API
    body = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    
    # Used to send the POST request to the LLM
    response = requests.post(url, headers=headers, json=body, timeout=60)
    
    # An exception is raised if something goes wrong
    # A message is displayed from API error
    if response.status_code != 200:
        raise RuntimeError(
            f"LLM API error {response.status_code}: {response.text}"
        )

    # Will only return the LLM's output response for data collection
    data = response.json()
    return data["choices"][0]["message"]["content"]


# 2. Load Prompts Read from the CSV file 
# CSV file is made of 40 test prompts + 40 role-based prompts selected 
# from the TruthfulQA dataset 
def load_prompts(input_csv):
    # extracts the text fields under the prompt variable in CSV file
    prompts = []
    # Used to open the csv file 
    with open(input_csv, newline="", encoding="utf-8") as f:
        # DictReader reads each column using header
        reader = csv.DictReader(f)
        for row in reader:
            # Removes some of the case sensitivity with header names
            prompt_text = (
                row.get("prompt")
                or row.get("Prompt")
                or row.get("question")
                or row.get("Question")
            )
            # Will skip the row if no prompt text is found
            if not prompt_text:
                continue
            row["__prompt_text__"] = prompt_text
            prompts.append(row)
    return prompts


# MAIN function
# the actual running of the program
def main():
    # Sets the CLI arguments 
    parser = argparse.ArgumentParser(description="Run prompts through ChatGPT")

    # Configurations for arguments
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--base_url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api_key_env", required=True)
    parser.add_argument("--system_prompt", default=None)
    parser.add_argument("--sleep", type=float, default=0.5)

    # parses the arguments
    args = parser.parse_args()

    # Loads the API key from environment
    api_key = os.getenv(args.api_key_env)
    # If no API key is found, scripts stops
    if not api_key:
        raise EnvironmentError(
            f"Environment variable {args.api_key_env} is not set. "
            "Set it to your API key before running."
        )

    # Loads all the prompts from csv file
    print(f"Loading prompts from: {args.input_csv}")
    prompts = load_prompts(args.input_csv)
    print(f"Found {len(prompts)} prompts.")

    # Prepares the ouput csv file of results
    fieldnames = list(prompts[0].keys())
    for extra in ["model_name", "response_text"]:
        if extra not in fieldnames:
            fieldnames.append(extra)

    # Opens the csv file of outputs to begin writing 
    with open(args.output_csv, "w", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()

        # Machine will loop through every prompt feeding it to the LLM
        for i, row in enumerate(prompts, start=1):
            prompt_text = row["__prompt_text__"]
            # Shows progress of what prompt number it is on out of the 80 prompts
            print(f"[{i}/{len(prompts)}] Sending prompt: {prompt_text[:80]!r}...")

            # Calls the LLM to avoid errors mid-run
            try:
                response_text = call_llm(
                    base_url=args.base_url,
                    api_key=api_key,
                    model=args.model,
                    user_content=prompt_text,
                    system_prompt=args.system_prompt,
                )
            except Exception as e:
                print(f"Error on prompt {i}: {e}")
                response_text = f"ERROR: {e}"

            # Used to save the output responses from the LLM into the output csv
            row["model_name"] = args.model
            row["response_text"] = response_text
            writer.writerow(row)
            # System waits before the next call
            time.sleep(args.sleep)
    # Notifies user that the program has finished running
    print(f"Done. Results saved to: {args.output_csv}")

if __name__ == "__main__":
    main()
