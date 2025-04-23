from pathlib import Path

def load_file_content(file_path):
    with open(file_path, encoding="utf-8") as file:
        return file.read()

# Load API_PROMPT
api_prompt_path = Path(__file__).parent.parent.parent / "prompts" / "datalab-api-prompt.md"
API_PROMPT = load_file_content(api_prompt_path)

# Load SYSTEM_PROMPT
system_prompt_path = Path(__file__).parent.parent.parent / "prompts" / "system-prompt.md"
SYSTEM_PROMPT = load_file_content(system_prompt_path)

# Load CODE_PROMPT
code_prompt_path = Path(__file__).parent.parent.parent / "prompts" / "code-prompt.md"
CODE_PROMPT = load_file_content(code_prompt_path)

# Load INGEST_PROMPT
ingest_prompt_path = Path(__file__).parent.parent.parent / "prompts" / "ingest-prompt.md"
INGEST_PROMPT = load_file_content(ingest_prompt_path)

if __name__ == "__main__":
    print(SYSTEM_PROMPT)
    print(API_PROMPT)
    print(CODE_PROMPT)
    print(INGEST_PROMPT)
