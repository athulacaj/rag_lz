import os
import json
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from functions.make_section import extract_sections
from functions.query_utils import get_data_using_llm
import re
from config import MODEL_NAME,PARSER,PROJECT

should_owerrite=False

GENERAL_TEMPLATE = """Role: You are an expert Resume/CV Parser.

give me a proper Json response which can parse with the following keys:

general: name,email,position
output: should return a json with key general

Input CV Text:

{question}
"""
SKILLS_TEMPLATE = """
give me a json response as a list of skills
no other keys or content would be there only key skills. It should be only a list of skills.It should be clean and simple.

output: should return a json with key skills
for eg: {{skills:[]}}

Input CV Text:

{question}
"""
EXPERIENCE_TEMPLATE = """
give me a proper Json response which can parse with the following keys:
experience:[{{company_name,start_date,end_date,position,description}}]
here experience is a list of objects with keys company_name,start_date,end_date,position,description.
Try your best to get the values from the text.If you cant get the values from the text, just return empty string.
Dont include any other keys or content.
output: should return a json with key experience

Input CV Text:
{question}
"""

FULL_TEMPLATE = """
Role: You are an expert Resume/CV Parser.
give me a proper Json response which can parse with the following keys:
general: name,email,position
skills: list of skills
experience:[{{company_name,start_date,end_date,position,description}}],
here experience is a list of objects with keys company_name,start_date,end_date,position,description.

Try your best to get the values from the text.If you cant get the values from the text, just return empty string.
Dont include any other keys or content.
re
Input CV Text:
{question}
"""

# Define the Prompt Template
prompts_template = {
    "general": GENERAL_TEMPLATE,
    "skills": SKILLS_TEMPLATE,
    "experience": EXPERIENCE_TEMPLATE
}
def parser_with_llm_full(data,cv_text):
    prompt = ChatPromptTemplate.from_template(FULL_TEMPLATE)
    model = ChatOllama(model=MODEL_NAME, format="json")
    chain = prompt | model
    response = chain.invoke({"cv_text": cv_text})
    content = response.content
    cleaned_content = content.strip()
    if cleaned_content.startswith("```json"):
        cleaned_content = cleaned_content[7:]
    if cleaned_content.endswith("```"):
        cleaned_content = cleaned_content[:-3]
    cleaned_content = cleaned_content.strip()

    try:
        json_data = json.loads(cleaned_content)
        # optionally add source filename
        # json_data["_source_file"] = filename 
        print(json_data)
        return json_data
    except json.JSONDecodeError as e:
        print(f"Failed to decode JSON {e}")


def find_email_from_text(text):
    email_regex = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,4}"
    match = re.search(email_regex, text)
    if match:
        return match.group(0)
    return None

def parser_with_llm(data,cv_data):
    structured_data={}
    for key, value in prompts_template.items():
        print("calling llm for key ",key)
        json_data = get_data_using_llm(data[key],value,"",MODEL_NAME)
        try:
            # json_data = json.loads(cleaned_content)
            #  check the key json_data[key] 

            
            if(json_data and json_data[key] ):
                structured_data[key]=json_data[key]
            if key=="general":
                email = find_email_from_text(cv_data)
                if email:
                    structured_data[key]["email"] = email
        except json.JSONDecodeError as e:
            print(f"Failed to decode JSON {e}")
        except Exception as e:
            print(f"Failed to decode JSON {json_data}")
            # throw e
            raise e
    return structured_data
    
def parser_md_to_json(data_path):
    if not os.path.exists(data_path):
        print(f"Directory '{data_path}' does not exist.")
        return []

    results = []

    processed_count = 0
    
    # Count json files first
    md_files = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f)) and f.lower().endswith(".md")]
    total_files = len(md_files)

    print(f"Found {total_files} .json files in {data_path}")

    for filename in md_files:
        #  check the file is already existing
        os.makedirs(os.path.join("processed",PROJECT,"json",PARSER,MODEL_NAME),exist_ok=True)
        path_to_save=os.path.join("processed",PROJECT,"json",PARSER,MODEL_NAME, filename.replace(".md", ".json"))
        if os.path.exists(path_to_save):
            if not should_owerrite:
                print(f"File {filename} already exists. Skipping...")
                continue
        file_path = os.path.join(data_path, filename)
        print(f"Processing {filename}...")
        with open(file_path, "r", encoding="utf-8") as f:
            data = f.read()
            print("extracting sections")
            json_data=extract_sections(data)
            print("calling llm to make structured data")
            llm_data=parser_with_llm(json_data,data)
            if llm_data:
                json_data["structured_data"]=llm_data
                processed_count += 1
            # save json data to a file
            with open(path_to_save, "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=4)
                print("saved json data to a file")

    print(f"Successfully processed {processed_count}/{total_files} files.")
    return results

if __name__ == "__main__":
    # Example usage

    parser_md_to_json(os.path.join("processed",PROJECT,"md",PARSER))
    # output = parser_with_llm("processed/json/marker")
    
    # # Optional: save to a file to verify
    # if output:
    #     with open("parsed_cvs.json", "w", encoding="utf-8") as f:
    #         json.dump(output, f, indent=4)
    #     print("Saved parsed data to parsed_cvs.json")

