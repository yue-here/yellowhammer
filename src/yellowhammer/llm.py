import os
import json
import chromadb
from typing import Any, cast
from dataclasses import dataclass
from datetime import date
from datalab_api import DatalabClient
from typing import Annotated, Any, Union
from typing_extensions import TypeAlias
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models import KnownModelName
from .prompt import API_PROMPT, SYSTEM_PROMPT, CODE_PROMPT, INGEST_PROMPT
from .fulltext_search import FullTextSearchTool
from .utils import process_item_for_chat
from pydantic import BaseModel, Field, ConfigDict
from dotenv import load_dotenv

# Load environment variables from .env file for local development
load_dotenv()

# Context dependencies for agent
@dataclass
class Deps:
    client: DatalabClient
    datalab_api_key: str | None
    datalab_url: str | None
    item_manifest: dict[str, Any] | None = None
    item_search_id_buffer: dict[str, Any] | None = None
    items_buffer: dict[str, Any] | None = None
    search_buffer: list[dict] | None = None
    search_tool: FullTextSearchTool | None = None

# Output schemas
class TextResponse(BaseModel):
    text: str = Field(description="A conversational response")

class CodeResponse(BaseModel):
    text: str = Field(description="A description of the code")
    code: str = Field(description="A code snippet")

Response: TypeAlias = Union[TextResponse, CodeResponse]

# Define preferred model
default_model = 'google-gla:gemini-2.5-flash-lite'
model = cast(KnownModelName, os.getenv('AI_MODEL', default_model)) # let user set the model via env var

# API key will be detected from environment variables. For the main providers:
# OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY
# TODO: add getpass to example notebook for user to input API key
# PydanticAI known models: https://ai.pydantic.dev/api/models/base/

# Use Output Functions to hand off to a new agent that outputs directly to the user
# Use this to avoid rephrasing by the main agent.
# Recently implemented in https://github.com/pydantic/pydantic-ai/pull/1785

# Direct output function for item inspection
async def inspect_items_output(ctx: RunContext[Deps], prompt: str) -> str:
    """
    If search item details have been successfully retrieved with a get_item_details call,
    this function uses an LLM agent to inspect these items and returns the result directly.

    Args:
        prompt: A system prompt for the inspection agent
    """
    if ctx.deps.items_buffer == None:
        return "Error: No items buffer found, run get_item_details first."

    # Items buffer to plaintext
    items_buffer = json.dumps(ctx.deps.items_buffer, separators=(',', ':'))
    
    # Create an agent to inspect the items
    inspection_agent = Agent(model, system_prompt=prompt, result_type=Response, instrument=True)
    result = await inspection_agent.run(items_buffer, deps=ctx.deps)

    # Return the result directly - no need to ask main agent to rephrase
    return result.output if result.output else "No inspection result available."

# Direct output function for query inspection
async def inspect_query_output(ctx: RunContext[Deps], prompt: str) -> str:
    """
    If a search query has been successfully run with a item_query call,
    this function uses an LLM agent to inspect the search buffer and returns the result directly.

    Args:
        prompt: A system prompt for the inspection agent
    """
    if ctx.deps.search_buffer == None:
        return "Error: No search buffer found, run item_query first."
    
    # Search buffer to plaintext
    search_buffer = json.dumps(ctx.deps.search_buffer, separators=(',', ':'))
    inspection_agent = Agent(model, system_prompt=prompt, result_type=Response, instrument=True)
    result = await inspection_agent.run(search_buffer, deps=ctx.deps)
    
    # Return the result directly - no need to ask main agent to rephrase
    return result.output if result.output else "No inspection result available."

# Direct output function for code writing
async def code_writer_output(ctx: RunContext[Deps], prompt: str) -> str:
    """
    This function uses an LLM agent to write code based on a prompt. 
    It has information about the datalab API syntax and returns the result directly.

    Args:
        prompt: A system prompt for the code writer agent
    """
    code_agent = Agent(model, result_type=Response, system_prompt=CODE_PROMPT, instrument=True)
    result = await code_agent.run(prompt, deps=ctx.deps)
    
    # Return the result directly - no need to ask main agent to rephrase
    return result.output if result.output else "No code generation result available."


# Define main agent
datalab_agent = Agent[Deps, Union[Response, str]](
    model,
    system_prompt=SYSTEM_PROMPT.format(API_PROMPT),
    output_type=[Response, inspect_items_output, inspect_query_output, code_writer_output], # Output can be Response or output function
    deps_type=Deps,
    retries=2,
    instrument=True, # Enable logging
)

# Define tools
# TODO add an option to return items of all types in the API call
# @datalab_agent.tool(retries=2)
# async def get_items(ctx: RunContext[Deps], item_type: str) -> str:
#     """Use the datalab get_items function to get a JSON of all items in the database.

#     Args:
#         ctx: The context.
#         item_type: Usually 'samples'

#     Returns:

#     """
#     # TODO check API key when the agent is initialized
#     # if ctx.deps.datalab_api_key is None:
#     #     # if no API key is provided, say so
#     #     return {'error': 'No datalab API key found.'}
    
#     if ctx.deps.item_manifest is None:
#         with ctx.deps.client(ctx.deps.datalab_url) as client:
#             item_type = "samples"
#             ctx.deps.item_manifest = client.get_items(item_type=item_type)
#             # Initialize the search tool with the fetched items
#             ctx.deps.search_tool = FullTextSearchTool(ctx.deps.item_manifest)
#             return "Items retrieved."
    
#     if ctx.deps.item_manifest is not None and ctx.deps.search_tool is None:
#         ctx.deps.search_tool = FullTextSearchTool(ctx.deps.item_manifest)

#     if ctx.deps.item_manifest is not None:
#         return "Items already available."

#     # Set 2 retries in decorator to prevent looping
#     raise ModelRetry('No data returned, try again.')

# Tool to query item manifest if it exists
@datalab_agent.tool
async def item_query(ctx: RunContext[Deps], query: str) -> str | dict[str, Any] | list[dict[str, Any]]:
    """Use full-text search to find items matching your query.
    This searches through all fields in the item manifest including nested data.
    
    Args:
        query: Keyword query string (e.g., "John Doe Fe3O4")

    Returns:
        List of dictionaries each representing one matching item
    """
    if ctx.deps.item_manifest is None:
        return {'error': 'Use get_items to generate items list first'}
    
    # Initialize search tool if not already done
    if ctx.deps.search_tool is None:
        ctx.deps.search_tool = FullTextSearchTool(ctx.deps.item_manifest)
    
    # Perform the search using the FullTextSearchTool
    matching_items = ctx.deps.search_tool.natural_language_search(query)
    
    if matching_items:
        # store the query in the search buffer
        ctx.deps.search_buffer = matching_items

        # Store the item IDs in the buffer for get_item_details
        ctx.deps.item_search_id_buffer = [item.get('item_id') for item in matching_items if 'item_id' in item]
        
        return f'Retrieved {len(matching_items)} items matching {query}'
    else:
        return {'error': f'No items found matching query: "{query}"'}

# Tool to make get_item API calls on item search id buffer
@datalab_agent.tool
async def get_item_details(ctx: RunContext[Deps], simplify_JSON: bool = True) -> str | dict[str, Any]:
    """Use the datalab get_item function to get details JSONs of specific items in the manifest.
    This function reads a list of item_ids and retrieves the corresponding items using the datalab API.
    item_query must be run first to generate the item search id buffer.
    """
    if ctx.deps.item_search_id_buffer == None:
        return {'error': 'No search id buffer found, run item_query first.'}
    
    if ctx.deps.items_buffer == None:
        ctx.deps.items_buffer = {}
    
    with ctx.deps.client(ctx.deps.datalab_url) as client:
        for item_id in ctx.deps.item_search_id_buffer:
            item = client.get_item(item_id=item_id)
            
            if simplify_JSON:
                item = process_item_for_chat(item)
            ctx.deps.items_buffer[item_id] = item
        return "Items retrieved."
    
    return {'error': 'No items found.'}

# For further development
# @datalab_agent.tool
# async def vector_search(ctx: RunContext[Deps], query: str, n_results: int = 5) -> str:
#     """Use chromadb to perform a vector search on the item manifest.
    
#     Args:
#         query: A string to search for in the item manifest
#         n_results: Number of top results to return (default: 5)
        
#     Returns:
#         String confirmation with number of results found
#     """
#     if ctx.deps.item_manifest is None:
#         return {'error': 'Use get_items to generate items list first'}
    
#     # Initialize the Chroma client (in-memory)
#     client = chromadb.Client()
    
#     # Create or get the collection
#     try:
#         collection = client.get_collection("items")
#     except:
#         collection = client.create_collection("items")
    
#     # Extract item IDs and text content from the manifest
#     item_ids = []
#     documents = []
    
#     for item in ctx.deps.item_manifest:
#         item_id = str(item.get('item_id', ''))
#         if not item_id:
#             continue
            
#         # Convert the item to a string representation for the document
#         item_text = json.dumps(item, ensure_ascii=False)
        
#         item_ids.append(item_id)
#         documents.append(item_text)
    
#     # Add items to the collection if there are any valid items
#     if item_ids and documents:
#         collection.add(
#             ids=item_ids,
#             documents=documents
#         )
    
#     # Perform a vector search
#     results = collection.query(
#         query_texts=[query],
#         n_results=min(n_results, len(item_ids))
#     )
    
#     if not results or not results['ids'][0]:
#         return {'error': f'No items found matching query: "{query}"'}
    
#     # Get the matching item IDs
#     matching_ids = results['ids'][0]
    
#     # Store results in buffers for further processing
#     matching_items = [item for item in ctx.deps.item_manifest 
#                      if str(item.get('item_id', '')) in matching_ids]
    
#     # Store the results in the search buffer
#     ctx.deps.search_buffer = matching_items
    
#     # Store the item IDs in the buffer for get_item_details
#     ctx.deps.item_search_id_buffer = [item.get('item_id') for item in matching_items if 'item_id' in item]
    
#     return f'Retrieved {len(matching_items)} items matching vector search for "{query}"'

# # Agent to ingest items
# ingestion_agent = Agent(
#     model,
#     system_prompt=INGEST_PROMPT.format(date.today()),
#     result_type=Response,
#     deps_type=Deps,
#     retries=2,
#     instrument=True,
# )

# class CreateItemInput(BaseModel):
#     item_id: str = Field(description="A short identifier for the item")
#     # item_type: str = Field(description="The type of the item") # default to 'samples'
#     name: str = Field(description="The name of the item")
#     chemform: str = Field(description="The chemical form of the item")
#     date: str = Field(description="Date of item creation, ISO formatted (YYYY-MM-DD)")
#     file: str = Field(description="File path to the associated file, if specified")


# @ingestion_agent.tool
# async def create_item(ctx: RunContext[Deps], inputs: CreateItemInput) -> str:
#     """
#     Use the datalab client to create a new datalab item.
#     """
#     if ctx.deps.datalab_api_key is None:
#         return {'error': 'No datalab API key found.'}
    
#     if ctx.deps.datalab_url is None:
#         return {'error': 'No datalab URL found.'}

#     with ctx.deps.client(ctx.deps.datalab_url) as client:
#         sample_dict = {
#             "name": inputs.name,
#             "chemform": inputs.chemform,
#             "date": inputs.date,
#         }
#         try:
#             created = client.create_item(
#                 item_id=inputs.item_id,
#                 item_type="samples",
#                 item_data=sample_dict
#             )
            
#             created_item_id = created["item_id"]

#             if inputs.file:
#                 # Check if the file exists
#                 if not os.path.exists(inputs.file):
#                     return {'error': f'Item {inputs.item_id} created but cannot find file {inputs.file}'}
                
#                 try:
#                     client.upload_file(
#                         item_id=created_item_id, file_path=inputs.file
#                     )
#                 except Exception as e:
#                     return {'error': f'Item {inputs.item_id} created but file upload failed: {str(e)}'}
            
#             return f"Item {inputs.item_id} created successfully with ID {created_item_id}."

#         except Exception as e:
#             return {'error': f'Error creating item: {str(e)}'}


# @datalab_agent.tool
# async def item_creator(ctx: RunContext[Deps], prompt: str) -> str:
#     """
#     This tool is used to create a new item in the datalab. 
#     It uses an agent to create an item specified by natural language.
    
#     Args:
#         str: A system prompt for the item creation agent
#     """
#     # Create an agent to create the item
#     result = await ingestion_agent.run(prompt, deps=ctx.deps)

#     return result.output
#     return result.output
# @datalab_agent.tool
# async def item_creator(ctx: RunContext[Deps], prompt: str) -> str:
#     """
#     This tool is used to create a new item in the datalab. 
#     It uses an agent to create an item specified by natural language.
    
#     Args:
#         str: A system prompt for the item creation agent
#     """
#     # Create an agent to create the item
#     result = await ingestion_agent.run(prompt, deps=ctx.deps)

#     return result.output
#     return result.output
