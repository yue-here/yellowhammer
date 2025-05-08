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

# Dependencies
@dataclass
class Deps:
    client: DatalabClient
    datalab_api_key: str | None
    datalab_url: str | None
    item_manifest: dict[str, Any] | None = None
    # block_manifest: dict[str, Any] | None = None
    item_search_id_buffer: dict[str, Any] | None = None
    items_buffer: dict[str, Any] | None = None
    search_buffer: list[dict] | None = None
    search_tool: FullTextSearchTool | None = None

# Output schemas
class TextResponse(BaseModel):
    text: str = Field(description="A short conversational response")

class CodeResponse(BaseModel):
    text: str = Field(description="A short description of the code")
    code: str = Field(description="A code snippet")

Response: TypeAlias = Union[TextResponse, CodeResponse]

# Define agent and preferred model
default_model = 'anthropic:claude-3-5-haiku-latest'
model = cast(KnownModelName, os.getenv('AI_MODEL', default_model)) # let user set the model
datalab_agent = Agent(
    model,
    system_prompt=SYSTEM_PROMPT.format(API_PROMPT), # TODO: fix prompts
    result_type=Response,
    deps_type=Deps,
    retries=2,
    instrument=True,
)

# Define tools
# Can we add an option to return items of all types in the API call?
@datalab_agent.tool(retries=2)
async def get_items(ctx: RunContext[Deps], item_type: str) -> str:
    """Use the datalab get_items function to get a JSON of all items in the database.

    Args:
        ctx: The context.
        item_type: Usually 'samples'
    """
    # TODO check API key when the agent is initialized
    # if ctx.deps.datalab_api_key is None:
    #     # if no API key is provided, say so
    #     return {'error': 'No datalab API key found.'}
    
    if ctx.deps.item_manifest == None:
        with ctx.deps.client(ctx.deps.datalab_url) as client:
            item_type = "samples"
            ctx.deps.item_manifest = client.get_items(item_type=item_type)
            # Initialize the search tool with the fetched items
            ctx.deps.search_tool = FullTextSearchTool(ctx.deps.item_manifest)
            return "Items retrieved."
        
    # Set 2 retries in decorator to prevent looping
    raise ModelRetry('No data returned, try again.')


# @datalab_agent.tool
# async def inspect_blocks(ctx: RunContext[Deps]) -> str:
#     """Inspect retrieved blocks in the block manifest.
#     """

#     if ctx.deps.block_manifest == None:
#         with ctx.deps.client(ctx.deps.datalab_url) as client:
#             ctx.deps.block_manifest = client.get_block_info
    
#     # Blocks buffer to plaintext
#     blocks_buffer = json.dumps(ctx.deps.block_manifest, indent=2)
    
#     r = await datalab_agent.run(blocks_buffer, deps=ctx.deps)
    
#     return r.data

# # Tool to identify blocks present in the current datalab - may not be necessary
# @datalab_agent.tool
# async def block_query(ctx: RunContext[Deps], query: str) -> list[dict[str, Any]]: 
#     """Find blocks that are present in the current datalab."
    
#     Args:
#         query: Search the block list for this query string.

#     Returns:
#         List of dicts for each block matching the query 
#     """
#     if ctx.deps.block_manifest == None:
#         with ctx.deps.client(ctx.deps.datalab_url) as client:
#             if client.get_block_info() == None:
#                 return {'error': 'No blocks found.'}
#             ctx.deps.block_manifest = client.get_block_info()
    
#     # Check if blocks with the query exist and return them in a list
#     if any(item.get('id') == query for item in ctx.deps.block_manifest):
#         return [
#             item for item in ctx.deps.block_manifest 
#             if any(block.get('blocktype') == query for block in item.get('blocks', []))
#                 ]
#     else:
#         return {'error': 'Queried block not found.'}
    
# Helper function to search through nested dicts/lists (the item manifest)
# def _search_nested(obj, query, case_sensitive=False):
#     """Recursively search through nested dictionaries and lists for a query.
    
#     Args:
#         obj: The object to search (dict, list, or primitive value)
#         query: The string to search for
#         case_sensitive: Whether the search should be case-sensitive
        
#     Returns:
#         bool: True if the query was found, False otherwise
#     """
#     if isinstance(obj, dict):
#         # Look through all keys and values
#         return any(_search_nested(k, query, case_sensitive) for k in obj.keys()) or \
#                any(_search_nested(v, query, case_sensitive) for v in obj.values())
#     elif isinstance(obj, list):
#         # Look through all items in the list
#         return any(_search_nested(item, query, case_sensitive) for item in obj)
#     elif isinstance(obj, str):
#         # String comparison
#         if case_sensitive:
#             return query in obj
#         else:
#             return query.lower() in obj.lower()
#     elif obj is not None:
#         # For numbers or other types, convert to string and check
#         return str(query) == str(obj)
#     return False

# Tool to query item manifest if it exists
@datalab_agent.tool
async def item_query(ctx: RunContext[Deps], query: str) -> list[dict[str, Any]]:
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
async def get_item_details(ctx: RunContext[Deps], simplify_JSON:bool = True) -> dict[str, Any]:
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

# Ask agent to inspect items in the items buffer
@datalab_agent.tool
async def inspect_items(ctx: RunContext[Deps], prompt: str) -> str:
    """
    If search item details have been successfully retrieved with a get_item_details call,
    this tool uses an LLM agent to inspect these items. Prompt this agent based on the user's request.
    The user does not see this tool's response so incorporate outputs into your final repsonse to the user. 

    Args:
        prompt: A system prompt for the inspection agent
    """
    if ctx.deps.items_buffer == None:
        return {'error': 'No items buffer found, run get_item_details first.'}

    # Items buffer to plaintext
    items_buffer = json.dumps(ctx.deps.items_buffer, separators=(',', ':'))
    
    # Create an agent to inspect the items
    inspection_agent = Agent(model, system_prompt=prompt, instrument=True)
    result = await inspection_agent.run(items_buffer, deps=ctx.deps)
    
    return result.data

# Add the inspect_query function after inspect_items
@datalab_agent.tool
async def inspect_query(ctx: RunContext[Deps], prompt: str) -> str:
    """
    If a search query has been successfully run with a item_query call,
    this tool uses an LLM agent to inspect the search buffer. Prompt this agent based on the user's request.
    item_query must be run first to generate the search buffer.

    Args:
        prompt: A system prompt for the inspection agent
    """
    if ctx.deps.search_buffer == None:
        return {'error': 'No search buffer found, run item_query first.'}
    # Search buffer to plaintext
    search_buffer = json.dumps(ctx.deps.search_buffer, separators=(',', ':'))
    inspection_agent = Agent(model, system_prompt=prompt, instrument=True)
    result = await inspection_agent.run(search_buffer, deps=ctx.deps)
    return result.data

@datalab_agent.tool
async def code_writer(ctx: RunContext[Deps], prompt: str) -> Response:
    """
    This tool is used to write code based on a prompt. It has information about the datalab API syntax.
    Args:
        prompt: A system prompt for the code writer agent 
    """
    code_agent = Agent(model, result_type = Response, system_prompt=CODE_PROMPT, instrument=True)
    result = await code_agent.run(prompt, deps=ctx.deps)
    return result.data

# Requires further development
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

# Separate agent to ingest items
ingestion_agent = Agent(
    model,
    system_prompt=INGEST_PROMPT.format(date.today()),
    result_type=Response,
    deps_type=Deps,
    retries=2,
    instrument=True,
)

class CreateItemInput(BaseModel):
    item_id: str = Field(description="A short identifier for the item")
    # item_type: str = Field(description="The type of the item") # default to 'samples'
    name: str = Field(description="The name of the item")
    chemform: str = Field(description="The chemical form of the item")
    date: str = Field(description="Date of item creation, ISO formatted (YYYY-MM-DD)")
    file: str = Field(description="File path to the associated file, if specified")


@ingestion_agent.tool
async def create_item(ctx: RunContext[Deps], inputs: CreateItemInput) -> str:
    """
    Use the datalab client to create a new datalab item.
    """
    if ctx.deps.datalab_api_key is None:
        return {'error': 'No datalab API key found.'}
    
    if ctx.deps.datalab_url is None:
        return {'error': 'No datalab URL found.'}

    with ctx.deps.client(ctx.deps.datalab_url) as client:
        sample_dict = {
            "name": inputs.name,
            "chemform": inputs.chemform,
            "date": inputs.date,
        }
        try:
            created = client.create_item(
                item_id=inputs.item_id,
                item_type="samples",
                item_data=sample_dict
            )
            
            created_item_id = created["item_id"]

            if inputs.file:
                # Check if the file exists
                if not os.path.exists(inputs.file):
                    return {'error': f'Item {inputs.item_id} created but cannot find file {inputs.file}'}
                
                try:
                    client.upload_file(
                        item_id=created_item_id, file_path=inputs.file
                    )
                except Exception as e:
                    return {'error': f'Item {inputs.item_id} created but file upload failed: {str(e)}'}
            
            return f"Item {inputs.item_id} created successfully with ID {created_item_id}."

        except Exception as e:
            return {'error': f'Error creating item: {str(e)}'}


@datalab_agent.tool
async def item_creator(ctx: RunContext[Deps], prompt: str) -> str:
    """
    This tool is used to create a new item in the datalab. 
    It uses an agent to create an item specified by natural language.
    
    Args:
        str: A system prompt for the item creation agent
    """
    # Create an agent to create the item
    result = await ingestion_agent.run(prompt, deps=ctx.deps)

    return result.data
