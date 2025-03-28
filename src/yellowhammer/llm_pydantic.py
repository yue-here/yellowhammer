import os
import json
from typing import Any, cast
from dataclasses import dataclass

from datalab_api import DatalabClient

from typing import Annotated, Any, Union
from typing_extensions import TypeAlias

from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models import KnownModelName
from .prompt import API_PROMPT, SYSTEM_PROMPT


# Dependencies
@dataclass
class Deps:
    client: DatalabClient
    datalab_api_key: str | None
    datalab_url: str | None
    item_manifest: dict[str, Any] | None = None
    block_manifest: dict[str, Any] | None = None
    item_search_id_buffer: dict[str, Any] | None = None
    items_buffer: dict[str, Any] | None = None

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
    retries=2
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
            return "Items retrieved."
        
    # Set 2 retries in decorator to prevent looping
    raise ModelRetry('No data returned, try again.')


@datalab_agent.tool
async def inspect_blocks(ctx: RunContext[Deps]) -> str:
    """Inspect retrieved blocks in the block manifest.
    """

    if ctx.deps.block_manifest == None:
        with ctx.deps.client(ctx.deps.datalab_url) as client:
            ctx.deps.block_manifest = client.get_block_info
    
    # Blocks buffer to plaintext
    blocks_buffer = json.dumps(ctx.deps.block_manifest, indent=2)
    
    r = await datalab_agent.run(blocks_buffer, deps=ctx.deps)
    
    return r.data

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
def _search_nested(obj, query, case_sensitive=False):
    """Recursively search through nested dictionaries and lists for a query.
    
    Args:
        obj: The object to search (dict, list, or primitive value)
        query: The string to search for
        case_sensitive: Whether the search should be case-sensitive
        
    Returns:
        bool: True if the query was found, False otherwise
    """
    if isinstance(obj, dict):
        # Look through all keys and values
        return any(_search_nested(k, query, case_sensitive) for k in obj.keys()) or \
               any(_search_nested(v, query, case_sensitive) for v in obj.values())
    elif isinstance(obj, list):
        # Look through all items in the list
        return any(_search_nested(item, query, case_sensitive) for item in obj)
    elif isinstance(obj, str):
        # String comparison
        if case_sensitive:
            return query in obj
        else:
            return query.lower() in obj.lower()
    elif obj is not None:
        # For numbers or other types, convert to string and check
        return str(query) == str(obj)
    return False

# Tool to query item manifest if it exists
@datalab_agent.tool
async def item_query(ctx: RunContext[Deps], query: str) -> list[dict[str, Any]]:
    """Use the datalab search function to search through the item manifest at all nesting levels.
    Generates a list of item_ids that can be used with get_item_details.
    
    Args:
        query: Search string to find in any field at any nesting level.

    Returns:
        List of dictionaries each representing one matching item
    """
    if ctx.deps.item_manifest is None:
        return {'error': 'Use get_items to generate items list first'}
        
    # Search all items, looking at all levels of nesting
    matching_items = [
        item for item in ctx.deps.item_manifest 
        if _search_nested(item, query)
    ]
    
    if matching_items:
        ctx.deps.item_search_id_buffer = [item.get('item_id') for item in matching_items if 'item_id' in item]
        return matching_items
    else:
        return {'error': f'No items found containing "{query}"'}

# Tool to make get_item API calls on item search id buffer
@datalab_agent.tool
async def get_item_details(ctx: RunContext[Deps]) -> dict[str, Any]:
    """Use the datalab get_item function to get details JSONs of specific items in the manifest.
    This function reads a list of item_ids and retrieves the corresponding items using the datalab API.
    item_query must be run first to generate the item search id buffer.
    """
    if ctx.deps.item_search_id_buffer == None:
        return {'error': 'No item search buffer found, run item_query first.'}
    
    if ctx.deps.items_buffer == None:
        ctx.deps.items_buffer = {}
    
    with ctx.deps.client(ctx.deps.datalab_url) as client:
        for item_id in ctx.deps.item_search_id_buffer:
            item = client.get_item(item_id=item_id)
            ctx.deps.items_buffer[item_id] = item
        return "Items retrieved."
    
    return {'error': 'No items found.'}

# Ask agent to inspect items in the items buffer
@datalab_agent.tool
async def inspect_items(ctx: RunContext[Deps]) -> str:
    """Inspect retrieved items in the items buffer.
    Only use this if the items buffer has been populated with get_item_details!
    """

    if ctx.deps.items_buffer == None:
        return {'error': 'No items buffer found, run get_item_details first.'}

    # Items buffer to plaintext
    items_buffer = json.dumps(ctx.deps.items_buffer, indent=2)
    
    r = await datalab_agent.run(items_buffer, deps=ctx.deps)
    
    return r.data


# Validation may be helpful for some outputs
# @datalab_agent.result_validator
# async def validate_result(ctx: RunContext[Deps], result: Response) -> Response:
#     # Do some validation here