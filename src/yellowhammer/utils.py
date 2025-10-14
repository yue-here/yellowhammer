import os
import re
from pathlib import Path
from typing import Tuple, List

def parse_file_paths(prompt: str) -> Tuple[str, List[str]]:
    """
    Parse a prompt string to extract file paths and return cleaned prompt with paths list.
    
    Matches paths that:
    - Have common file extensions (jpg, jpeg, png, gif, bmp, pdf, txt, etc.)
    - Can be absolute or relative paths
    - Can contain common path characters (letters, numbers, underscores, dots, slashes, dashes)
    
    Args:
        prompt: The input prompt string that may contain file paths
        
    Returns:
        Tuple of (cleaned_prompt, list_of_paths)
    """
    # Pattern matches file paths with extensions
    # Supports: alphanumeric, underscores, dots, forward/back slashes, dashes, colons (for absolute paths)
    # This pattern is more permissive to capture complex paths like /var/folders/...
    file_path_pattern = r'(?:^|(?<=\s))([A-Za-z0-9_.:~/\\\-]+\.(?:jpg|jpeg|png|gif|bmp|tiff|pdf|txt|doc|docx|xls|xlsx|csv|json|xml|raw|xrdml|xy))(?:(?=\s)|(?=[?!,;.])|$)'
    
    # Find all file paths
    matches = re.finditer(file_path_pattern, prompt, re.IGNORECASE)
    paths = [match.group(1) for match in matches]
    
    # Remove file paths from the prompt
    cleaned_prompt = re.sub(file_path_pattern, '', prompt, flags=re.IGNORECASE)
    
    # Clean up extra whitespace
    cleaned_prompt = re.sub(r'\s+', ' ', cleaned_prompt).strip()
    
    return cleaned_prompt, paths


def verify_and_load_image(file_path: str) -> dict | None:
    """
    Verify a file path exists and if it's an image, load it as BinaryContent.
    
    Args:
        file_path: Path to the file (relative or absolute)
        
    Returns:
        Dictionary with 'type' and 'content' keys if successful image, None otherwise
        - For images: {'type': 'image', 'content': BinaryContent object, 'path': str}
        - For non-images: {'type': 'file', 'path': str} (for future extension)
        - For non-existent files: None
    """
    from pydantic_ai import BinaryContent
    
    # Supported image extensions
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    
    # Convert to Path object for easier manipulation
    path = Path(file_path)
    
    # Check if file exists
    if not path.exists():
        return None
    
    # Get file extension
    extension = path.suffix.lower()
    
    # Check if it's an image
    if extension in IMAGE_EXTENSIONS:
        try:
            # Read file content
            data = path.read_bytes()
            
            # Map extension to media type
            media_type_map = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.gif': 'image/gif',
                '.bmp': 'image/bmp',
                '.tiff': 'image/tiff',
                '.webp': 'image/webp'
            }
            
            media_type = media_type_map.get(extension, 'image/jpeg')
            
            # Create BinaryContent for the image
            binary_content = BinaryContent(data=data, media_type=media_type)
            
            return {
                'type': 'image',
                'content': binary_content,
                'path': str(path.absolute())
            }
        except Exception as e:
            print(f"Error reading image file {file_path}: {e}")
            return None
    else:
        # For non-image files, return file info for future extension
        return {
            'type': 'file',
            'path': str(path.absolute())
        }


def process_prompt_with_images(prompt: str) -> Tuple[List, List[str]]:
    """
    Process a prompt to extract images and construct a message list for the LLM.
    
    This function:
    1. Parses the prompt for file paths
    2. Verifies file existence
    3. Loads images as BinaryContent
    4. Constructs a message list suitable for pydantic-ai agents
    
    Args:
        prompt: The user's prompt string, potentially containing file paths
        
    Returns:
        Tuple of (message_list, warnings)
        - message_list: List suitable for passing to agent.run() or agent.run_sync()
        - warnings: List of warning messages about missing or non-image files
    """
    # Parse file paths from prompt
    cleaned_prompt, file_paths = parse_file_paths(prompt)
    
    # Initialize message list with the cleaned text prompt
    messages = [cleaned_prompt] if cleaned_prompt else []
    warnings = []
    
    # Process each detected file path
    for file_path in file_paths:
        result = verify_and_load_image(file_path)
        
        if result is None:
            warnings.append(f"File not found: {file_path}")
        elif result['type'] == 'image':
            # Add image to messages
            messages.append(result['content'])
        elif result['type'] == 'file':
            # Non-image file - skip for now but could be extended later
            warnings.append(f"File '{file_path}' is not an image format (skipped for now)")
    
    return messages, warnings


def process_item_for_chat(item: dict) -> dict:
    """Process an item to make it more suitable for LLM consumption by removing large fields
    and simplifying the structure.
    """
    import copy
    import json
    
    # Work on a copy to avoid modifying the original
    item_info = copy.deepcopy(item)
    
    # Extract filenames for easier reference
    item_filenames = {}
    if "files" in item_info:
        for file in item_info.get("files", []):
            if "immutable_id" in file and "name" in file:
                item_filenames[str(file["immutable_id"])] = file["name"]
    
    # Keys to remove from the top level
    top_level_keys_to_remove = [
        "display_order", "creator_ids", "refcode", "last_modified",
        "revision", "revisions", "immutable_id", "file_ObjectIds"
    ]
    
    for k in top_level_keys_to_remove:
        item_info.pop(k, None)
    
    # Process blocks if they exist
    big_data_keys = ["bokeh_plot_data", "b64_encoded_image"]
    
    if "blocks_obj" in item_info:
        blocks = []
        for block_id, block in item_info.get("blocks_obj", {}).items():
            # Skip chat blocks
            if block.get("blocktype") == "chat":
                continue
                
            # Remove large or unnecessary fields
            block_fields_to_remove = ["item_id", "block_id", "collection_id"] + big_data_keys
            for field in block_fields_to_remove:
                block.pop(field, None)
            
            # Replace file_id with actual filename
            file_id = block.pop("file_id", None)
            if file_id and file_id in item_filenames:
                block["file"] = item_filenames[file_id]
                
            blocks.append(block)
        
        # Replace blocks_obj with a simpler list
        item_info.pop("blocks_obj", None)
        item_info["blocks"] = blocks
    
    # Simplify relationships
    if "relationships" in item_info:
        for i, relationship in enumerate(item_info["relationships"]):
            item_info["relationships"][i] = {
                k: v for k, v in relationship.items() 
                if k in ["item_id", "type", "relation"]
            }
    
    # Simplify files to just names
    if "files" in item_info:
        item_info["files"] = [file["name"] for file in item_info.get("files", []) if "name" in file]
    
    # Simplify creators to just names
    if "creators" in item_info:
        item_info["creators"] = [
            creator.get("display_name") for creator in item_info.get("creators", [])
            if "display_name" in creator
        ]
    
    # Process quantity fields in various constituent types
    for key in ["synthesis_constituents", "positive_electrode", "negative_electrode", "electrolyte"]:
        if key in item_info:
            for constituent in item_info[key]:
                if "quantity" in constituent and "unit" in constituent:
                    constituent["quantity"] = f"{constituent.get('quantity', 'unknown')} {constituent.get('unit', '')}"
                    constituent.pop("unit", None)
    
    # Remove empty values
    item_info = {k: v for k, v in item_info.items() if v}
    
    return item_info