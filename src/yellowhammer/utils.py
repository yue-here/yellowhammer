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