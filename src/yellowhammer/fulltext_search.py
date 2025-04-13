import json
from typing import List, Dict, Any, Optional, Union
from whoosh.index import create_in
from whoosh.fields import Schema, TEXT, ID, KEYWORD
from whoosh.qparser import QueryParser, MultifieldParser, OrGroup
from whoosh.analysis import StemmingAnalyzer
from whoosh.filedb.filestore import RamStorage

class FullTextSearchTool:
    def __init__(self, samples: List[Dict]):
        self.samples = samples
        self.storage = RamStorage()
        self._build_index()
        
    def _extract_nested_values(self, sample: Dict) -> Dict[str, str]:
        """Extract values from nested structures for indexing"""
        extracted = {}
        
        # Extract block titles and types
        if 'blocks' in sample and isinstance(sample['blocks'], list):
            block_titles = []
            block_types = []
            for block in sample['blocks']:
                if 'title' in block:
                    block_titles.append(block['title'])
                if 'blocktype' in block:
                    block_types.append(block['blocktype'])
            
            extracted['block_titles'] = ' '.join(block_titles)
            extracted['block_types'] = ' '.join(block_types)
        
        # Extract creator names
        if 'creators' in sample and isinstance(sample['creators'], list):
            creator_names = []
            for creator in sample['creators']:
                if 'display_name' in creator:
                    creator_names.append(creator['display_name'])
            
            extracted['creator_names'] = ' '.join(creator_names)
        
        return extracted
        
    def _build_index(self):
        """Build a full-text search index from the samples with field-specific indexing in memory"""
        # Create a more comprehensive schema
        analyzer = StemmingAnalyzer()
        schema = Schema(
            id=ID(stored=True),
            content=TEXT(analyzer=analyzer),
            name=TEXT(stored=True),
            formula=TEXT(stored=True),
            item_id=ID(stored=True),
            type=KEYWORD(stored=True, commas=True),
            date=ID(stored=True),
            creator_names=TEXT(stored=True),
            block_titles=TEXT(stored=True),
            block_types=KEYWORD(stored=True, commas=True)
        )
        
        # Create in-memory index
        ix = self.storage.create_index(schema)
        writer = ix.writer()
        
        # Add documents to index
        for i, sample in enumerate(self.samples):
            # Convert sample to string for indexing
            content = json.dumps(sample)
            
            # Extract values from nested structures
            nested_values = self._extract_nested_values(sample)
            
            writer.add_document(
                id=str(i),
                content=content,
                name=sample.get('name', ''),
                formula=sample.get('characteristic_chemical_formula', ''),
                item_id=sample.get('item_id', ''),
                type=sample.get('type', ''),
                date=sample.get('date', ''),
                creator_names=nested_values.get('creator_names', ''),
                block_titles=nested_values.get('block_titles', ''),
                block_types=nested_values.get('block_types', '')
            )
            
        writer.commit()
    
    def search(self, query_text: str, fields=None, limit: int = 10) -> List[Dict]:
        """
        Search for samples matching the query text
        
        Args:
            query_text: Text to search for
            fields: List of specific fields to search in (None means search all fields)
            limit: Maximum number of results to return
            
        Returns:
            List of matching samples
        """
        # Get the in-memory index
        ix = self.storage.open_index()
        
        with ix.searcher() as searcher:
            if fields and isinstance(fields, list):
                # Search in specific fields with OR between fields
                parser = MultifieldParser(fields, ix.schema, group=OrGroup)
            else:
                # Default to searching in content field (which has everything)
                parser = QueryParser("content", ix.schema)
            
            # Make query more flexible by wrapping terms in wildcards if simple term
            if len(query_text.split()) == 1 and not any(c in query_text for c in ':"*?'):
                query_text = f"*{query_text}*"
                
            # Parse the query
            query = parser.parse(query_text)
            
            # Search
            results = searcher.search(query, limit=limit)
            
            # Get the original samples
            matching_samples = []
            for result in results:
                sample_id = int(result["id"])
                matching_samples.append(self.samples[sample_id])
                
            return matching_samples
    
    def search_by_field(self, field_queries: Dict[str, str], limit: int = 10) -> List[Dict]:
        """
        Search using field-specific queries
        
        Args:
            field_queries: Dictionary mapping field names to query strings
            limit: Maximum number of results to return
            
        Returns:
            List of matching samples
        """
        # Get the in-memory index
        ix = self.storage.open_index()
        
        with ix.searcher() as searcher:
            query_parts = []
            
            for field, query_text in field_queries.items():
                parser = QueryParser(field, ix.schema)
                query_parts.append(parser.parse(query_text))
            
            # Combine all query parts (implicit AND)
            from whoosh.query import And
            if query_parts:
                combined_query = And(query_parts)
                results = searcher.search(combined_query, limit=limit)
                
                matching_samples = []
                for result in results:
                    sample_id = int(result["id"])
                    matching_samples.append(self.samples[sample_id])
                    
                return matching_samples
            
            return []

    def natural_language_search(self, query_text: str, limit: int = 100) -> List[Dict]:
        """
        Search for samples using natural language query.
        This method is optimized for use with the yellowhammer agent.
        
        Args:
            query_text: Natural language query string
            limit: Maximum number of results to return
            
        Returns:
            List of matching samples
        """
        # Parse the query to determine which fields might be relevant
        field_queries = enhance_fulltext_query(query_text)
        
        if "content" in field_queries and len(field_queries) == 1:
            # If only searching general content, use normal search
            return self.search(field_queries["content"], limit=limit)
        else:
            # Otherwise use field-specific search
            return self.search_by_field(field_queries, limit=limit)

    def update_samples(self, new_samples: List[Dict]):
        """
        Update the search index with new samples
        
        Args:
            new_samples: New list of samples to index
        """
        self.samples = new_samples
        self.storage = RamStorage()  # Create a fresh storage
        self._build_index()  # Rebuild the index

# Improve the enhance_fulltext_query function to better handle natural language
def enhance_fulltext_query(query: str, data_sample: Dict = None) -> Dict[str, str]:
    """
    Parse a natural language query into field-specific queries.
    This is a simple implementation that could be replaced with an LLM call.
    
    Args:
        query: Natural language query
        data_sample: Sample of the data structure (to help understand schema)
        
    Returns:
        Dictionary of field queries
    """
    query = query.lower()
    field_queries = {}
    
    # Simple keyword mapping
    if "formula" in query or "chemical" in query:
        field_queries["formula"] = query.replace("formula", "").replace("chemical", "").strip()
    
    if "name" in query:
        field_queries["name"] = query.replace("name", "").strip()
        
    if "creator" in query or "author" in query or "by" in query:
        field_queries["creator_names"] = query.replace("creator", "").replace("author", "").replace("by", "").strip()
        
    if "type" in query:
        field_queries["type"] = query.replace("type", "").strip()
        
    if "date" in query:
        field_queries["date"] = query.replace("date", "").strip()
        
    if "id" in query or "item_id" in query:
        field_queries["item_id"] = query.replace("id", "").replace("item_id", "").strip()
    
    # If no specific fields detected or very short query, search all content
    if not field_queries or len(query.split()) <= 3:
        field_queries["content"] = query
        
    return field_queries
