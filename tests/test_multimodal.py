"""
Tests for multimodal (image) support utility functions.
"""
import os
import tempfile
from pathlib import Path
from yellowhammer.utils import parse_file_paths, verify_and_load_image, process_prompt_with_images


def test_parse_file_paths():
    """Test parsing file paths from prompts."""
    # Test with single image
    prompt1 = "What do you see in this image: IMG_1648.jpg"
    cleaned, paths = parse_file_paths(prompt1)
    assert cleaned == "What do you see in this image:"
    assert "IMG_1648.jpg" in paths
    
    # Test with multiple files
    prompt2 = "Compare test.png and data.csv files"
    cleaned, paths = parse_file_paths(prompt2)
    assert "test.png" in paths
    assert "data.csv" in paths
    
    # Test with paths
    prompt3 = "Analyze examples/IMG_1648.jpg and other_dir/file.pdf"
    cleaned, paths = parse_file_paths(prompt3)
    assert "examples/IMG_1648.jpg" in paths
    assert "other_dir/file.pdf" in paths
    
    # Test with no files
    prompt4 = "Just a simple question"
    cleaned, paths = parse_file_paths(prompt4)
    assert cleaned == "Just a simple question"
    assert len(paths) == 0


def test_verify_and_load_image():
    """Test image verification and loading."""
    # Test with non-existent file
    result = verify_and_load_image("non_existent_file.jpg")
    assert result is None
    
    # Create a temporary image file for testing
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        # Write minimal PNG header
        tmp_file.write(b'\x89PNG\r\n\x1a\n')
        tmp_path = tmp_file.name
    
    try:
        # Test with existing image
        result = verify_and_load_image(tmp_path)
        assert result is not None
        assert result['type'] == 'image'
        assert 'content' in result
        assert 'path' in result
        
        # Verify BinaryContent properties
        from pydantic_ai import BinaryContent
        assert isinstance(result['content'], BinaryContent)
        assert result['content'].media_type == 'image/png'
    finally:
        # Clean up
        os.unlink(tmp_path)
    
    # Create a temporary non-image file
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp_file:
        tmp_file.write(b'test content')
        tmp_path = tmp_file.name
    
    try:
        # Test with non-image file
        result = verify_and_load_image(tmp_path)
        assert result is not None
        assert result['type'] == 'file'
        assert 'path' in result
    finally:
        os.unlink(tmp_path)


def test_process_prompt_with_images():
    """Test complete prompt processing with images."""
    # Test with no images
    messages, warnings = process_prompt_with_images("Simple question without files")
    assert len(messages) == 1
    assert messages[0] == "Simple question without files"
    assert len(warnings) == 0
    
    # Test with non-existent file
    messages, warnings = process_prompt_with_images("Check this file: missing.jpg")
    assert len(messages) == 1
    assert "Check this file:" in messages[0]
    assert len(warnings) == 1
    assert "File not found" in warnings[0]
    
    # Create a temporary image for testing
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
        tmp_file.write(b'\xff\xd8\xff')  # JPEG header
        tmp_path = tmp_file.name
    
    try:
        # Test with existing image
        messages, warnings = process_prompt_with_images(f"What is in {tmp_path}?")
        assert len(messages) == 2  # Text + image
        assert "What is in" in messages[0]
        
        # Verify second message is BinaryContent
        from pydantic_ai import BinaryContent
        assert isinstance(messages[1], BinaryContent)
        assert messages[1].media_type == 'image/jpeg'
        
        assert len(warnings) == 0
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    test_parse_file_paths()
    test_verify_and_load_image()
    test_process_prompt_with_images()
    print("✅ All tests passed!")
