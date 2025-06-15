"""Tests for core modules."""

import pytest

from src.core.document_processor import (
    Block,
    BlockType,
    Chunk,
    ChunkingConfig,
    DocumentPipeline,
    DocumentProcessor,
    Tokenizer,
)


class TestTokenizer:
    """Test tokenizer functionality."""

    def test_count_tokens(self):
        """Test token counting."""
        tokenizer = Tokenizer()
        text = "Hello world! This is a test."
        count = tokenizer.count(text)
        assert count > 0
        assert isinstance(count, int)

    def test_split_sentences(self):
        """Test sentence splitting."""
        tokenizer = Tokenizer()
        text = "First sentence. Second sentence! Third sentence?"
        sentences = tokenizer.split_sentences(text)
        assert len(sentences) == 3
        assert sentences[0] == "First sentence."
        assert sentences[1] == "Second sentence!"
        assert sentences[2] == "Third sentence?"

    def test_truncate_to_tokens(self):
        """Test token truncation."""
        tokenizer = Tokenizer()
        text = "This is a long text that should be truncated to fewer tokens."
        truncated = tokenizer.truncate_to_tokens(text, max_tokens=10)
        truncated_count = tokenizer.count(truncated)
        assert truncated_count <= 10


class TestDocumentProcessor:
    """Test document processing."""

    def test_process_markdown_text(self):
        """Test processing markdown text."""
        processor = DocumentProcessor()
        text = """# Heading 1

This is a paragraph.

## Heading 2

Another paragraph with some content.

```python
def example():
    return "code"
```

- List item 1
- List item 2

> This is a quote
"""
        blocks = processor.process_text(text, "md")
        
        # Check that we have different block types
        block_types = [block.type for block in blocks]
        assert BlockType.HEADING in block_types
        assert BlockType.PARAGRAPH in block_types
        assert BlockType.CODE in block_types
        assert BlockType.LIST in block_types
        assert BlockType.QUOTE in block_types

    def test_clean_blocks(self):
        """Test block cleaning."""
        processor = DocumentProcessor()
        blocks = [
            Block(BlockType.PARAGRAPH, "  Multiple   spaces  "),
            Block(BlockType.PARAGRAPH, ""),
            Block(BlockType.PARAGRAPH, "Normal text"),
        ]
        
        cleaned = processor.clean_blocks(blocks)
        assert len(cleaned) == 2  # Empty block removed
        assert cleaned[0].text == "Multiple spaces"
        assert cleaned[1].text == "Normal text"

    def test_blocks_to_sections(self):
        """Test converting blocks to sections."""
        processor = DocumentProcessor()
        blocks = [
            Block(BlockType.HEADING, "Chapter 1", metadata={"level": 1}),
            Block(BlockType.PARAGRAPH, "First paragraph"),
            Block(BlockType.HEADING, "Section 1.1", metadata={"level": 2}),
            Block(BlockType.PARAGRAPH, "Second paragraph"),
            Block(BlockType.HEADING, "Chapter 2", metadata={"level": 1}),
            Block(BlockType.PARAGRAPH, "Third paragraph"),
        ]
        
        sections = processor.blocks_to_sections(blocks)
        assert len(sections) == 3
        assert sections[0].path == ["Chapter 1"]
        assert sections[1].path == ["Chapter 1", "Section 1.1"]
        assert sections[2].path == ["Chapter 2"]


class TestChunker:
    """Test chunking functionality."""

    def test_chunk_section(self, document_pipeline, sample_text):
        """Test chunking a section."""
        chunks, stats = await document_pipeline.process_document(
            sample_text, "test_doc", "md"
        )
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, Chunk) for chunk in chunks)
        assert all(chunk.doc_id == "test_doc" for chunk in chunks)
        assert stats["total_chunks"] == len(chunks)

    def test_chunk_overlap(self, chunking_config):
        """Test chunk overlap functionality."""
        from src.core.document_processor import Chunker, Section, Tokenizer
        
        tokenizer = Tokenizer()
        chunker = Chunker(tokenizer, chunking_config)
        
        # Create a section with multiple blocks
        blocks = [
            Block(BlockType.PARAGRAPH, "First paragraph with some content."),
            Block(BlockType.PARAGRAPH, "Second paragraph with more content."),
            Block(BlockType.PARAGRAPH, "Third paragraph with additional content."),
        ]
        section = Section("test_section", ["Test"], blocks)
        
        chunks = chunker.chunk_section(section, "test_doc")
        
        # Check that chunks have proper linking
        if len(chunks) > 1:
            assert chunks[0].next_chunk_id == chunks[1].id
            assert chunks[1].prev_chunk_id == chunks[0].id

    def test_dynamic_budgeting(self, chunking_config):
        """Test dynamic token budgeting."""
        from src.core.document_processor import Chunker, Tokenizer
        
        tokenizer = Tokenizer()
        chunker = Chunker(tokenizer, chunking_config)
        
        # Test with different block types
        code_blocks = [Block(BlockType.CODE, "print('hello world')")]
        paragraph_blocks = [Block(BlockType.PARAGRAPH, "This is regular text.")]
        
        code_budget = chunker._calculate_dynamic_budget(code_blocks)
        paragraph_budget = chunker._calculate_dynamic_budget(paragraph_blocks)
        
        # Code should have smaller budget due to lower weight
        assert code_budget <= paragraph_budget


class TestDocumentPipeline:
    """Test complete document pipeline."""

    @pytest.mark.asyncio
    async def test_process_document(self, document_pipeline, sample_text):
        """Test complete document processing."""
        chunks, stats = await document_pipeline.process_document(
            sample_text, "test_doc", "md"
        )
        
        assert len(chunks) > 0
        assert stats["total_chunks"] > 0
        assert stats["total_sections"] > 0
        assert stats["total_blocks"] > 0
        assert stats["total_tokens"] > 0
        assert stats["avg_chunk_tokens"] > 0

    @pytest.mark.asyncio
    async def test_process_with_metadata(self, document_pipeline):
        """Test processing with metadata."""
        metadata = {"source": "test", "author": "tester"}
        chunks, stats = await document_pipeline.process_document(
            "Test content", "test_doc", "txt", metadata
        )
        
        for chunk in chunks:
            assert chunk.metadata["source"] == "test"
            assert chunk.metadata["author"] == "tester"

    @pytest.mark.asyncio
    async def test_process_empty_content(self, document_pipeline):
        """Test processing empty content."""
        chunks, stats = await document_pipeline.process_document(
            "", "test_doc", "txt"
        )
        
        assert len(chunks) == 0
        assert stats["total_chunks"] == 0


class TestChunkingConfig:
    """Test chunking configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ChunkingConfig()
        assert config.base_tokens == 512
        assert config.min_tokens == 128
        assert config.max_tokens == 1024
        assert config.overlap_ratio == 0.15
        assert BlockType.PARAGRAPH in config.type_weights

    def test_custom_config(self):
        """Test custom configuration."""
        config = ChunkingConfig(
            base_tokens=256,
            min_tokens=64,
            max_tokens=512,
            overlap_ratio=0.1,
        )
        assert config.base_tokens == 256
        assert config.min_tokens == 64
        assert config.max_tokens == 512
        assert config.overlap_ratio == 0.1