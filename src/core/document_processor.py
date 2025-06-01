"""Document processing and chunking module for SmartRAG."""

import hashlib
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import tiktoken
from tenacity import retry, stop_after_attempt, wait_exponential


class BlockType(Enum):
    """Document block types."""

    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST = "list"
    TABLE = "table"
    CODE = "code"
    IMAGE = "image"
    QUOTE = "quote"


@dataclass
class Block:
    """Represents a document block."""

    type: BlockType
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    start_offset: int = 0
    end_offset: int = 0


@dataclass
class Section:
    """Represents a document section."""

    id: str
    path: List[str]
    blocks: List[Block]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """Represents a document chunk."""

    id: str
    text: str
    doc_id: str
    section_path: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    token_count: int = 0
    start_page: Optional[int] = None
    end_page: Optional[int] = None
    prev_chunk_id: Optional[str] = None
    next_chunk_id: Optional[str] = None


@dataclass
class ChunkingConfig:
    """Configuration for chunking strategy."""

    base_tokens: int = 512
    min_tokens: int = 128
    max_tokens: int = 1024
    overlap_ratio: float = 0.15
    type_weights: Dict[BlockType, float] = field(
        default_factory=lambda: {
            BlockType.PARAGRAPH: 1.0,
            BlockType.CODE: 0.6,
            BlockType.TABLE: 0.7,
            BlockType.LIST: 0.9,
            BlockType.HEADING: 0.5,
            BlockType.QUOTE: 0.8,
            BlockType.IMAGE: 0.3,
        }
    )


class Tokenizer:
    """Token counting and text splitting utilities."""

    def __init__(self, model_name: str = "cl100k_base"):
        """Initialize tokenizer."""
        self.encoding = tiktoken.get_encoding(model_name)

    def count(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))

    def split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting - can be enhanced with spaCy or NLTK
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to maximum token count."""
        tokens = self.encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        truncated_tokens = tokens[:max_tokens]
        return self.encoding.decode(truncated_tokens)


class DocumentProcessor:
    """Process documents and extract structured blocks."""

    def __init__(self):
        """Initialize document processor."""
        self.tokenizer = Tokenizer()

    def process_text(self, text: str, doc_type: str = "txt") -> List[Block]:
        """Process plain text into blocks."""
        blocks = []
        lines = text.split("\n")
        current_block_lines = []
        current_type = BlockType.PARAGRAPH
        offset = 0

        for line in lines:
            stripped = line.strip()

            # Detect block type
            if not stripped:
                if current_block_lines:
                    # End current block
                    block_text = "\n".join(current_block_lines)
                    blocks.append(
                        Block(
                            type=current_type,
                            text=block_text,
                            start_offset=offset,
                            end_offset=offset + len(block_text),
                        )
                    )
                    offset += len(block_text) + 1
                    current_block_lines = []
                    current_type = BlockType.PARAGRAPH
            elif stripped.startswith("#"):
                # Markdown heading
                if current_block_lines:
                    # Save previous block
                    block_text = "\n".join(current_block_lines)
                    blocks.append(
                        Block(
                            type=current_type,
                            text=block_text,
                            start_offset=offset,
                            end_offset=offset + len(block_text),
                        )
                    )
                    offset += len(block_text) + 1
                    current_block_lines = []

                # Create heading block
                level = len(re.match(r"^#+", stripped).group())
                blocks.append(
                    Block(
                        type=BlockType.HEADING,
                        text=stripped.lstrip("#").strip(),
                        metadata={"level": level},
                        start_offset=offset,
                        end_offset=offset + len(line),
                    )
                )
                offset += len(line) + 1
                current_type = BlockType.PARAGRAPH
            elif stripped.startswith("```"):
                # Code block
                if current_block_lines:
                    block_text = "\n".join(current_block_lines)
                    blocks.append(
                        Block(
                            type=current_type,
                            text=block_text,
                            start_offset=offset,
                            end_offset=offset + len(block_text),
                        )
                    )
                    current_block_lines = []

                # Collect code block
                code_lines = [line]
                offset_start = offset
                offset += len(line) + 1

                # Find closing ```
                for next_line in lines[lines.index(line) + 1 :]:
                    code_lines.append(next_line)
                    offset += len(next_line) + 1
                    if next_line.strip().startswith("```"):
                        break

                lang = stripped[3:].strip() if len(stripped) > 3 else ""
                code_text = "\n".join(code_lines[1:-1]) if len(code_lines) > 2 else ""
                blocks.append(
                    Block(
                        type=BlockType.CODE,
                        text=code_text,
                        metadata={"language": lang},
                        start_offset=offset_start,
                        end_offset=offset,
                    )
                )
                current_type = BlockType.PARAGRAPH
            elif re.match(r"^[-*+]\s+", stripped) or re.match(r"^\d+\.\s+", stripped):
                # List item
                if current_type != BlockType.LIST:
                    if current_block_lines:
                        block_text = "\n".join(current_block_lines)
                        blocks.append(
                            Block(
                                type=current_type,
                                text=block_text,
                                start_offset=offset,
                                end_offset=offset + len(block_text),
                            )
                        )
                        current_block_lines = []
                    current_type = BlockType.LIST

                current_block_lines.append(line)
            elif stripped.startswith(">"):
                # Quote block
                if current_type != BlockType.QUOTE:
                    if current_block_lines:
                        block_text = "\n".join(current_block_lines)
                        blocks.append(
                            Block(
                                type=current_type,
                                text=block_text,
                                start_offset=offset,
                                end_offset=offset + len(block_text),
                            )
                        )
                        current_block_lines = []
                    current_type = BlockType.QUOTE

                current_block_lines.append(stripped[1:].strip())
            else:
                # Regular paragraph line
                if current_type not in [BlockType.PARAGRAPH, BlockType.LIST, BlockType.QUOTE]:
                    if current_block_lines:
                        block_text = "\n".join(current_block_lines)
                        blocks.append(
                            Block(
                                type=current_type,
                                text=block_text,
                                start_offset=offset,
                                end_offset=offset + len(block_text),
                            )
                        )
                        current_block_lines = []
                    current_type = BlockType.PARAGRAPH

                current_block_lines.append(line)

        # Add final block if exists
        if current_block_lines:
            block_text = "\n".join(current_block_lines)
            blocks.append(
                Block(
                    type=current_type,
                    text=block_text,
                    start_offset=offset,
                    end_offset=offset + len(block_text),
                )
            )

        return blocks

    def clean_blocks(self, blocks: List[Block]) -> List[Block]:
        """Clean and normalize blocks."""
        cleaned = []
        for block in blocks:
            # Remove excessive whitespace
            text = re.sub(r"\s+", " ", block.text).strip()
            if not text:
                continue

            # Update block with cleaned text
            block.text = text
            cleaned.append(block)

        return cleaned

    def blocks_to_sections(self, blocks: List[Block]) -> List[Section]:
        """Organize blocks into hierarchical sections."""
        sections = []
        current_section_path = []
        current_blocks = []
        section_id = 0

        for block in blocks:
            if block.type == BlockType.HEADING:
                # Save previous section if exists
                if current_blocks:
                    sections.append(
                        Section(
                            id=f"section_{section_id}",
                            path=current_section_path.copy(),
                            blocks=current_blocks,
                        )
                    )
                    section_id += 1
                    current_blocks = []

                # Update section path based on heading level
                level = block.metadata.get("level", 1)
                if level <= len(current_section_path):
                    current_section_path = current_section_path[: level - 1]
                current_section_path.append(block.text)
            else:
                current_blocks.append(block)

        # Add final section
        if current_blocks:
            sections.append(
                Section(
                    id=f"section_{section_id}",
                    path=current_section_path.copy() if current_section_path else ["Main"],
                    blocks=current_blocks,
                )
            )

        return sections


class Chunker:
    """Intelligent document chunking with dynamic sizing and overlap."""

    def __init__(self, tokenizer: Tokenizer, config: ChunkingConfig):
        """Initialize chunker."""
        self.tokenizer = tokenizer
        self.config = config

    def chunk_section(self, section: Section, doc_id: str) -> List[Chunk]:
        """Chunk a section into optimized chunks."""
        chunks = []
        current_chunk_text = []
        current_tokens = 0
        chunk_id = 0

        # Calculate dynamic token budget for this section
        target_tokens = self._calculate_dynamic_budget(section.blocks)

        for block in section.blocks:
            block_text = block.text
            block_tokens = self.tokenizer.count(block_text)

            # Check if adding this block exceeds target
            if current_tokens + block_tokens > target_tokens and current_chunk_text:
                # Create chunk
                chunk_text = "\n\n".join(current_chunk_text)
                chunk = self._create_chunk(
                    chunk_id=f"{doc_id}_chunk_{chunk_id}",
                    text=chunk_text,
                    doc_id=doc_id,
                    section_path=section.path,
                    token_count=current_tokens,
                )
                chunks.append(chunk)
                chunk_id += 1

                # Calculate overlap
                overlap_tokens = int(target_tokens * self.config.overlap_ratio)
                overlap_text = self._get_overlap_text(current_chunk_text, overlap_tokens)

                # Start new chunk with overlap
                current_chunk_text = [overlap_text] if overlap_text else []
                current_tokens = self.tokenizer.count(overlap_text) if overlap_text else 0

            # Add block to current chunk
            current_chunk_text.append(block_text)
            current_tokens += block_tokens

        # Add final chunk
        if current_chunk_text:
            chunk_text = "\n\n".join(current_chunk_text)
            chunk = self._create_chunk(
                chunk_id=f"{doc_id}_chunk_{chunk_id}",
                text=chunk_text,
                doc_id=doc_id,
                section_path=section.path,
                token_count=current_tokens,
            )
            chunks.append(chunk)

        # Link chunks
        for i in range(len(chunks)):
            if i > 0:
                chunks[i].prev_chunk_id = chunks[i - 1].id
            if i < len(chunks) - 1:
                chunks[i].next_chunk_id = chunks[i + 1].id

        return chunks

    def _calculate_dynamic_budget(self, blocks: List[Block]) -> int:
        """Calculate dynamic token budget based on block types."""
        if not blocks:
            return self.config.base_tokens

        # Calculate weighted average based on block types
        total_weight = 0
        total_tokens = 0

        for block in blocks:
            weight = self.config.type_weights.get(block.type, 1.0)
            tokens = self.tokenizer.count(block.text)
            total_weight += weight
            total_tokens += tokens * weight

        if total_weight == 0:
            return self.config.base_tokens

        # Calculate density factor
        avg_tokens_per_block = total_tokens / total_weight / len(blocks)
        density_factor = min(1.5, max(0.5, 100 / avg_tokens_per_block))

        # Calculate target tokens
        target = int(self.config.base_tokens * density_factor)
        return max(self.config.min_tokens, min(self.config.max_tokens, target))

    def _get_overlap_text(self, chunk_text: List[str], overlap_tokens: int) -> str:
        """Get overlap text from the end of current chunk."""
        if not chunk_text:
            return ""

        # Take last block and split into sentences
        last_block = chunk_text[-1]
        sentences = self.tokenizer.split_sentences(last_block)

        overlap_text = []
        current_tokens = 0

        # Add sentences from the end until we reach overlap tokens
        for sentence in reversed(sentences):
            sentence_tokens = self.tokenizer.count(sentence)
            if current_tokens + sentence_tokens > overlap_tokens:
                break
            overlap_text.insert(0, sentence)
            current_tokens += sentence_tokens

        return " ".join(overlap_text)

    def _create_chunk(
        self,
        chunk_id: str,
        text: str,
        doc_id: str,
        section_path: List[str],
        token_count: int,
    ) -> Chunk:
        """Create a chunk with metadata."""
        # Generate checksum
        checksum = hashlib.sha256(text.encode()).hexdigest()[:16]

        return Chunk(
            id=chunk_id,
            text=text,
            doc_id=doc_id,
            section_path=section_path,
            metadata={
                "checksum": checksum,
                "char_count": len(text),
            },
            token_count=token_count,
        )


class DocumentPipeline:
    """Complete document processing pipeline."""

    def __init__(self, chunking_config: Optional[ChunkingConfig] = None):
        """Initialize pipeline."""
        self.processor = DocumentProcessor()
        self.tokenizer = Tokenizer()
        self.chunking_config = chunking_config or ChunkingConfig()
        self.chunker = Chunker(self.tokenizer, self.chunking_config)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def process_document(
        self,
        content: str,
        doc_id: str,
        doc_type: str = "txt",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Chunk], Dict[str, Any]]:
        """Process document through complete pipeline."""
        # Process into blocks
        blocks = self.processor.process_text(content, doc_type)

        # Clean blocks
        blocks = self.processor.clean_blocks(blocks)

        # Organize into sections
        sections = self.processor.blocks_to_sections(blocks)

        # Chunk sections
        all_chunks = []
        for section in sections:
            chunks = self.chunker.chunk_section(section, doc_id)
            all_chunks.extend(chunks)

        # Calculate statistics
        stats = {
            "total_chunks": len(all_chunks),
            "total_sections": len(sections),
            "total_blocks": len(blocks),
            "total_tokens": sum(c.token_count for c in all_chunks),
            "avg_chunk_tokens": (
                sum(c.token_count for c in all_chunks) / len(all_chunks)
                if all_chunks
                else 0
            ),
        }

        # Add metadata to chunks
        if metadata:
            for chunk in all_chunks:
                chunk.metadata.update(metadata)

        return all_chunks, stats