"""
Improved chunking logic for scientific papers.
Extracts from PDF sections and creates optimally-sized chunks with filtering and overlap.
"""

from typing import Dict, List, Tuple


def create_chunks_from_sections(
    filename: str,
    metadata: dict,
    sections: dict,
    max_chunk_size: int = 2500,
    overlap_size: int = 200
) -> Tuple[List[str], List[Dict], List[str]]:
    """
    Improved chunking logic with filtering, size control, and overlap.
    Compatible with backend/main.py ingest endpoint format.
    
    Args:
        filename: Name of the PDF file
        metadata: Metadata dict from processor (title, authors, arxiv_id)
        sections: Sections dict from processor (header -> content)
        max_chunk_size: Maximum characters per chunk (default: 2500)
        overlap_size: Characters to overlap between adjacent chunks (default: 200)
    
    Returns:
        Tuple of (documents, metadatas, ids) ready for ChromaDB insertion
    """
    # Low-value sections to skip (based on corpus analysis)
    SKIP_SECTIONS = {
        "preamble", "references", "bibliography",
        "acknowledgements", "acknowledgments", "acknowledgement",
        "author contributions", "author contribution",
        "additional information", "data availability",
        "code availability", "code and data availability",
        "competing interests", "competing financial interests",
        "conflict of interest", "funding information", "funding",
        "extended author information", "online content",
        "reporting summary", "article",
        "reprints and permissions information is available at http://www.nature.com/reprints",
        "amanda a. volk 1 & milad abolhasani 1",
        "supplementary information", "supporting information"
    }
    
    def _split_text_hard(text: str, max_size: int) -> List[str]:
        """Hard split text when no natural boundaries exist."""
        if len(text) <= max_size:
            return [text]
        
        chunks = []
        remaining = text
        
        while len(remaining) > max_size:
            split_point = max_size
            sentence_end = max(
                remaining.rfind('. ', 0, max_size),
                remaining.rfind('! ', 0, max_size),
                remaining.rfind('? ', 0, max_size)
            )
            
            if sentence_end > max_size * 0.5:
                split_point = sentence_end + 1
            else:
                space_pos = remaining.rfind(' ', 0, max_size)
                if space_pos > max_size * 0.5:
                    split_point = space_pos
            
            chunks.append(remaining[:split_point].strip())
            remaining = remaining[split_point:].strip()
        
        if remaining:
            chunks.append(remaining)
        
        return chunks
    
    # Prepare base metadata
    parent_id = metadata.get("arxiv_id") or filename or "unknown_doc"
    parent_id = parent_id.replace(" ", "_").replace(":", "_").replace("-", "_")
    
    base_meta = {
        "parent_id": parent_id,
        "filename": filename,
        "title": metadata.get("title", "Unknown"),
        "authors": ", ".join(metadata.get("authors", []))
    }
    
    docs, metas, ids = [], [], []
    
    for header, content in sections.items():
        if not content.strip():
            continue
        
        # Filter: Skip low-value sections
        if header.lower().strip() in SKIP_SECTIONS:
            continue
        
        # Filter: Skip very short sections
        if len(content) < 100:
            continue
        
        # Size control: Split large sections
        if len(content) <= max_chunk_size:
            # Section fits in one chunk
            chunk_id = f"{parent_id}#{header.replace(' ', '_')[:50]}"
            chunk_meta = {**base_meta, "section": header}
            chunk_content = content.replace("\n", " ")
            
            docs.append(chunk_content)
            metas.append(chunk_meta)
            ids.append(chunk_id)
        else:
            # Split large section into multiple chunks
            paragraphs = content.split("\n\n")
            current_chunk = []
            current_size = 0
            sub_chunk_idx = 0
            
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                
                para_size = len(para)
                
                # If single paragraph exceeds max size, split it further
                if para_size > max_chunk_size:
                    # Save any accumulated content first
                    if current_chunk:
                        chunk_text = "\n\n".join(current_chunk)
                        chunk_id = f"{parent_id}#{header.replace(' ', '_')[:30]}_part{sub_chunk_idx}"
                        chunk_meta = {**base_meta, "section": header}
                        
                        docs.append(chunk_text.replace("\n", " "))
                        metas.append(chunk_meta)
                        ids.append(chunk_id)
                        
                        current_chunk = []
                        current_size = 0
                        sub_chunk_idx += 1
                    
                    # Hard split the oversized paragraph
                    para_chunks = _split_text_hard(para, max_chunk_size - overlap_size)
                    
                    for i, para_chunk in enumerate(para_chunks):
                        chunk_id = f"{parent_id}#{header.replace(' ', '_')[:30]}_part{sub_chunk_idx}"
                        chunk_meta = {**base_meta, "section": header}
                        
                        docs.append(para_chunk.replace("\n", " "))
                        metas.append(chunk_meta)
                        ids.append(chunk_id)
                        
                        sub_chunk_idx += 1
                        
                        # Add overlap for next iteration
                        if i < len(para_chunks) - 1 and overlap_size > 0:
                            current_chunk = [para_chunk[-overlap_size:]]
                            current_size = overlap_size
                        else:
                            current_chunk = []
                            current_size = 0
                    
                    continue
                
                # Check if adding this paragraph would exceed max size
                if current_size + para_size > max_chunk_size and current_chunk:
                    # Save current chunk
                    chunk_text = "\n\n".join(current_chunk)
                    chunk_id = f"{parent_id}#{header.replace(' ', '_')[:30]}_part{sub_chunk_idx}"
                    chunk_meta = {**base_meta, "section": header}
                    
                    docs.append(chunk_text.replace("\n", " "))
                    metas.append(chunk_meta)
                    ids.append(chunk_id)
                    
                    # Overlap: Keep last paragraph for context
                    if overlap_size > 0 and len(current_chunk) > 0:
                        overlap_text = current_chunk[-1]
                        if len(overlap_text) <= overlap_size:
                            current_chunk = [overlap_text, para]
                            current_size = len(overlap_text) + para_size
                        else:
                            current_chunk = [overlap_text[-overlap_size:], para]
                            current_size = overlap_size + para_size
                    else:
                        current_chunk = [para]
                        current_size = para_size
                    
                    sub_chunk_idx += 1
                else:
                    # Add paragraph to current chunk
                    current_chunk.append(para)
                    current_size += para_size
            
            # Don't forget the last chunk
            if current_chunk:
                chunk_text = "\n\n".join(current_chunk)
                chunk_id = f"{parent_id}#{header.replace(' ', '_')[:30]}_part{sub_chunk_idx}"
                chunk_meta = {**base_meta, "section": header}
                
                docs.append(chunk_text.replace("\n", " "))
                metas.append(chunk_meta)
                ids.append(chunk_id)
    
    return docs, metas, ids
