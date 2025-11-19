from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Protocol
from pathlib import Path
import numpy as np

@dataclass
class PageContext:
    """Represents the state of a page as it flows through the pipeline."""
    doc_id: str
    page_index: int
    # Visual data
    image: Optional[np.ndarray] = None  # BGR image (OpenCV format) or None if not loaded
    image_path: Optional[Path] = None
    width: int = 0
    height: int = 0
    
    # Layout data
    raw_layout: Optional[Dict] = None  # Output from LLM
    figures: List[Dict] = field(default_factory=list)
    text_blocks: List[Dict] = field(default_factory=list)
    page_text: str = ""
    
    # Metadata / Diagnostics
    violations: Dict[str, int] = field(default_factory=lambda: {'missing_caption_ref': 0, 'equation_overlap': 0})
    absorb_events: List[Dict] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_layout_record(self) -> Dict:
        """Convert to the legacy dictionary format for structured_layout.jsonl."""
        return {
            'doc_id': self.doc_id,
            'page_index': self.page_index,
            'page_size': [self.width, self.height],
            'engine': self.metadata.get('engine', 'llm'),
            'figures': self.figures,
            'text_blocks': self.text_blocks,
            'page_text': self.page_text,
            'violations': self.violations
        }

class PipelineStage(Protocol):
    def process(self, ctx: PageContext) -> PageContext:
        ...
