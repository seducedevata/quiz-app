"""
Multimodal Document Processing Pipeline

This module implements state-of-the-art document processing for complex, colorful documents
like textbooks with text, images, diagrams, and varied layouts.

Features:
- Document Layout Analysis (DLA)
- OCR with layout awareness
- Image extraction and captioning
- Table detection and extraction
- Multimodal AI integration
- Structured output (XML/CSV/JSON)
"""

import os
import logging
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import pytesseract
import pdf2image
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration
import torch

logger = logging.getLogger(__name__)


@dataclass
class DocumentRegion:
    """Represents a detected region in a document"""

    region_type: str  # 'text', 'image', 'table', 'heading', 'caption'
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    content: str
    confidence: float
    page_number: int
    image_path: Optional[str] = None


@dataclass
class ProcessedDocument:
    """Represents a fully processed document"""

    title: str
    total_pages: int
    regions: List[DocumentRegion]
    images: List[str]
    metadata: Dict[str, Any]
    processing_time: float


class MultimodalDocumentProcessor:
    """Advanced multimodal document processing pipeline"""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the multimodal processor

        Args:
            config: Configuration dictionary with processing settings
        """
        self.config = config
        self.output_dir = Path(config.get("output_dir", "data/processed_documents"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Image processing settings
        self.dpi = config.get("dpi", 300)
        self.image_format = config.get("image_format", "PNG")

        # OCR settings
        self.ocr_config = config.get("ocr_config", "--oem 3 --psm 6")

        # Initialize AI models
        self._init_ai_models()

        logger.info("🔥 Multimodal Document Processor initialized!")

    def _init_ai_models(self):
        """Initialize AI models for multimodal processing"""
        try:
            # Image captioning model
            self.caption_processor = BlipProcessor.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            )
            self.caption_model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            )

            # Document layout analysis (using a pre-trained model)
            # Note: In production, you'd use a specialized DLA model
            self.layout_analyzer = pipeline(
                "object-detection", model="microsoft/table-transformer-detection"
            )

            logger.info("✅ AI models loaded successfully")

        except Exception as e:
            logger.error(f"❌ Error loading AI models: {e}")
            # Fallback to basic processing
            self.caption_processor = None
            self.caption_model = None
            self.layout_analyzer = None

    def process_document(self, document_path: str) -> ProcessedDocument:
        """
        Process a complete document through the multimodal pipeline

        Args:
            document_path: Path to the document (PDF, image, etc.)

        Returns:
            ProcessedDocument: Fully processed document with structured data
        """
        start_time = datetime.now()
        document_path = Path(document_path)

        logger.info(f"🚀 Starting multimodal processing of: {document_path.name}")

        try:
            # Step 1: Convert document to images
            page_images = self._convert_to_images(document_path)

            # Step 2: Process each page
            all_regions = []
            all_images = []

            for page_num, page_image in enumerate(page_images, 1):
                logger.info(f"📄 Processing page {page_num}/{len(page_images)}")

                # Preprocess image
                processed_image = self._preprocess_image(page_image)

                # Document Layout Analysis
                regions = self._analyze_layout(processed_image, page_num)

                # Extract content from each region
                for region in regions:
                    if region.region_type == "text":
                        region.content = self._extract_text(processed_image, region.bbox)
                    elif region.region_type == "image":
                        image_path = self._extract_and_save_image(
                            processed_image, region.bbox, page_num, len(all_images)
                        )
                        region.image_path = image_path
                        all_images.append(image_path)

                        # Generate AI caption
                        if self.caption_model:
                            region.content = self._generate_image_caption(image_path)

                all_regions.extend(regions)

            # Step 3: Create structured document
            processing_time = (datetime.now() - start_time).total_seconds()

            processed_doc = ProcessedDocument(
                title=document_path.stem,
                total_pages=len(page_images),
                regions=all_regions,
                images=all_images,
                metadata={
                    "source_file": str(document_path),
                    "processed_at": datetime.now().isoformat(),
                    "processor_version": "1.0.0",
                    "ai_models_used": bool(self.caption_model),
                },
                processing_time=processing_time,
            )

            logger.info(f"✅ Document processed successfully in {processing_time:.2f}s")
            logger.info(f"📊 Extracted {len(all_regions)} regions and {len(all_images)} images")

            return processed_doc

        except Exception as e:
            logger.error(f"❌ Error processing document: {e}")
            raise

    def process_document_for_training(self, document_path: str) -> str:
        """
        Process a document and return training-ready text

        Args:
            document_path: Path to the document

        Returns:
            Training-ready text combining OCR text and image captions
        """
        try:
            processed_doc = self.process_document(document_path)
            return self._convert_to_training_text(processed_doc)
        except Exception as e:
            logger.error(f"❌ Error processing document for training: {e}")
            return ""

    def _convert_to_training_text(self, processed_doc: ProcessedDocument) -> str:
        """Convert ProcessedDocument to training text"""
        try:
            text_parts = []

            # Add document title
            if processed_doc.title:
                text_parts.append(f"Document: {processed_doc.title}")

            # Process regions by page
            current_page = 0
            for region in processed_doc.regions:
                if region.page_number != current_page:
                    current_page = region.page_number
                    text_parts.append(f"\n--- Page {current_page} ---")

                if region.content:
                    if region.region_type == "text":
                        # Clean and add text content
                        clean_text = region.content.strip()
                        if clean_text:
                            text_parts.append(clean_text)
                    elif region.region_type == "image" and region.content:
                        # Add image caption as descriptive text for training
                        text_parts.append(f"[Image Description: {region.content}]")

            result = "\n\n".join(text_parts)
            logger.debug(f"Converted to {len(result)} characters of training text")
            return result

        except Exception as e:
            logger.error(f"Error converting to training text: {e}")
            return ""

    def _convert_to_images(self, document_path: Path) -> List[Image.Image]:
        """Convert document to list of page images"""
        if document_path.suffix.lower() == ".pdf":
            # Convert PDF to images
            pages = pdf2image.convert_from_path(document_path, dpi=self.dpi, fmt=self.image_format)
            logger.info(f"📄 Converted PDF to {len(pages)} page images")
            return pages
        else:
            # Single image file
            image = Image.open(document_path)
            return [image]

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image for better OCR and analysis"""
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Enhance contrast and sharpness for better OCR
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.2)

        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.1)

        return image

    def _analyze_layout(self, image: Image.Image, page_num: int) -> List[DocumentRegion]:
        """
        Analyze document layout to identify different regions

        This is a simplified version. In production, you'd use specialized DLA models
        like LayoutLM or cloud services like Google Document AI
        """
        regions = []

        # Convert PIL image to numpy array for OpenCV
        img_array = np.array(image)

        # Simple region detection using contours (basic approach)
        # In production, replace with advanced DLA models
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        # Find text regions using simple thresholding
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)

            # Filter out very small regions
            if w > 50 and h > 20:
                # Simple heuristic to classify regions
                aspect_ratio = w / h
                area = w * h

                if aspect_ratio > 3 and area > 5000:
                    region_type = "text"
                elif aspect_ratio < 2 and area > 10000:
                    region_type = "image"
                else:
                    region_type = "text"

                region = DocumentRegion(
                    region_type=region_type,
                    bbox=(x, y, x + w, y + h),
                    content="",
                    confidence=0.8,  # Placeholder
                    page_number=page_num,
                )
                regions.append(region)

        logger.info(f"🔍 Detected {len(regions)} regions on page {page_num}")
        return regions

    def _extract_text(self, image: Image.Image, bbox: Tuple[int, int, int, int]) -> str:
        """Extract text from a specific region using OCR"""
        x1, y1, x2, y2 = bbox

        # Crop the region
        cropped = image.crop((x1, y1, x2, y2))

        # Run OCR
        text = pytesseract.image_to_string(cropped, config=self.ocr_config)

        return text.strip()

    def _extract_and_save_image(
        self, image: Image.Image, bbox: Tuple[int, int, int, int], page_num: int, image_index: int
    ) -> str:
        """Extract and save an image region"""
        x1, y1, x2, y2 = bbox

        # Crop the image
        cropped = image.crop((x1, y1, x2, y2))

        # Save the image
        image_filename = f"page_{page_num}_image_{image_index}.png"
        image_path = self.output_dir / "images" / image_filename
        image_path.parent.mkdir(exist_ok=True)

        cropped.save(image_path)

        return str(image_path)

    def _generate_image_caption(self, image_path: str) -> str:
        """Generate AI caption for an extracted image"""
        if not self.caption_model:
            return "Image extracted (AI captioning not available)"

        try:
            # Load and process image
            image = Image.open(image_path)
            inputs = self.caption_processor(image, return_tensors="pt")

            # Generate caption
            out = self.caption_model.generate(**inputs, max_length=50)
            caption = self.caption_processor.decode(out[0], skip_special_tokens=True)

            return f"AI Caption: {caption}"

        except Exception as e:
            logger.error(f"Error generating caption: {e}")
            return "Image extracted (caption generation failed)"


# Alias for backward compatibility
MultimodalProcessor = MultimodalDocumentProcessor