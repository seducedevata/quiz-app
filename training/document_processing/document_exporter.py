"""
Document Export Module

Exports processed multimodal documents to various formats:
- XML (structured document format)
- CSV (tabular data for AI training)
- JSON (machine-readable format)
- Training datasets for AI models
"""

import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
from datetime import datetime
import logging

from .multimodal_processor import ProcessedDocument, DocumentRegion

logger = logging.getLogger(__name__)


class DocumentExporter:
    """Export processed documents to various formats"""

    def __init__(self, output_dir: str = None):
        # 🚀 CRITICAL FIX: Use central configuration manager for paths
        if output_dir is None:
            try:
                from .proper_config_manager import get_config_manager
                config_manager = get_config_manager()
                config = config_manager.get_config()
                # Use the configured data directory for exports
                output_dir = str(Path(config.data_directory) / "exports")
                logger.info(f"✅ Using configured data directory for exports: {output_dir}")
            except Exception as e:
                logger.warning(f"⚠️ Could not get configured data directory: {e}")
                output_dir = "data/exports"  # fallback
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_to_xml(self, document: ProcessedDocument, filename: str = None) -> str:
        """
        Export document to structured XML format

        Args:
            document: Processed document
            filename: Output filename (optional)

        Returns:
            str: Path to exported XML file
        """
        if not filename:
            filename = f"{document.title}_structured.xml"

        output_path = self.output_dir / filename

        # Create root element
        root = ET.Element("document")
        root.set("title", document.title)
        root.set("total_pages", str(document.total_pages))
        root.set("processing_time", str(document.processing_time))
        root.set("processed_at", document.metadata.get("processed_at", ""))

        # Add metadata
        metadata_elem = ET.SubElement(root, "metadata")
        for key, value in document.metadata.items():
            meta_item = ET.SubElement(metadata_elem, "item")
            meta_item.set("key", key)
            meta_item.text = str(value)

        # Group regions by page
        pages_dict = {}
        for region in document.regions:
            page_num = region.page_number
            if page_num not in pages_dict:
                pages_dict[page_num] = []
            pages_dict[page_num].append(region)

        # Add pages and regions
        for page_num in sorted(pages_dict.keys()):
            page_elem = ET.SubElement(root, "page")
            page_elem.set("number", str(page_num))

            for region in pages_dict[page_num]:
                region_elem = ET.SubElement(page_elem, region.region_type)
                region_elem.set("confidence", str(region.confidence))
                region_elem.set(
                    "bbox", f"{region.bbox[0]},{region.bbox[1]},{region.bbox[2]},{region.bbox[3]}"
                )

                if region.region_type == "image" and region.image_path:
                    region_elem.set("image_path", region.image_path)

                region_elem.text = region.content

        # Write XML file
        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ", level=0)
        tree.write(output_path, encoding="utf-8", xml_declaration=True)

        logger.info(f"✅ Exported XML to: {output_path}")
        return str(output_path)

    def export_to_csv(self, document: ProcessedDocument, filename: str = None) -> str:
        """
        Export document to CSV format for AI training

        Args:
            document: Processed document
            filename: Output filename (optional)

        Returns:
            str: Path to exported CSV file
        """
        if not filename:
            filename = f"{document.title}_training_data.csv"

        output_path = self.output_dir / filename

        # Prepare data for CSV
        csv_data = []

        for region in document.regions:
            row = {
                "page_number": region.page_number,
                "region_type": region.region_type,
                "content": region.content,
                "confidence": region.confidence,
                "bbox_x1": region.bbox[0],
                "bbox_y1": region.bbox[1],
                "bbox_x2": region.bbox[2],
                "bbox_y2": region.bbox[3],
                "image_path": region.image_path or "",
                "document_title": document.title,
                "total_pages": document.total_pages,
            }
            csv_data.append(row)

        # Write CSV file
        df = pd.DataFrame(csv_data)
        df.to_csv(output_path, index=False, encoding="utf-8")

        logger.info(f"✅ Exported CSV to: {output_path}")
        return str(output_path)

    def export_to_json(self, document: ProcessedDocument, filename: str = None) -> str:
        """
        Export document to JSON format

        Args:
            document: Processed document
            filename: Output filename (optional)

        Returns:
            str: Path to exported JSON file
        """
        if not filename:
            filename = f"{document.title}_data.json"

        output_path = self.output_dir / filename

        # Convert to JSON-serializable format
        json_data = {
            "title": document.title,
            "total_pages": document.total_pages,
            "processing_time": document.processing_time,
            "metadata": document.metadata,
            "images": document.images,
            "regions": [],
        }

        for region in document.regions:
            region_data = {
                "region_type": region.region_type,
                "bbox": region.bbox,
                "content": region.content,
                "confidence": region.confidence,
                "page_number": region.page_number,
                "image_path": region.image_path,
            }
            json_data["regions"].append(region_data)

        # Write JSON file
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        logger.info(f"✅ Exported JSON to: {output_path}")
        return str(output_path)

    def export_training_dataset(
        self, document: ProcessedDocument, dataset_type: str = "vqa"
    ) -> Dict[str, str]:
        """
        Export document as AI training dataset

        Args:
            document: Processed document
            dataset_type: Type of dataset ('vqa', 'captioning', 'classification')

        Returns:
            Dict[str, str]: Paths to exported dataset files
        """
        output_paths = {}

        if dataset_type == "vqa":
            # Visual Question Answering dataset
            output_paths.update(self._export_vqa_dataset(document))
        elif dataset_type == "captioning":
            # Image captioning dataset
            output_paths.update(self._export_captioning_dataset(document))
        elif dataset_type == "classification":
            # Document classification dataset
            output_paths.update(self._export_classification_dataset(document))

        return output_paths

    def _export_vqa_dataset(self, document: ProcessedDocument) -> Dict[str, str]:
        """Export as Visual Question Answering dataset"""
        vqa_data = []

        # Group text and images by page
        pages_dict = {}
        for region in document.regions:
            page_num = region.page_number
            if page_num not in pages_dict:
                pages_dict[page_num] = {"text": [], "images": []}

            if region.region_type == "text":
                pages_dict[page_num]["text"].append(region.content)
            elif region.region_type == "image" and region.image_path:
                pages_dict[page_num]["images"].append(
                    {"path": region.image_path, "caption": region.content}
                )

        # Create VQA pairs
        for page_num, content in pages_dict.items():
            page_text = " ".join(content["text"])

            for img_data in content["images"]:
                # Generate question-answer pairs
                vqa_item = {
                    "image_path": img_data["path"],
                    "question": f"What is shown in this image from page {page_num}?",
                    "answer": img_data["caption"],
                    "context": page_text[:500],  # Limit context length
                    "page_number": page_num,
                    "document_title": document.title,
                }
                vqa_data.append(vqa_item)

        # Save VQA dataset
        vqa_path = self.output_dir / f"{document.title}_vqa_dataset.json"
        with open(vqa_path, "w", encoding="utf-8") as f:
            json.dump(vqa_data, f, indent=2, ensure_ascii=False)

        logger.info(f"✅ Exported VQA dataset: {vqa_path}")
        return {"vqa": str(vqa_path)}

    def _export_captioning_dataset(self, document: ProcessedDocument) -> Dict[str, str]:
        """Export as image captioning dataset"""
        captioning_data = []

        for region in document.regions:
            if region.region_type == "image" and region.image_path:
                caption_item = {
                    "image_path": region.image_path,
                    "caption": region.content,
                    "page_number": region.page_number,
                    "document_title": document.title,
                    "bbox": region.bbox,
                }
                captioning_data.append(caption_item)

        # Save captioning dataset
        caption_path = self.output_dir / f"{document.title}_captioning_dataset.json"
        with open(caption_path, "w", encoding="utf-8") as f:
            json.dump(captioning_data, f, indent=2, ensure_ascii=False)

        logger.info(f"✅ Exported captioning dataset: {caption_path}")
        return {"captioning": str(caption_path)}

    def _export_classification_dataset(self, document: ProcessedDocument) -> Dict[str, str]:
        """Export as document classification dataset"""
        classification_data = []

        for region in document.regions:
            class_item = {
                "content": region.content,
                "label": region.region_type,
                "confidence": region.confidence,
                "page_number": region.page_number,
                "document_title": document.title,
                "bbox": region.bbox,
            }
            classification_data.append(class_item)

        # Save classification dataset
        class_path = self.output_dir / f"{document.title}_classification_dataset.json"
        with open(class_path, "w", encoding="utf-8") as f:
            json.dump(classification_data, f, indent=2, ensure_ascii=False)

        logger.info(f"✅ Exported classification dataset: {class_path}")
        return {"classification": str(class_path)}

    def export_all_formats(self, document: ProcessedDocument) -> Dict[str, str]:
        """
        Export document to all available formats

        Args:
            document: Processed document

        Returns:
            Dict[str, str]: Paths to all exported files
        """
        export_paths = {}

        # Export to standard formats
        export_paths["xml"] = self.export_to_xml(document)
        export_paths["csv"] = self.export_to_csv(document)
        export_paths["json"] = self.export_to_json(document)

        # Export training datasets
        training_datasets = self.export_training_dataset(document, "vqa")
        export_paths.update(training_datasets)

        training_datasets = self.export_training_dataset(document, "captioning")
        export_paths.update(training_datasets)

        training_datasets = self.export_training_dataset(document, "classification")
        export_paths.update(training_datasets)

        logger.info(f"🎉 Exported document to {len(export_paths)} formats")
        return export_paths
