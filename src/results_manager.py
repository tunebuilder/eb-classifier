import csv
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ResultsManager:
    """Manages storage and export of LLM analysis results."""
    
    def __init__(self, model_name: str = "unknown"):
        self.results = []
        self.errors = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_name = self._sanitize_model_name(model_name)
        
        # CSV headers as specified in the requirements
        self.csv_headers = [
            "article_title",
            "inclusion_exclusion_decision", 
            "category",
            "detailed_reasoning_for_decision",
            "citation",
            "open_access_url",
            "confidence",
            "source_file",
            "timestamp"
        ]
    
    def _sanitize_model_name(self, model_name: str) -> str:
        """Sanitize model name for use in filenames."""
        import re
        # Replace common model name patterns with cleaner versions
        sanitized = model_name.lower()
        sanitized = re.sub(r'claude-opus-4.*', 'claude-opus-4', sanitized)
        sanitized = re.sub(r'o3.*', 'o3', sanitized)
        # Remove any characters that aren't alphanumeric, dash, or underscore
        sanitized = re.sub(r'[^a-z0-9\-_]', '-', sanitized)
        # Remove multiple consecutive dashes
        sanitized = re.sub(r'-+', '-', sanitized)
        # Remove leading/trailing dashes
        sanitized = sanitized.strip('-')
        return sanitized if sanitized else "unknown"
    
    def add_result(self, llm_result: Dict[str, Any], source_file: str):
        """Add a successful LLM analysis result."""
        try:
            # Map LLM result to CSV format
            csv_row = {
                "article_title": llm_result.get("article_title", ""),
                "inclusion_exclusion_decision": llm_result.get("inclusion_decision", ""),
                "category": llm_result.get("category", ""),
                "detailed_reasoning_for_decision": llm_result.get("detailed_reasoning", ""),
                "citation": "",  # Not extracted by LLM - would need separate processing
                "open_access_url": "",  # Not extracted by LLM - would need separate processing
                "confidence": "",  # Not provided by current LLM schema
                "source_file": source_file,
                "timestamp": datetime.now().isoformat()
            }
            
            self.results.append(csv_row)
            logger.info(f"Added result for {source_file}")
            
        except Exception as e:
            error_msg = f"Error adding result for {source_file}: {str(e)}"
            logger.error(error_msg)
            self.add_error(source_file, error_msg)
    
    def add_error(self, source_file: str, error_message: str):
        """Add an error record."""
        error_record = {
            "source_file": source_file,
            "error_message": error_message,
            "timestamp": datetime.now().isoformat()
        }
        self.errors.append(error_record)
        logger.error(f"Added error for {source_file}: {error_message}")
    
    def get_stats(self) -> Dict[str, int]:
        """Get processing statistics."""
        return {
            "total_processed": len(self.results) + len(self.errors),
            "successful": len(self.results),
            "failed": len(self.errors)
        }
    
    def export_to_csv(self, output_dir: str = "output") -> str:
        """Export results to CSV file."""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate filename with model name
            filename = f"evidence_base_classifier_results_{self.model_name}_{self.timestamp}.csv"
            filepath = os.path.join(output_dir, filename)
            
            # Write CSV
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.csv_headers)
                writer.writeheader()
                writer.writerows(self.results)
            
            logger.info(f"Exported {len(self.results)} results to {filepath}")
            return filepath
            
        except Exception as e:
            error_msg = f"Error exporting to CSV: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)
    
    def export_errors_to_csv(self, output_dir: str = "output") -> str:
        """Export errors to CSV file."""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate filename with model name
            filename = f"evidence_base_classifier_errors_{self.model_name}_{self.timestamp}.csv"
            filepath = os.path.join(output_dir, filename)
            
            # Write CSV
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ["source_file", "error_message", "timestamp"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.errors)
            
            logger.info(f"Exported {len(self.errors)} errors to {filepath}")
            return filepath
            
        except Exception as e:
            error_msg = f"Error exporting errors to CSV: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)
    
    def export_errors_to_text(self, output_dir: str = "logs") -> str:
        """Export errors to text file."""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate filename with model name
            filename = f"evidence_base_classifier_errors_{self.model_name}_{self.timestamp}.txt"
            filepath = os.path.join(output_dir, filename)
            
            # Write text file
            with open(filepath, 'w', encoding='utf-8') as txtfile:
                txtfile.write(f"Evidence Base Classifier - Error Log\n")
                txtfile.write(f"Generated: {datetime.now().isoformat()}\n")
                txtfile.write(f"Total Errors: {len(self.errors)}\n")
                txtfile.write("=" * 50 + "\n\n")
                
                for i, error in enumerate(self.errors, 1):
                    txtfile.write(f"Error #{i}\n")
                    txtfile.write(f"File: {error['source_file']}\n")
                    txtfile.write(f"Time: {error['timestamp']}\n")
                    txtfile.write(f"Error: {error['error_message']}\n")
                    txtfile.write("-" * 30 + "\n\n")
            
            logger.info(f"Exported {len(self.errors)} errors to {filepath}")
            return filepath
            
        except Exception as e:
            error_msg = f"Error exporting errors to text: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)
    
    def get_results_summary(self) -> str:
        """Get a formatted summary of results."""
        stats = self.get_stats()
        
        summary = f"""
## Processing Summary
- **Total files processed:** {stats['total_processed']}
- **Successfully analyzed:** {stats['successful']}
- **Failed to analyze:** {stats['failed']}
        """
        
        if self.results:
            # Count by decision
            included = sum(1 for r in self.results if r['inclusion_exclusion_decision'] == 'Included')
            excluded = sum(1 for r in self.results if r['inclusion_exclusion_decision'] == 'Excluded')
            
            summary += f"\n### Analysis Results\n"
            summary += f"- **Included papers:** {included}\n"
            summary += f"- **Excluded papers:** {excluded}\n"
            
            # Count by category (for included papers)
            if included > 0:
                categories = {}
                for r in self.results:
                    if r['inclusion_exclusion_decision'] == 'Included':
                        cat = r['category']
                        categories[cat] = categories.get(cat, 0) + 1
                
                summary += f"\n### Categories (Included Papers)\n"
                for cat, count in categories.items():
                    summary += f"- **{cat}:** {count}\n"
        
        return summary
    
    def get_failed_files_for_display(self) -> List[Dict[str, str]]:
        """Get failed files formatted for UI display."""
        return [
            {
                "File": error["source_file"],
                "Error Reason": error["error_message"],
                "Timestamp": error["timestamp"]
            }
            for error in self.errors
        ] 