import pdfplumber
import pytesseract
from PIL import Image
import io
import re
import logging
from typing import Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFProcessor:
    """Enhanced PDF text extraction with OCR fallback and error handling."""
    
    def __init__(self):
        self.ocr_threshold = 50  # Minimum characters to consider text extraction successful
    
    def extract_text_from_pdf(self, pdf_file_obj, filename: str) -> Tuple[str, bool, Optional[str]]:
        """
        Extract text from PDF with OCR fallback.
        
        Args:
            pdf_file_obj: File object containing PDF data
            filename: Name of the PDF file for logging
            
        Returns:
            Tuple of (extracted_text, success_flag, error_message)
        """
        try:
            # First, try native text extraction with pdfplumber
            text = self._extract_native_text(pdf_file_obj, filename)
            
            if self._is_text_sufficient(text):
                logger.info(f"Successfully extracted native text from {filename}")
                cleaned_text = self._clean_text(text)
                return cleaned_text, True, None
            else:
                # Text layer missing or insufficient, try OCR
                logger.info(f"Native text insufficient for {filename}, attempting OCR")
                ocr_text = self._extract_ocr_text(pdf_file_obj, filename)
                
                if ocr_text:
                    cleaned_text = self._clean_text(ocr_text)
                    return cleaned_text, True, None
                else:
                    error_msg = f"Both native and OCR text extraction failed for {filename}"
                    logger.error(error_msg)
                    return "", False, error_msg
                    
        except Exception as e:
            error_msg = f"Error processing {filename}: {str(e)}"
            logger.error(error_msg)
            return "", False, error_msg
    
    def _extract_native_text(self, pdf_file_obj, filename: str) -> str:
        """Extract text using pdfplumber (native text layer)."""
        try:
            # Reset file pointer
            pdf_file_obj.seek(0)
            
            with pdfplumber.open(pdf_file_obj) as pdf:
                text_parts = []
                
                for page_num, page in enumerate(pdf.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text_parts.append(page_text)
                    except Exception as e:
                        logger.warning(f"Error extracting text from page {page_num + 1} of {filename}: {e}")
                        continue
                
                return "\n".join(text_parts)
                
        except Exception as e:
            logger.error(f"Native text extraction failed for {filename}: {e}")
            return ""
    
    def _extract_ocr_text(self, pdf_file_obj, filename: str) -> str:
        """Extract text using OCR (for scanned PDFs)."""
        try:
            # Reset file pointer
            pdf_file_obj.seek(0)
            
            with pdfplumber.open(pdf_file_obj) as pdf:
                ocr_text_parts = []
                
                for page_num, page in enumerate(pdf.pages):
                    try:
                        # Convert page to image
                        page_image = page.to_image(resolution=300)
                        
                        # Convert to PIL Image for pytesseract
                        pil_image = page_image.original
                        
                        # Perform OCR
                        page_text = pytesseract.image_to_string(pil_image, lang='eng')
                        
                        if page_text.strip():
                            ocr_text_parts.append(page_text)
                            
                    except Exception as e:
                        logger.warning(f"OCR failed for page {page_num + 1} of {filename}: {e}")
                        continue
                
                return "\n".join(ocr_text_parts)
                
        except Exception as e:
            logger.error(f"OCR text extraction failed for {filename}: {e}")
            return ""
    
    def _is_text_sufficient(self, text: str) -> bool:
        """Check if extracted text is sufficient (not just whitespace/minimal content)."""
        if not text:
            return False
        
        # Remove whitespace and count meaningful characters
        cleaned = re.sub(r'\s+', '', text)
        return len(cleaned) >= self.ocr_threshold
    
    def _clean_text(self, text: str) -> str:
        """Clean and sanitize extracted text."""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove excessive line breaks
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Remove common PDF artifacts
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)  # Remove control characters
        text = re.sub(r'[^\x20-\x7E\n\r\t]', '', text)    # Keep only printable ASCII + newlines/tabs
        
        return text
    
    def is_pdf_readable(self, pdf_file_obj, filename: str) -> Tuple[bool, Optional[str]]:
        """
        Check if PDF is readable (not corrupted, encrypted, etc.).
        
        Returns:
            Tuple of (is_readable, error_message)
        """
        try:
            pdf_file_obj.seek(0)
            
            with pdfplumber.open(pdf_file_obj) as pdf:
                # Try to access basic PDF properties
                num_pages = len(pdf.pages)
                
                if num_pages == 0:
                    return False, f"PDF {filename} has no pages"
                
                # Try to access first page to check for encryption/corruption
                first_page = pdf.pages[0]
                _ = first_page.bbox  # This will fail if PDF is corrupted
                
                return True, None
                
        except Exception as e:
            error_msg = f"PDF {filename} is not readable: {str(e)}"
            return False, error_msg 