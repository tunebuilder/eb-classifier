import json
import logging
import time
from typing import Dict, Any, Optional, Tuple
from openai import OpenAI
import anthropic

# Configure logging
logger = logging.getLogger(__name__)

class LLMClient:
    """LLM client supporting both OpenAI and Anthropic APIs with structured output."""
    
    def __init__(self):
        self.openai_client = None
        self.anthropic_client = None
        
        # JSON schema for structured output
        self.json_schema = {
            "type": "object",
            "properties": {
                "article_title": {
                    "type": "string",
                    "description": "Title of the academic article under review"
                },
                "inclusion_decision": {
                    "type": "string",
                    "description": "Inclusion or exclusion decision for the paper",
                    "enum": ["Included", "Excluded"]
                },
                "justification": {
                    "type": "string",
                    "description": "Brief justification for the inclusion/exclusion decision"
                },
                "category": {
                    "type": "string",
                    "description": "The category assigned to the included paper. Required for all outputs (N/A if excluded).",
                    "enum": ["Client", "FLW", "Feasibility", "Data", "Grey Literature", "N/A"]
                },
                "detailed_reasoning": {
                    "type": "string",
                    "description": "Detailed reasoning for the inclusion or exclusion decision, reflecting critical review and criteria."
                }
            },
            "required": ["article_title", "inclusion_decision", "justification", "category", "detailed_reasoning"],
            "additionalProperties": False
        }
        
        # System prompt for the research assistant
        self.system_prompt = """You are an expert research assistant tasked with analyzing academic papers to determine their inclusion in the CommCare Evidence Base. Your role is to thoroughly read each paper and make classification decisions based on specific criteria.

# Your Task
For each paper provided, you must:
1. Determine inclusion/exclusion with clear justification
2. Assign appropriate category if included
3. Provide detailed reasoning for your decision
4. Include the article's title in your output

# Classification Categories
## INCLUDE Categories
**Client**: Peer-reviewed papers investigating the effect of equipping frontline workers (FLWs) with CommCare on client outcomes
- Focus: How CommCare impacts the people receiving services
- Examples: Patient health outcomes, service uptake, treatment adherence

**FLW**: Peer-reviewed papers examining how CommCare impacts frontline worker service delivery
- Focus: How CommCare affects the service providers themselves
- Examples: Worker productivity, job satisfaction, capacity, service quality

**Feasibility**: Peer-reviewed papers that don't fit Client or FLW categories but demonstrate overall acceptability of CommCare or describe important conceptual frameworks
- Focus: Implementation feasibility, acceptability studies, theoretical frameworks
- Examples: Usability studies, adoption barriers, conceptual models

**Data**: Papers demonstrating value derived from data collected by CommCare
- Focus: How CommCare-collected data provides insights or drives decisions
- Note: This category is typically ignored due to vagueness—use sparingly and only for strong cases

**Grey Literature**: Non-peer-reviewed studies deemed important for understanding CommCare
- Must also fit into one of the above categories
- Examples: Important reports, white papers, theses with significant findings

## EXCLUDE Criteria
### Automatic Exclusions
- Pre-prints (not peer-reviewed)
- Protocol papers (study designs without results)
- Data collection tool only: Papers mentioning CommCare solely as a survey platform without discussing impact on outcomes or service delivery
- Digitization only: Papers using CommCare only to transcribe/digitize paper forms (unless comparing CommCare vs. paper or describing specific features that improved data collection)

### Additional Exclusion Reasons
- Systematic reviews where CommCare papers are already in the Evidence Base
- Unclear platform: Studies where it's uncertain whether CommCare or another platform (e.g., Community Health Toolkit) was evaluated
- Insufficient focus: Papers where CommCare is mentioned but not central to findings
- Workshop summaries: Brief mentions without depth about CommCare's role
- Weak evidence: Papers with minimal discussion of CommCare's contribution

# Analysis Framework
**Step 1: Initial Assessment**
- Is this peer-reviewed? (If no, consider Grey Literature)
- Is this a pre-print or protocol paper? (If yes, exclude)

**Step 2: CommCare Role Analysis**

- How prominent is CommCare in the study?
- Is CommCare just a data collection tool or does it contribute to outcomes?
- Are there specific CommCare features discussed?
- Is there comparison with other methods/tools?

**Step 3: Category Assignment**

- Does it measure client outcomes? → Client
- Does it measure FLW impact? → FLW
- Does it assess feasibility/acceptability? → Feasibility
- Does it demonstrate data value? → Data (use sparingly)
- Non-peer-reviewed but important? → Grey Literature + another category

**Step 4: Strength Assessment**

- Is the evidence strong enough for inclusion?
- Is CommCare's role clearly articulated?
- Are the findings meaningful for the Evidence Base?

# Output
You must use the paper_review tool to provide your structured analysis. The tool expects:
- article_title: Title of the academic article under review
- inclusion_decision: "Included" or "Excluded"
- justification: Brief justification for the inclusion/exclusion decision
- category: One of ["Client", "FLW", "Feasibility", "Data", "Grey Literature", "N/A"]
- detailed_reasoning: Detailed reasoning for the inclusion or exclusion decision

Always use the paper_review tool to submit your analysis."""

    def initialize_clients(self, openai_api_key: Optional[str] = None, anthropic_api_key: Optional[str] = None):
        """Initialize API clients with provided keys."""
        if openai_api_key:
            try:
                self.openai_client = OpenAI(api_key=openai_api_key)
                logger.info("OpenAI client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                
        if anthropic_api_key:
            try:
                self.anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
                logger.info("Anthropic client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Anthropic client: {e}")
    
    def analyze_paper_openai(self, paper_text: str, filename: str) -> Tuple[Dict[str, Any], bool, Optional[str]]:
        """Analyze paper using OpenAI's API with structured output."""
        if not self.openai_client:
            return {}, False, "OpenAI client not initialized"
        
        try:
            logger.info(f"Analyzing {filename} with OpenAI O3")
            
            response = self.openai_client.chat.completions.create(
                model="o3",
                messages=[
                    {
                        "role": "developer",
                        "content": [
                            {
                                "type": "text",
                                "text": self.system_prompt
                            }
                        ]
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": paper_text
                            }
                        ]
                    }
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "paper_review",
                        "strict": True,
                        "schema": self.json_schema
                    }
                },
                reasoning_effort="high"
            )
            
            # Parse the response
            content = response.choices[0].message.content
            result = json.loads(content)
            
            logger.info(f"Successfully analyzed {filename} with OpenAI")
            return result, True, None
            
        except Exception as e:
            error_msg = f"OpenAI API error for {filename}: {str(e)}"
            logger.error(error_msg)
            return {}, False, error_msg
    
    def analyze_paper_anthropic(self, paper_text: str, filename: str) -> Tuple[Dict[str, Any], bool, Optional[str]]:
        """Analyze paper using Anthropic's API with structured output and streaming."""
        if not self.anthropic_client:
            return {}, False, "Anthropic client not initialized"
        
        try:
            logger.info(f"Analyzing {filename} with Claude Opus 4 (streaming enabled)")
            
            # Use streaming for long-running operations
            with self.anthropic_client.messages.stream(
                model="claude-opus-4-20250514",
                max_tokens=20000,  # Reduced to leave room for thinking tokens budget
                temperature=1,
                system=self.system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": paper_text
                            }
                        ]
                    }
                ],
                tools=[
                    {
                        "name": "paper_review",
                        "description": "Review an academic article and make inclusion/exclusion decisions with detailed reasoning",
                        "input_schema": self.json_schema
                    }
                ]
            ) as stream:
                # Get the final message from the stream
                message = stream.get_final_message()
            
            # Parse the response - Anthropic returns tool use in the content
            content = message.content
            if content and len(content) > 0:
                # Look for tool use in the response
                for block in content:
                    if hasattr(block, 'type') and block.type == 'tool_use':
                        result = block.input
                        logger.info(f"Successfully analyzed {filename} with Anthropic (streaming)")
                        return result, True, None
                
                # If no tool use found, try to parse as JSON
                if hasattr(content[0], 'text'):
                    try:
                        result = json.loads(content[0].text)
                        logger.info(f"Successfully analyzed {filename} with Anthropic (streaming fallback)")
                        return result, True, None
                    except json.JSONDecodeError:
                        pass
            
            error_msg = f"Unexpected response format from Anthropic for {filename}"
            logger.error(error_msg)
            return {}, False, error_msg
            
        except Exception as e:
            # If streaming fails, try non-streaming as fallback
            logger.warning(f"Streaming failed for {filename}, trying non-streaming: {str(e)}")
            try:
                message = self.anthropic_client.messages.create(
                    model="claude-opus-4-20250514",
                    max_tokens=20000,  # Reduced to leave room for thinking tokens budget
                    temperature=1,
                    system=self.system_prompt,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": paper_text
                                }
                            ]
                        }
                    ],
                    tools=[
                        {
                            "name": "paper_review",
                            "description": "Review an academic article and make inclusion/exclusion decisions with detailed reasoning",
                            "input_schema": self.json_schema
                        }
                    ]
                )
                
                # Parse the response
                content = message.content
                if content and len(content) > 0:
                    # Look for tool use in the response
                    for block in content:
                        if hasattr(block, 'type') and block.type == 'tool_use':
                            result = block.input
                            logger.info(f"Successfully analyzed {filename} with Anthropic (non-streaming fallback)")
                            return result, True, None
                    
                    # If no tool use found, try to parse as JSON
                    if hasattr(content[0], 'text'):
                        try:
                            result = json.loads(content[0].text)
                            logger.info(f"Successfully analyzed {filename} with Anthropic (non-streaming JSON fallback)")
                            return result, True, None
                        except json.JSONDecodeError:
                            pass
                
                error_msg = f"Unexpected response format from Anthropic for {filename} (both streaming and non-streaming failed)"
                logger.error(error_msg)
                return {}, False, error_msg
                
            except Exception as fallback_e:
                error_msg = f"Anthropic API error for {filename} (streaming and non-streaming both failed): {str(e)} | Fallback: {str(fallback_e)}"
                logger.error(error_msg)
                return {}, False, error_msg
    
    def analyze_paper(self, paper_text: str, filename: str, model: str) -> Tuple[Dict[str, Any], bool, Optional[str]]:
        """Analyze paper using the specified model."""
        # Check if text is too long (rough token estimation: 1 token ≈ 4 characters)
        estimated_tokens = len(paper_text) // 4
        
        # Enhanced warnings for different document sizes
        if estimated_tokens > 180000:
            logger.warning(f"Text for {filename} is very long ({estimated_tokens} estimated tokens), approaching context limit (200k)")
            return {}, False, f"Document too large ({estimated_tokens} tokens) - exceeds safe processing limit"
        elif estimated_tokens > 150000:
            logger.warning(f"Large document ({estimated_tokens} estimated tokens) - processing may take 5-10 minutes")
        elif estimated_tokens > 100000:
            logger.info(f"Medium-large document ({estimated_tokens} estimated tokens) - processing may take 2-5 minutes")
        elif estimated_tokens > 50000:
            logger.info(f"Large document ({estimated_tokens} estimated tokens) - processing may take 1-2 minutes")
        
        # For very long documents with Claude, ensure streaming is used
        if "claude" in model.lower() and estimated_tokens > 100000:
            logger.info(f"Using streaming for large document processing: {filename}")
        
        if "o3" in model.lower():
            return self.analyze_paper_openai(paper_text, filename)
        elif "claude" in model.lower():
            return self.analyze_paper_anthropic(paper_text, filename)
        else:
            return {}, False, f"Unsupported model: {model}"
    
    def validate_result(self, result: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate the structured output against our schema."""
        try:
            # Check required fields
            required_fields = ["article_title", "inclusion_decision", "justification", "category", "detailed_reasoning"]
            for field in required_fields:
                if field not in result:
                    return False, f"Missing required field: {field}"
                if not result[field] or (isinstance(result[field], str) and not result[field].strip()):
                    return False, f"Empty required field: {field}"
            
            # Check enum values
            if result["inclusion_decision"] not in ["Included", "Excluded"]:
                return False, f"Invalid inclusion_decision: {result['inclusion_decision']}"
            
            valid_categories = ["Client", "FLW", "Feasibility", "Data", "Grey Literature", "N/A"]
            if result["category"] not in valid_categories:
                return False, f"Invalid category: {result['category']}"
            
            # Logic validation: excluded papers should have N/A category
            if result["inclusion_decision"] == "Excluded" and result["category"] != "N/A":
                logger.warning(f"Excluded paper has non-N/A category: {result['category']}, correcting to N/A")
                result["category"] = "N/A"
            
            return True, None
            
        except Exception as e:
            return False, f"Validation error: {str(e)}" 
