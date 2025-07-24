"""
Code parsing utilities adapted from BARC
"""

import re
from typing import List, Tuple, Optional


def extract_code_elements(code: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Extract concept, description, and plan from generated code
    
    Returns:
        (concepts, description, plan)
    """
    # Concept extraction
    concept = None
    concept_patterns = [
        r'# concepts:\s*\n#?(.*?)(?=\n\n|\n#|\Z)',  # Multi-line format
        r'# concepts:\s*(.*?)(?=\n\n|\n#|\Z)',      # Single-line format
    ]
    
    for pattern in concept_patterns:
        concept_match = re.search(pattern, code, re.DOTALL)
        if concept_match:
            concept = concept_match.group(1).strip()
            concept = re.sub(r'\n#?\s*', ' ', concept).strip()
            break
    
    # Description extraction
    description = None
    
    # Try multiple patterns for description
    desc_patterns = [
        # Multi-line format with # on each line
        r'# description:\s*\n((?:#[^\n]*\n)+)',
        # Single line format
        r'# description:\s*(.*?)(?=\n|$)',
        # Inline format (description on same line)
        r'# description:\s*([^\n]+)',
    ]
    
    for pattern in desc_patterns:
        desc_match = re.search(pattern, code, re.DOTALL)
        if desc_match:
            if pattern == desc_patterns[0]:  # Multi-line format
                # Remove comment markers and join lines
                desc_lines = re.findall(r'#\s*(.*?)$', desc_match.group(1), re.MULTILINE)
                description = ' '.join([line.strip() for line in desc_lines if line.strip()])
            else:  # Single line formats
                description = desc_match.group(1).strip()
            break
    
    # If no explicit description tag, look for comments after concepts
    if not description:
        after_concepts_pattern = r'# concepts:.*?\n((?:#[^\n]*\n)+)'
        after_concepts_match = re.search(after_concepts_pattern, code, re.DOTALL)
        
        if after_concepts_match:
            desc_lines = re.findall(r'#\s*(.*?)$', after_concepts_match.group(1), re.MULTILINE)
            # Check if first line contains "description:"
            if desc_lines and 'description:' in desc_lines[0]:
                # Extract description from the line
                desc_text = desc_lines[0].split('description:', 1)[1].strip()
                # Add any following lines
                if len(desc_lines) > 1:
                    desc_text += ' ' + ' '.join([line.strip() for line in desc_lines[1:] if line.strip()])
                description = desc_text
            else:
                description = ' '.join([line.strip() for line in desc_lines 
                                      if line.strip() and not line.startswith('description:')])
    
    # Plan extraction
    plan = None
    plan_patterns = [
        r'# Plan:\s*\n((?:#[^\n]*\n)+)',  # Explicit Plan: tag
        r'def main\(.*?\):\s*\n\s*# Plan:\s*\n(.*?)(?=\n\s*\w)',  # Plan in main function
        r'def main\(.*?\):[^\n]*\n\s*#\s*([\d\.\s]+.*?)(?=\n\s*\w)',  # Numbered steps
    ]
    
    for pattern in plan_patterns:
        plan_match = re.search(pattern, code, re.DOTALL)
        if plan_match:
            plan_text = plan_match.group(1).strip()
            plan = re.sub(r'\n\s*#\s*', '\n', plan_text).strip()
            plan = re.sub(r'^#\s*', '', plan, flags=re.MULTILINE)
            break
    
    return concept, description, plan


def parse_code(response: str) -> List[str]:
    """Extract Python code blocks from response"""
    # Look for code blocks
    code_pattern = r'```python(.*?)```'
    matches = re.findall(code_pattern, response, re.DOTALL)
    
    if matches:
        return [match.strip() for match in matches]
    
    # If no code blocks, look for def transform/main pattern
    # Try to find the entire function definition including body
    def_patterns = [
        # Match def transform/main and everything until the next def, class, or end of string
        r'(def transform\s*\([^)]*\):[^\n]*(?:\n(?!def\s|class\s).*)*)',
        r'(def main\s*\([^)]*\):[^\n]*(?:\n(?!def\s|class\s).*)*)'
    ]
    
    for pattern in def_patterns:
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            return [match.strip() for match in matches]
    
    # Return the whole response as a single code block
    return [response.strip()] if response.strip() else []