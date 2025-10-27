"""
Enhanced Tavily Drug Research Team with Structured Table Output
7 specialized drug research agents + support agents using Tavily
Input-driven research for specific drugs with temporal constraints
Outputs structured table format as requested
"""

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.tavily import TavilyTools
#from agno.tools.brave import BraveSearch
from agno.tools.exa import ExaTools
from agno.tools.trafilatura import TrafilaturaTools  # Web scraping
from agno.tools.duckduckgo import DuckDuckGoTools  # Additional search
from agno.team import Team
from agno.os import AgentOS
from agno.workflow import Workflow
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import os
import csv
import re
from datetime import datetime
from pathlib import Path
from agno.models.google import Gemini

# ========== ENVIRONMENT SETUP ==========
os.environ['OPENAI_API_KEY'] = ''
os.environ["TAVILY_API_KEY"]="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"
os.environ['GOOGLE_API_KEY'] = 'AIzaSyC7lgYKzuRkQn28bD8eW__L-0DaqN85Spo'
os.environ['SERPER_API_KEY'] = 'deeca30593561cd44b29d68276eff5b6769ef6f8'  # Add your Serper API key for Google search
os.environ['EXA_API_KEY'] = '094d09ba-e303-4323-8150-41847df574e2'  # Add your Exa API key for deeper web search

# ========== HELPER FUNCTION ==========
def extract_content(result) -> str:
    """Extract string content from RunOutput object"""
    if hasattr(result, 'content'):
        return str(result.content)
    elif hasattr(result, 'text'):
        return str(result.text)
    elif hasattr(result, 'message'):
        return str(result.message)
    else:
        return str(result)

# ========== HELPER FUNCTIONS FOR DATA PARSING ==========

def parse_markdown_table_row(line: str) -> List[str]:
    """Parse a markdown table row into list of column values"""
    # Remove pipe symbols and split
    line = line.strip()
    if not line.startswith('|'):
        return None
    
    # Remove leading and trailing pipes
    line = line[1:-1] if line.endswith('|') else line[1:]
    
    # Split by pipe and strip whitespace
    columns = [col.strip() for col in line.split('|')]
    
    # Filter out separator rows (contain only dashes and hyphens)
    if all(re.match(r'^[\s\-:]+$', col) for col in columns):
        return None
    
    return columns


def convert_url_to_plain_text(text: str) -> str:
    """Convert markdown link format [text](url) to plain URL"""
    # Pattern for markdown links
    pattern = r'\[([^\]]+)\]\(([^)]+)\)'
    
    # Replace [text](url) with just url
    def replacer(match):
        return match.group(2)
    
    result = re.sub(pattern, replacer, text)
    return result.strip()


def clean_table_data(content: str) -> List[List[str]]:
    """Extract and clean table data from markdown content"""
    lines = content.split('\n')
    clean_rows = []
    seen_rows = set()  # Track seen rows to avoid duplicates
    
    for line in lines:
        if not line.strip():
            continue
        
        parsed = parse_markdown_table_row(line)
        if parsed and len(parsed) >= 13:  # Must have all columns
            # Clean each column: remove markdown, trim whitespace
            cleaned_columns = [convert_url_to_plain_text(col.strip()) for col in parsed]
            cleaned_columns = [col.replace('\n', ' ').replace(',', ';') for col in cleaned_columns]  # Replace newlines and commas in content
            
            # Skip header-like rows (rows that match column names exactly)
            if cleaned_columns[0].lower() in ['category', 'sub category', 'category ']:
                continue
            
            # Create a signature for the row to avoid duplicates
            row_signature = tuple(cleaned_columns[:6])  # Use first 6 columns as signature
            if row_signature not in seen_rows:
                seen_rows.add(row_signature)
                clean_rows.append(cleaned_columns)
    
    return clean_rows


# ========== OUTPUT FORMATTER ==========
def format_to_structured_table(content: str, research_input) -> str:
    """Convert research content to structured table format with CORRECT column order"""
    
    # CORRECTED COLUMN ORDER (URL is LAST):
    # Category, Sub Category, Date, Drug Name, Generic Name, Manufacturer, Disease Name, 
    # Development Summary, Detailed Description, Country, Competitive Implication, 
    # Patient Population Affected, URL
    
    header_row = """Category,Sub Category,Date,Drug Name,Generic Name,Manufacturer,Disease Name,Development Summary,Detailed Description,Country,Competitive Implication,Patient Population Affected,URL
"""
    
    # Create markdown table header (for display)
    markdown_table = """
# Comprehensive Drug Research Report - Structured Output

## Research Parameters
- **Drug Name:** {drug_name}
- **Generic Name:** {generic_name}
- **Manufacturer:** {manufacturer}
- **Research Period:** {target_month} {target_year}
- **Therapeutic Area:** {therapeutic_area}

## Research Results Table

| Category | Sub Category | Date | Drug Name | Generic Name | Manufacturer | Disease Name | Development Summary | Detailed Description | Country | Competitive Implication | Patient Population Affected | URL |
|----------|--------------|------|-----------|--------------|--------------|--------------|-------------------|-------------------|---------|----------------------|---------------------------|-----|
""".format(
        drug_name=research_input.drug_name,
        generic_name=research_input.generic_name or 'Not specified',
        manufacturer=research_input.manufacturer,
        target_month=research_input.target_month,
        target_year=research_input.target_year,
        therapeutic_area=research_input.therapeutic_area or 'Not specified'
    )
    
    # Parse and clean the table data
    parsed_rows = clean_table_data(content)
    
    # Convert to CSV format
    csv_rows = []
    for row in parsed_rows:
        if len(row) >= 13:
            # Reorder columns: Keep the structure as-is since agents should output in correct order
            csv_row = ','.join(f'"{col}"' for col in row)
            csv_rows.append(csv_row)
    
    csv_content = header_row + '\n'.join(csv_rows) + '\n'
    
    # Build markdown table
    markdown_rows = []
    for row in parsed_rows:
        if len(row) >= 13:
            # Convert to markdown format
            markdown_row = '| ' + ' | '.join(row) + ' |'
            markdown_rows.append(markdown_row)
    
    markdown_table += '\n'.join(markdown_rows)
    
    # Save CSV file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Replace special characters that cause file system errors
    safe_manufacturer = research_input.manufacturer.replace('/', '_').replace('\\', '_').replace(' ', '_')
    csv_filename = f"{research_input.drug_name.replace(' ', '_')}_{safe_manufacturer}_{timestamp}.csv"
    
    try:
        with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
            f.write(csv_content)
        csv_status = f"✅ CSV file saved: {csv_filename}"
    except Exception as e:
        csv_status = f"❌ Error saving CSV: {str(e)}"
    
    return f"{markdown_table}\n\n---\n\n## File Output\n{csv_status}"

# ========== INPUT DATA MODEL ==========
class DrugResearchInput(BaseModel):
    """Structured input for drug research queries"""
    drug_name: str = Field(description="Brand/trade name of the drug")
    manufacturer: str = Field(description="Pharmaceutical company that manufactures the drug")
    generic_name: Optional[str] = Field(default=None, description="Generic/chemical name of the drug")
    target_month: str = Field(description="Target month for research (e.g., 'January', 'February')")
    target_year: str = Field(description="Target year for research (e.g., '2024', '2025')")
    therapeutic_area: Optional[str] = Field(default=None, description="Therapeutic area or indication (optional)")

    def get_search_context(self) -> str:
        """Generate search context string for agents"""
        context = f"Drug: {self.drug_name}"
        if self.generic_name:
            context += f" (Generic: {self.generic_name})"
        context += f" | Manufacturer: {self.manufacturer}"
        context += f" | Time Period: {self.target_month} {self.target_year}"
        if self.therapeutic_area:
            context += f" | Therapeutic Area: {self.therapeutic_area}"
        return context

    def get_temporal_constraint(self) -> str:
        """Generate temporal constraint for searches"""
        return f"Only search for information from {self.target_month} {self.target_year} or specify if data is from a different time period"

# ========== STRUCTURED OUTPUT INSTRUCTIONS ==========
STRUCTURED_OUTPUT_INSTRUCTIONS = """

CRITICAL: Format ALL your findings as table rows using this EXACT format:

| Category | Sub Category | Date | Drug Name | Generic Name | Manufacturer | Disease Name | Development Summary | Detailed Description | Country | Competitive Implication | Patient Population Affected | URL |

NOTE: URL is the LAST column (13th position)!

MANDATORY OUTPUT REQUIREMENTS:
1. Each finding must be ONE table row in the above format
2. Use pipe (|) separators between columns
3. Fill ALL columns for each row - use "Not Available" if information is missing
4. Detailed Description must be comprehensive (100-200 words capturing ALL relevant information from the source)
5. Date format: STRICTLY use YYYY-MM-DD format ONLY (e.g., 2025-10-15). NEVER use "Month YYYY" format. If exact day is unknown, use first day of month (e.g., 2025-10-01)
6. URL: Include ONLY working, accessible URLs - Verify URLs work before including. Use direct links to PDFs or primary sources when available. If URL doesn't work, mark as "Not Available"
7. Country: Specify US, Canada, or "US, Canada" - be specific
8. Competitive Implication: Provide strategic analysis (2-4 sentences) focusing on impact on key competitors (Dupixent, Adbry, Opzelura, Ebglyss, etc.). If document contains no competitive implication, write "No competitive implication stated."
9. Patient Population Affected: Use exact phrasing from document (e.g., "Adults ≥18 with moderate-to-severe atopic dermatitis inadequately controlled by topical corticosteroids")
10. Provide multiple rows if you find multiple relevant findings
11. Date Extraction Priority: Use effective date from document, then publication date, then regulatory action date - ALL must be within specified month/year

EXAMPLE ROW FORMAT:
| Marketed Assets | Label Updates | 2025-09-18 | Opzelura | ruxolitinib | Incyte | atopic dermatitis | FDA approved expanded indication for children ages 2-11 | FDA approved supplemental New Drug Application (sNDA) for Opzelura (ruxolitinib) cream 1.5% for short-term and non-continuous chronic treatment of mild to moderate atopic dermatitis in non-immunocompromised children 2 years and older. Approval based on pivotal Phase 3 TRuE-AD3 trial (NCT04921969) that met primary endpoint of IGA-TS and secondary endpoint of EASI75. Safety profile consistent with previous data with no new safety signals. Most common adverse reaction was upper respiratory tract infection. Regulatory driver: sNDA approval following successful Phase 3 pediatric trial results. | United States | Significant competitive advantage for Opzelura over Dupixent, Adbry, and other systemic agents by expanding into younger pediatric population where topical alternatives are preferred. First JAK inhibitor approved for this age group in AD, potentially capturing market share from topical corticosteroids and positioning favorably against biologics requiring injections in young children. | Non-immunocompromised children ages 2-11 years with mild to moderate atopic dermatitis whose disease is not well controlled with topical prescription therapies or when those therapies are not recommended | https://www.drugs.com/newdrugs/incyte-announces-additional-fda-approval-opzelura-ruxolitinib-cream-children-ages-2-11-atopic-6615.html |

CRITICAL CATEGORY RESTRICTIONS:
- ONLY use "Marketed Assets" as the Category for ALL findings
- ALLOWED Sub Categories (use these EXACTLY):
  * Label Updates (for FDA approvals, sNDAs, indication expansions)
  * Safety Concern (for safety warnings, adverse events, recalls)
  * Market Dynamics (for market access, formulary listings, pricing changes)
  * Guideline Update (for treatment guideline changes, care standards)
  * Clinical Data (for published study results, trial outcomes)
  * Regulatory Delay (for delayed approvals, missed PDUFA dates)
  * RWE Study (for real-world evidence, observational studies)

DO NOT use any other categories or subcategories!

CRITICAL DATE EXTRACTION RULES:
- Date MUST be in STRICT YYYY-MM-DD format (e.g., 2025-10-15)
- NEVER use "October 2025" or "Month YYYY" format
- Date Priority Order:
  1. Use effective date from document (the date the change became effective)
  2. If no effective date, use document publication/posting date
  3. If no explicit date, use regulatory action date mentioned
  4. If exact day unknown but month confirmed, use first day (e.g., 2025-10-01)
- All dates MUST fall within the specified target month
- Cross-reference multiple date fields in source documents
- Flag and exclude any dates outside the target period

URL VALIDATION REQUIREMENTS:
- ONLY include URLs that are accessible and lead to the actual document
- Prefer direct links to PDFs or primary sources
- For press releases referencing documents, include primary PDF URL
- Test URLs before including - if a URL doesn't work, mark as "Not Available"
- For multiple URLs: put primary first, append others in parentheses
- Never include URLs to documents that don't exist or return 404

EVIDENCE EXTRACTION STANDARDS:
- Extract ONLY facts explicitly present in the document
- Do NOT infer, speculate, or add interpretation
- For each field, capture direct quotes when possible
- Include page/paragraph reference for PDFs (e.g., "see p.3, item 2")
- Maintain source URL and retrieval timestamp
- If document references multiple drugs/developments, create separate row for each

CLINICAL TRIAL REPORTING STANDARDS (if applicable):
1. Maintain boundaries between different trials - format: "[STUDY NAME] (N=[sample size]): [finding]"
2. ALL efficacy/safety results MUST include comparator: "X% treatment group vs. Y% comparator group"
3. When source mentions multiple studies, extract each study's data separately with clear labeling

"""

# ========== INITIALIZE TOOLS ==========
# Multiple search tools for comprehensive research
tavily_tools = TavilyTools()

# Additional search tools - Uncomment when API keys are available
# Get API keys from:
# - Serper API: https://serper.dev (for Exa if using Serper)
# - Exa AI: https://exa.ai
# - Brave: https://brave.com/search/api
# bravesearch_tools = None  # BraveSearch() - Uncomment when API key available
# exa_tools = None  # ExaTools() - Uncomment when API key available  
# scraping_tools = None  # TrafilaturaTools() - Free, no API key needed
# duckduckgo_tools = None  # DuckDuckGoTools() - Free, no API key needed

#  Add your API keys to environment variables above (lines 30-31)
# Then uncomment the tools you want to use below:

# Option 1: Enable scraping (free, no API key needed)
scraping_tools = TrafilaturaTools()

# Option 2: Enable DuckDuckGo search (free, no API key needed)
duckduckgo_tools = DuckDuckGoTools()

# Option 3: Add API keys and enable Brave/Exa
# bravesearch_tools = BraveSearch()  # Need BRAVE_API_KEY
exa_tools = ExaTools()  # Need EXA_API_KEY or EXA_API_KEY env variable

# ========== 7 SPECIALIZED DRUG RESEARCH AGENTS ==========

# Enhanced Research Instructions with Deep Search Strategy
ENHANCED_RESEARCH_INSTRUCTIONS = """

CRITICAL: You MUST conduct multiple, varied searches REGARDLESS OF THE DATE PROVIDED. Use ALL available tools for deep research.

AVAILABLE TOOLS - USE THEM ALL:
1. Tavily Tools - Primary search engine
2. Exa Tools - Deep semantic search with content understanding
3. Trafilatura Tools - Web scraping for full content extraction
4. DuckDuckGo Tools - Additional search engine with unique results

SEARCH STRATEGY:

1. PRIMARY SEARCHES (Use Tavily + Exa + DuckDuckGo):
   - Search: "drug name" + "event type" + "month year"
   - Company press releases: site:company.com + "drug name" + date
   - Regulatory sites: site:fda.gov + "drug name" + date
   - Search with filetype:pdf for downloadable documents
   - USE EXA for semantic search to find conceptually related content
   - USE TRAFILATURA to scrape full content from URLs you find

2. SECONDARY SEARCHES (If primary fails):
   - Use alternative date formats: "Oct 2024", "2024-10", "October 20, 2024"
   - Search with generic name if brand name yields no results
   - Try competitor comparisons: "drug name vs competitor" + date
   - Look for conference abstracts: "drug name" + "conference" + date

3. DEEP CONTENT EXTRACTION:
   - When you find a relevant URL, USE TRAFILATURA to scrape full content
   - Look for PDFs: "drug name" filetype:pdf + date
   - Check regulatory databases: FDA, Health Canada, ClinicalTrials.gov
   - Search medical journals: site:nejm.org OR site:jama.com + "drug name" + date
   - USE EXA to find semantically similar documents

4. MULTI-TOOL VALIDATION:
   - Cross-reference findings from multiple tools
   - Use Tavily for broad search, Exa for deep understanding, DuckDuckGo for alternative sources
   - Use Trafilatura to extract full content from promising URLs
   - If you find information from one tool, verify with another tool

5. HANDLING "NO DATA FOUND":
   - MUST try ALL tools (Tavily, Exa, DuckDuckGo) before reporting "no data"
   - USE TRAFILATURA to scrape any URLs you find
   - Try at least 5-10 different search strategies across all tools
   - NEVER assume "no data" just because of the date - ACTUALLY USE ALL TOOLS
   - Look for press releases, filings, or announcements near the date
   - Check if the event happened but was published slightly later
   - Only report "no data found" if ALL tools and searches return empty results

MANDATORY: Before reporting "no data", you must:
- Have used Tavily search at least 3 times with different queries
- Have used Exa search at least 2 times with different queries
- Have used DuckDuckGo search at least 2 times with different queries
- Have attempted scraping URLs if any were found

"""

# 1. Market Research Agent
market_research_agent = Agent(
    name="Drug Market Research Specialist",
    model=Gemini(id="gemini-2.5-pro", thinking_budget=1280, include_thoughts=True),
    tools=[tavily_tools, exa_tools, scraping_tools, duckduckgo_tools],
    instructions=[
        "CRITICAL: NEVER refuse to search based on date. Use all available tools (Tavily, Exa, scraping, DuckDuckGo) to conduct deep searches. If the user asks for October 2025 data, you MUST search using these tools and report results.",
        "",
        ENHANCED_RESEARCH_INSTRUCTIONS,
        "",
        "You are a pharmaceutical market research specialist with STRICT INPUT ADHERENCE:",
        "",
        "MANDATORY REQUIREMENTS:",
        "1. ONLY search for the EXACT drug name, manufacturer, and generic name provided",
        "2. Conduct comprehensive searches for the SPECIFIC month and year provided - search tools may find data regardless of date",
        "3. If searches return no results, then clearly state this - but DO NOT assume no data exists just because it's a 'future' date",
        "4. Do NOT search for similar drugs or different time periods",
        "",
        "YOUR RESEARCH FOCUS:",
        "1. Market size, revenue, and growth trends for the SPECIFIC drug",
        "2. Competitor analysis and market share data for that EXACT time period",
        "3. Pricing strategies and market access for the specified drug",
        "4. Market forecasts specifically mentioning the target drug",
        "5. Regulatory approvals affecting the specified drug in that time frame",
        "",
        "ENHANCED SEARCH STRATEGIES:",
        "- Search company press releases and investor presentations (site:company.com + 'press release' + date)",
        "- Look for FDA/Health Canada notices, label changes, safety communications",
        "- Search for payer/insurance documents (copay, formulary, medical policy, coverage)",
        "- Include SEC filings (site:sec.gov) and HTA documents",
        "- Use filetype:pdf to find downloadable documents",
        "- For Market Dynamics: search formulary listings, PBM memos, coverage changes"
        "",
        STRUCTURED_OUTPUT_INSTRUCTIONS,
        "",
        "MANDATORY: Use ONLY 'Marketed Assets' as Category",
        "Use appropriate subcategories: Label Updates, Safety Concern, Market Dynamics, Guideline Update, Clinical Data, Regulatory Delay, or RWE Study"
    ],
    markdown=True,
)

# 2. Clinical Trials Research Agent
clinical_trials_agent = Agent(
    name="Clinical Trials Research Specialist", 
    model=Gemini(id="gemini-2.5-pro", thinking_budget=1280, include_thoughts=True),
    tools=[tavily_tools, exa_tools, scraping_tools, duckduckgo_tools],
    instructions=[
        "CRITICAL: NEVER refuse to search based on date. Use all available tools (Tavily, Exa, scraping, DuckDuckGo) to conduct deep searches. If the user asks for October 2025 data, you MUST search using these tools and report results.",
        "",
        "You are a clinical trials research specialist with STRICT INPUT ADHERENCE:",
        "",
        "MANDATORY REQUIREMENTS:",
        "1. ONLY search for trials involving the EXACT drug name provided",
        "2. Conduct comprehensive searches for the SPECIFIC month and year - use search tools without date assumptions",
        "3. Include manufacturer name in searches to avoid confusion with similar drugs",
        "4. If using generic name, ensure it matches the specified drug exactly",
        "5. DO NOT assume data doesn't exist - SEARCH FIRST, then report results",
        "YOUR RESEARCH FOCUS:",
        "1. Clinical trials for the SPECIFIC drug in the target time period",
        "2. Trial results and data published/updated in that exact month/year",
        "3. FDA submissions and regulatory filings for that specific time frame",
        "4. Trial phase updates and recruitment status from that period",
        "5. Principal investigator announcements for the specified drug/time",
        "",
        "ENHANCED SEARCH STRATEGIES:",
        "- Check site:clinicaltrials.gov for trial updates with specific date range",
        "- Search medical journals (NEJM, JAMA, Lancet) and conference abstracts",
        "- Look for published results, interim data, readouts from specific time period",
        "- Search for pivotal trials, Phase 3 results, primary endpoints",
        "- For Clinical Data subcategory: focus on published peer-reviewed results"
        "",
        STRUCTURED_OUTPUT_INSTRUCTIONS,
        "",
        "MANDATORY: Use ONLY 'Marketed Assets' as Category",
        "Use appropriate subcategories: Label Updates, Safety Concern, Market Dynamics, Guideline Update, Clinical Data, Regulatory Delay, or RWE Study"
    ],
    markdown=True,
)

# 3. Copay & Insurance Coverage Agent
copay_coverage_agent = Agent(
    name="Drug Coverage & Copay Specialist",
    model=Gemini(id="gemini-2.5-pro", thinking_budget=1280, include_thoughts=True),
    tools=[tavily_tools, exa_tools, scraping_tools, duckduckgo_tools],
    instructions=[
        "CRITICAL: NEVER refuse to search based on date. Use all available tools (Tavily, Exa, scraping, DuckDuckGo) to conduct deep searches. If the user asks for October 2025 data, you MUST search using these tools and report results.",
        "",
        "You are a drug coverage and copay research specialist with STRICT INPUT ADHERENCE:",
        "",
        "MANDATORY REQUIREMENTS:",
        "1. ONLY research coverage for the EXACT drug name provided",
        "2. Conduct comprehensive searches for the SPECIFIC month and year provided",
        "3. Include manufacturer name to distinguish from similar drugs",
        "4. DO NOT reject searches based on date - let search tools find data if it exists",
        "",
        "YOUR RESEARCH FOCUS:",
        "1. Insurance formulary changes for the specific drug in target month/year",
        "2. Copay assistance program updates from that exact time period",
        "3. Medicare/Medicaid coverage policy changes for that drug/timeframe",
        "4. Prior authorization requirement updates in the specified period",
        "5. Patient access program announcements from that month/year",
        "",
        "ENHANCED SEARCH STRATEGIES:",
        "- Search PBM documents (site:express-scripts.com OR site:optum.com OR site:cvs.com + 'medical policy')",
        "- Look for CMS/Medicaid coverage memos and provincial formulary updates",
        "- Search for copay cards, patient assistance programs, reimbursement policies",
        "- Include payer bulletins, coverage criteria, formulary tier changes",
        "- Use terms: 'copay', 'reimbursement memo', 'medical policy', 'formulary decision', 'coverage criteria'",
        "- Search filetype:pdf for downloadable policy documents"
        "",
        STRUCTURED_OUTPUT_INSTRUCTIONS,
        "",
        "MANDATORY: Use ONLY 'Marketed Assets' as Category",
        "Use appropriate subcategories: Label Updates, Safety Concern, Market Dynamics, Guideline Update, Clinical Data, Regulatory Delay, or RWE Study"
    ],
    markdown=True,
)

# 4. Breakthrough Drugs & Innovation Agent
breakthrough_agent = Agent(
    name="Drug Breakthrough & Innovation Specialist",
    model=Gemini(id="gemini-2.5-pro", thinking_budget=1280, include_thoughts=True), 
    tools=[tavily_tools, exa_tools, scraping_tools, duckduckgo_tools],
    instructions=[
        "CRITICAL: NEVER refuse to search based on date. Use all available tools (Tavily, Exa, scraping, DuckDuckGo) to conduct deep searches. If the user asks for October 2025 data, you MUST search using these tools and report results.",
        "",
        "You are a breakthrough drugs and innovation specialist with STRICT INPUT ADHERENCE:",
        "",
        "MANDATORY REQUIREMENTS:",
        "1. ONLY search for breakthrough designations for the EXACT drug specified",
        "2. Conduct comprehensive searches for the SPECIFIC month and year using search tools",
        "3. Include manufacturer name to ensure correct drug identification",
        "4. DO NOT assume future dates have no data - SEARCH and find results if they exist",
        "",
        "YOUR RESEARCH FOCUS:",
        "1. FDA breakthrough therapy designations for the specific drug in target period",
        "2. Orphan drug designations announced in that exact month/year",
        "3. Accelerated approval pathway updates for the specified drug/timeframe",
        "4. Scientific publication mentions of the drug from that time period",
        "5. Innovation awards or recognition for the specific drug in that timeframe",
        "",
        "ENHANCED SEARCH STRATEGIES:",
        "- Search FDA BLA/NDA approval letters and breakthrough designations",
        "- Look for FDA approvals, PDUFA dates, regulatory communications",
        "- Search site:fda.gov + drug name + date for label updates",
        "- Check for orphan drug designations (site:fda.gov/rare-diseases)",
        "- For Label Updates subcategory: focus on FDA approval actions and label changes"
        "",
        STRUCTURED_OUTPUT_INSTRUCTIONS,
        "",
        "MANDATORY: Use ONLY 'Marketed Assets' as Category",
        "Use appropriate subcategories: Label Updates, Safety Concern, Market Dynamics, Guideline Update, Clinical Data, Regulatory Delay, or RWE Study"
    ],
    markdown=True,
)

# 5. Regulatory & Compliance Agent
regulatory_agent = Agent(
    name="Drug Regulatory & Compliance Specialist",
    model=Gemini(id="gemini-2.5-pro", thinking_budget=1280, include_thoughts=True),
    tools=[tavily_tools, exa_tools, scraping_tools, duckduckgo_tools],
    instructions=[
        "CRITICAL: NEVER refuse to search based on date. Use all available tools (Tavily, Exa, scraping, DuckDuckGo) to conduct deep searches. If the user asks for October 2025 data, you MUST search using these tools and report results.",
        "",
        "You are a pharmaceutical regulatory specialist with STRICT INPUT ADHERENCE:",
        "",
        "MANDATORY REQUIREMENTS:",
        "1. ONLY search for regulatory updates for the EXACT drug specified",
        "2. Conduct comprehensive searches for the SPECIFIC month and year",
        "3. Include manufacturer name in all searches for precise identification",
        "4. Search regulatory sources without date-based assumptions",
        "",
        "YOUR RESEARCH FOCUS:",
        "1. FDA approvals/rejections for the specific drug in target month/year",
        "2. Regulatory guidance updates affecting the drug in that period",
        "3. Safety communications specific to the drug from that timeframe",
        "4. Manufacturing compliance issues for the drug/manufacturer in that period",
        "5. REMS program updates for the specific drug in target timeframe",
        "",
        "ENHANCED SEARCH STRATEGIES:",
        "- Search site:fda.gov for safety communications, MedWatch alerts",
        "- Look for 'Dear Healthcare Provider' letters and FDA warnings",
        "- Check for product monograph updates, prescribing information changes",
        "- Search Health Canada notices (site:healthycanadians.gc.ca)",
        "- For Safety Concern subcategory: focus on adverse events, recalls, warnings"
        "",
        STRUCTURED_OUTPUT_INSTRUCTIONS,
        "",
        "MANDATORY: Use ONLY 'Marketed Assets' as Category",
        "Use appropriate subcategories: Label Updates, Safety Concern, Market Dynamics, Guideline Update, Clinical Data, Regulatory Delay, or RWE Study"
    ],
    markdown=True,
)

# 6. Adverse Events & Safety Agent  
safety_agent = Agent(
    name="Drug Safety & Adverse Events Specialist",
    model=Gemini(id="gemini-2.5-pro", thinking_budget=1280, include_thoughts=True),
    tools=[tavily_tools, exa_tools, scraping_tools, duckduckgo_tools],
    instructions=[
        "CRITICAL: NEVER refuse to search based on date. Use all available tools (Tavily, Exa, scraping, DuckDuckGo) to conduct deep searches. If the user asks for October 2025 data, you MUST search using these tools and report results.",
        "",
        "You are a drug safety and adverse events specialist with STRICT INPUT ADHERENCE:",
        "",
        "MANDATORY REQUIREMENTS:",
        "1. ONLY search for safety data for the EXACT drug specified",
        "2. Conduct comprehensive searches using the provided month and year",
        "3. Include manufacturer name to distinguish from other similar drugs",
        "4. Use search tools without date restrictions - find data if it exists",
        "",
        "YOUR RESEARCH FOCUS:",
        "1. Adverse event reports for the specific drug in target month/year",
        "2. FDA safety communications about the drug from that exact period",
        "3. Drug recall notices for the specific drug/manufacturer in that timeframe",
        "4. Safety profile updates or label changes from that period",
        "5. Pharmacovigilance data specific to the drug from that month/year",
        "",
        "ENHANCED SEARCH STRATEGIES:",
        "- Search FDA MedWatch, Health Canada advisories for that specific period",
        "- Look for safety data sheets, pharmacovigilance reports",
        "- Check for drug recalls, manufacturing issues, safety labeling changes",
        "- Search terms: 'safety', 'adverse event', 'warning', 'precaution', 'MedWatch'",
        "- For Safety Concern subcategory: focus on documented safety issues and warnings"
        "",
        STRUCTURED_OUTPUT_INSTRUCTIONS,
        "",
        "MANDATORY: Use ONLY 'Marketed Assets' as Category",
        "Use appropriate subcategories: Label Updates, Safety Concern, Market Dynamics, Guideline Update, Clinical Data, Regulatory Delay, or RWE Study"
    ],
    markdown=True,
)

# 7. Competitive Intelligence Agent
competitive_intel_agent = Agent(
    name="Drug Competitive Intelligence Specialist", 
    model=Gemini(id="gemini-2.5-pro", thinking_budget=1280, include_thoughts=True),
    tools=[tavily_tools, exa_tools, scraping_tools, duckduckgo_tools],
    instructions=[
        "CRITICAL: NEVER refuse to search based on date. Use all available tools (Tavily, Exa, scraping, DuckDuckGo) to conduct deep searches. If the user asks for October 2025 data, you MUST search using these tools and report results.",
        "",
        "You are a pharmaceutical competitive intelligence specialist with STRICT INPUT ADHERENCE:",
        "",
        "MANDATORY REQUIREMENTS:",
        "1. ONLY search for competitive intelligence about the EXACT drug specified",
        "2. Conduct comprehensive searches for the SPECIFIC month and year provided",
        "3. Include manufacturer name to ensure correct drug identification",
        "4. DO NOT reject searches based on date assumptions - let search tools work",
        "",
        "YOUR RESEARCH FOCUS:",
        "1. Competitor drug launches targeting the same indication in that period",
        "2. Patent challenges or generic competition announcements for that timeframe",
        "3. Biosimilar developments affecting the specific drug in target period",
        "4. Partnership or licensing deals involving the drug from that month/year",
        "5. Market positioning changes for the drug in the specified timeframe",
        "",
        "ENHANCED SEARCH STRATEGIES:",
        "- Compare against Dupixent, Adbry, Opzelura, Ebglyss, Cibinqo, Rinvoq",
        "- Look for competitor FDA approvals, market entry, new indications",
        "- Search for head-to-head studies, comparative effectiveness data",
        "- Monitor competitor websites, press releases, investor presentations",
        "- For competitive analysis: focus on how developments affect market positioning"
        "",
        STRUCTURED_OUTPUT_INSTRUCTIONS,
        "",
        "MANDATORY: Use ONLY 'Marketed Assets' as Category",
        "Use appropriate subcategories: Label Updates, Safety Concern, Market Dynamics, Guideline Update, Clinical Data, Regulatory Delay, or RWE Study"
    ],
    markdown=True,
)

# ========== SUPPORT AGENTS WITH STRUCTURED OUTPUT ==========

# Knowledge Synthesis Agent (Table Format)
knowledge_agent = Agent(
    name="Pharmaceutical Knowledge Synthesizer",
    model=Gemini(id="gemini-2.5-pro", thinking_budget=1280, include_thoughts=True),
    tools=[],
    instructions=[
        "You synthesize pharmaceutical research with STRICT ADHERENCE to user input parameters:",
        "",
        "SYNTHESIS REQUIREMENTS:",
        "1. ONLY synthesize data about the EXACT drug name specified by user",
        "2. ONLY include findings from the SPECIFIC month and year provided",
        "3. Clearly separate findings by time period if any data is from different dates",
        "4. Explicitly state when no data was found for the specified parameters",
        "",
        "CRITICAL: Convert ALL synthesis findings into the structured table format:",
        STRUCTURED_OUTPUT_INSTRUCTIONS,
        "",
        "MANDATORY: Use ONLY 'Marketed Assets' as Category for all findings",
        "Use appropriate subcategories: Label Updates, Safety Concern, Market Dynamics, Guideline Update, Clinical Data, Regulatory Delay, or RWE Study",
        "",
        "Take all the research agent outputs and consolidate them into additional table rows with synthesis insights."
    ],
    markdown=True,
)

# Content Analyzer Agent (Table Format)
content_analyzer = Agent(
    name="Pharmaceutical Content Analyzer",
    model=Gemini(id="gemini-2.5-pro", thinking_budget=1280, include_thoughts=True),
    tools=[],
    instructions=[
        "You analyze pharmaceutical research content with STRICT INPUT ADHERENCE:",
        "",
        "ANALYSIS REQUIREMENTS:",
        "1. ONLY analyze content related to the EXACT drug specified by user",
        "2. ONLY analyze data from the SPECIFIC month and year provided",
        "3. Focus analysis exclusively on the specified manufacturer's drug",
        "4. Clearly distinguish between target timeframe data and other periods",
        "",
        "CRITICAL: Present ALL analysis as structured table rows:",
        STRUCTURED_OUTPUT_INSTRUCTIONS,
        "",
        "MANDATORY: Use ONLY 'Marketed Assets' as Category for all findings",
        "Use appropriate subcategories: Label Updates, Safety Concern, Market Dynamics, Guideline Update, Clinical Data, Regulatory Delay, or RWE Study",
        "",
        "Analyze the consolidated research and present insights as additional table rows."
    ],
    markdown=True,
)

# Validation Agent (Table Format)
validation_agent = Agent(
    name="Research Validation Specialist",
    model=Gemini(id="gemini-2.5-pro", thinking_budget=1280, include_thoughts=True),
    tools=[tavily_tools, exa_tools, scraping_tools, duckduckgo_tools],
    instructions=[
        "CRITICAL: NEVER refuse to search based on date. Use all available tools (Tavily, Exa, scraping, DuckDuckGo) to conduct deep searches. If the user asks for October 2025 data, you MUST search using these tools and report results.",
        "",
        "You validate pharmaceutical research with ABSOLUTE ADHERENCE to user input:",
        "",
        "VALIDATION REQUIREMENTS:",
        "1. ONLY validate information about the EXACT drug name provided",
        "2. ONLY validate data from the SPECIFIC month and year specified",
        "3. Use additional searches ONLY for the specified drug/manufacturer/timeframe",
        "4. Flag any information that doesn't match the exact input parameters",
        "",
        "CRITICAL: Present ALL validation findings as structured table rows:",
        STRUCTURED_OUTPUT_INSTRUCTIONS,
        "",
        "MANDATORY: Use ONLY 'Marketed Assets' as Category for all findings",
        "Use appropriate subcategories: Label Updates, Safety Concern, Market Dynamics, Guideline Update, Clinical Data, Regulatory Delay, or RWE Study",
        "",
        "Validate the research findings and present validation results as additional table rows."
    ],
    markdown=True,
)

# ========== CORRECTED WORKFLOW CLASS ==========

class InputDrivenDrugResearchWorkflow(Workflow):
    def __init__(self):
        super().__init__(
            name="Input-Driven Structured Drug Research Workflow",
            description="Multi-agent pharmaceutical research with structured table output"
        )
    
    def run(self, research_input: DrugResearchInput) -> str:
        """Execute research workflow with structured table output"""
        
        print(f"🔬 Starting targeted drug research:")
        print(f"   Drug: {research_input.drug_name} ({research_input.generic_name or 'Generic name not provided'})")
        print(f"   Manufacturer: {research_input.manufacturer}")
        print(f"   Target Period: {research_input.target_month} {research_input.target_year}")
        
        # Create output directory for individual agent outputs
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_manufacturer = research_input.manufacturer.replace('/', '_').replace('\\', '_').replace(' ', '_')
        output_dir = Path(f"agent_outputs/{research_input.drug_name.replace(' ', '_')}_{safe_manufacturer}_{timestamp}")
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n📁 Saving individual agent outputs to: {output_dir}")
        
        # Create search context for all agents
        search_context = research_input.get_search_context()
        temporal_constraint = research_input.get_temporal_constraint()
        
        # Collect all structured table rows
        all_table_rows = []
        
        # Phase 1: Parallel Research with Structured Output
        agents_config = [
            ("Market Research", market_research_agent, 
             f"Research market data for {search_context}. {temporal_constraint}"),
            ("Clinical Trials", clinical_trials_agent,
             f"Research clinical trials for {search_context}. {temporal_constraint}"),
            ("Coverage & Copay", copay_coverage_agent,
             f"Research coverage information for {search_context}. {temporal_constraint}"), 
            ("Breakthrough Research", breakthrough_agent,
             f"Research breakthrough developments for {search_context}. {temporal_constraint}"),
            ("Regulatory Analysis", regulatory_agent,
             f"Research regulatory updates for {search_context}. {temporal_constraint}"),
            ("Safety Monitoring", safety_agent,
             f"Research safety information for {search_context}. {temporal_constraint}"),
            ("Competitive Intelligence", competitive_intel_agent,
             f"Research competitive intelligence for {search_context}. {temporal_constraint}")
        ]
        
        for name, agent, query in agents_config:
            print(f"📊 {name} research for {research_input.drug_name}...")
            result = agent.run(query)
            result_content = extract_content(result)
            
            # Save individual agent output to file
            safe_name = name.lower().replace(' ', '_').replace('&', 'and')
            agent_output_file = output_dir / f"{safe_name}_output.md"
            with open(agent_output_file, 'w', encoding='utf-8') as f:
                f.write(f"# {name} - Agent Output\n\n")
                f.write(f"**Drug:** {research_input.drug_name}\n")
                f.write(f"**Manufacturer:** {research_input.manufacturer}\n")
                f.write(f"**Period:** {research_input.target_month} {research_input.target_year}\n\n")
                f.write("## Research Results\n\n")
                f.write(result_content)
            
            print(f"   ✅ Saved to: {agent_output_file}")
            all_table_rows.append(result_content)
        
        # Phase 2: Knowledge Synthesis (Structured)
        print("🧠 Knowledge synthesis (structured format)...")
        synthesis_query = f"""
        Synthesize research findings in structured table format for:
        Drug: {research_input.drug_name} ({research_input.generic_name or 'generic not specified'})
        Manufacturer: {research_input.manufacturer}
        Time Period: {research_input.target_month} {research_input.target_year}
        
        Research Results:
        {chr(10).join(all_table_rows)}
        
        ONLY synthesize data matching these exact parameters and output as table rows.
        """
        synthesis = knowledge_agent.run(synthesis_query)
        synthesis_content = extract_content(synthesis)
        
        # Save synthesis output
        with open(output_dir / "knowledge_synthesis_output.md", 'w', encoding='utf-8') as f:
            f.write(f"# Knowledge Synthesis - Agent Output\n\n")
            f.write(f"**Drug:** {research_input.drug_name}\n")
            f.write(f"**Manufacturer:** {research_input.manufacturer}\n")
            f.write(f"**Period:** {research_input.target_month} {research_input.target_year}\n\n")
            f.write("## Synthesis Results\n\n")
            f.write(synthesis_content)
        
        print(f"   ✅ Saved synthesis to: {output_dir / 'knowledge_synthesis_output.md'}")
        all_table_rows.append(synthesis_content)
        
        # Phase 3: Content Analysis (Structured)  
        print("📈 Content analysis (structured format)...")
        analysis_query = f"""
        Analyze research findings in structured table format for:
        {search_context}
        Target Period: {research_input.target_month} {research_input.target_year}
        
        All Research Results:
        {chr(10).join(all_table_rows)}
        
        ONLY analyze content matching these exact parameters and output as table rows.
        """
        analysis = content_analyzer.run(analysis_query)
        analysis_content = extract_content(analysis)
        
        # Save analysis output
        with open(output_dir / "content_analysis_output.md", 'w', encoding='utf-8') as f:
            f.write(f"# Content Analysis - Agent Output\n\n")
            f.write(f"**Drug:** {research_input.drug_name}\n")
            f.write(f"**Manufacturer:** {research_input.manufacturer}\n")
            f.write(f"**Period:** {research_input.target_month} {research_input.target_year}\n\n")
            f.write("## Analysis Results\n\n")
            f.write(analysis_content)
        
        print(f"   ✅ Saved analysis to: {output_dir / 'content_analysis_output.md'}")
        all_table_rows.append(analysis_content)
        
        # Phase 4: Validation (Structured)
        print("✅ Validation (structured format)...")
        validation_query = f"""
        Validate research accuracy in structured table format for:
        {search_context}
        Target Period: {research_input.target_month} {research_input.target_year}
        
        All Analysis Results:
        {chr(10).join(all_table_rows)}
        
        Use additional searches ONLY for the specified drug/manufacturer/timeframe and output as table rows.
        """
        validation = validation_agent.run(validation_query)
        validation_content = extract_content(validation)
        
        # Save validation output
        with open(output_dir / "validation_output.md", 'w', encoding='utf-8') as f:
            f.write(f"# Validation - Agent Output\n\n")
            f.write(f"**Drug:** {research_input.drug_name}\n")
            f.write(f"**Manufacturer:** {research_input.manufacturer}\n")
            f.write(f"**Period:** {research_input.target_month} {research_input.target_year}\n\n")
            f.write("## Validation Results\n\n")
            f.write(validation_content)
        
        print(f"   ✅ Saved validation to: {output_dir / 'validation_output.md'}")
        all_table_rows.append(validation_content)
        
        # Combine all table rows
        combined_table_content = "\n".join(all_table_rows)
        
        # Format final output
        final_output = format_to_structured_table(combined_table_content, research_input)
        
        print(f"🎉 Structured research completed!")
        
        return final_output

# ========== USER INPUT INTERFACE ==========

def get_user_input() -> DrugResearchInput:
    """Interactive function to collect user input"""
    print("=== DRUG RESEARCH INPUT COLLECTION ===")
    
    drug_name = input("Enter drug name (brand/trade name): ").strip()
    manufacturer = input("Enter manufacturer name: ").strip()
    generic_name = input("Enter generic name (optional, press Enter to skip): ").strip() or None
    target_month = input("Enter target month (e.g., January, February): ").strip()
    target_year = input("Enter target year (e.g., 2024, 2025): ").strip()
    therapeutic_area = input("Enter therapeutic area (optional, press Enter to skip): ").strip() or None
    
    return DrugResearchInput(
        drug_name=drug_name,
        manufacturer=manufacturer,
        generic_name=generic_name,
        target_month=target_month,
        target_year=target_year,
        therapeutic_area=therapeutic_area
    )

# ========== AGENTOS SETUP ==========

tavily_drug_os = AgentOS(
    os_id="structured-tavily-drug-research",
    description="Structured pharmaceutical research system using Tavily with table output",
    agents=[
        market_research_agent, clinical_trials_agent, copay_coverage_agent,
        breakthrough_agent, regulatory_agent, safety_agent, 
        competitive_intel_agent, knowledge_agent, content_analyzer, validation_agent
    ],
    workflows=[InputDrivenDrugResearchWorkflow()]
)

app = tavily_drug_os.get_app()

if __name__ == "__main__":
    # Interactive usage
    print("🔬 STRUCTURED DRUG RESEARCH SYSTEM 🔬")
    print("📊 Outputs: Markdown Table + CSV File")
    
    # Get user input
    research_params = get_user_input()
    
    print(f"\n📋 Research Parameters Confirmed:")
    print(f"   Drug: {research_params.drug_name}")
    print(f"   Generic: {research_params.generic_name or 'Not specified'}")
    print(f"   Manufacturer: {research_params.manufacturer}")
    print(f"   Period: {research_params.target_month} {research_params.target_year}")
    
    # Execute research workflow with structured output
    workflow = InputDrivenDrugResearchWorkflow()
    result = workflow.run(research_params)
    print(result)