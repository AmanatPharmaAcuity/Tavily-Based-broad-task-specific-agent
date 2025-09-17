"""
Enhanced Tavily Drug Research Team with Structured Table Output
7 specialized drug research agents + support agents using Tavily
Input-driven research for specific drugs with temporal constraints
Outputs structured table format as requested
"""

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.tavily import TavilyTools
from agno.team import Team
from agno.os import AgentOS
from agno.workflow import Workflow
from pydantic import BaseModel, Field
from typing import Optional
import os
import csv
from datetime import datetime

# ========== ENVIRONMENT SETUP ==========
os.environ['OPENAI_API_KEY'] = ''
os.environ["TAVILY_API_KEY"]=""

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

# ========== OUTPUT FORMATTER ==========
def format_to_structured_table(content: str, research_input) -> str:
    """Convert research content to structured table format"""
    
    # Create CSV content
    csv_content = """Category,Sub Category,Date,URL,Drug Name,Generic Name,Manufacturer,Disease Name,Development Summary,Detailed Description,Country,Competitive Implication,Patient Population Affected
"""
    
    # Create markdown table header
    markdown_table = """
# Comprehensive Drug Research Report - Structured Output

## Research Parameters
- **Drug Name:** {drug_name}
- **Generic Name:** {generic_name}
- **Manufacturer:** {manufacturer}
- **Research Period:** {target_month} {target_year}
- **Therapeutic Area:** {therapeutic_area}

## Research Results Table

| Category | Sub Category | Date | URL | Drug Name | Generic Name | Manufacturer | Disease Name | Development Summary | Detailed Description | Country | Competitive Implication | Patient Population Affected |
|----------|--------------|------|-----|-----------|--------------|--------------|--------------|-------------------|-------------------|---------|----------------------|---------------------------|
""".format(
        drug_name=research_input.drug_name,
        generic_name=research_input.generic_name or 'Not specified',
        manufacturer=research_input.manufacturer,
        target_month=research_input.target_month,
        target_year=research_input.target_year,
        therapeutic_area=research_input.therapeutic_area or 'Not specified'
    )
    
    # Add the research content as structured entries
    markdown_table += content + "\n"
    
    # Save CSV file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"{research_input.drug_name.replace(' ', '_')}_{research_input.manufacturer.replace(' ', '_')}_{timestamp}.csv"
    
    try:
        with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
            f.write(csv_content)
            f.write(content.replace('|', ','))  # Simple conversion for CSV
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

| Category | Sub Category | Date | URL | Drug Name | Generic Name | Manufacturer | Disease Name | Development Summary | Detailed Description | Country | Competitive Implication | Patient Population Affected |

MANDATORY OUTPUT REQUIREMENTS:
1. Each finding must be ONE table row in the above format
2. Use pipe (|) separators between columns
3. Fill ALL columns for each row - use "Not Available" if information is missing
4. Detailed Description must be comprehensive (100+ words capturing all relevant information)
5. Date format: YYYY-MM-DD or Month YYYY
6. URL: Include actual source URLs found during research
7. Country: Specify US, Global, EU, or specific country
8. Provide multiple rows if you find multiple relevant findings

EXAMPLE ROW FORMAT:
| Market Research | Revenue Analysis | March 2024 | https://example.com/news | Ozempic | semaglutide | Novo Nordisk | Type 2 Diabetes | Q1 2024 revenue increased 25% | Novo Nordisk reported significant revenue growth for Ozempic in Q1 2024, with sales reaching $2.1 billion, representing a 25% increase compared to the same period last year. The growth was attributed to increased adoption in both diabetes and weight management indications, expanding market penetration in the US and European markets, and successful formulary placements with major insurance providers. The company also noted strong patient uptake following recent clinical data supporting cardiovascular benefits. | Global | Strengthens Novo Nordisk's market leadership in GLP-1 agonists | Approximately 15 million diabetes patients globally |

"""

# ========== INITIALIZE TOOLS ==========
tavily_tools = TavilyTools(
   
)

# ========== 7 SPECIALIZED DRUG RESEARCH AGENTS ==========

# 1. Market Research Agent
market_research_agent = Agent(
    name="Drug Market Research Specialist",
    model=OpenAIChat(id="gpt-4o"),
    tools=[tavily_tools],
    instructions=[
        "You are a pharmaceutical market research specialist with STRICT INPUT ADHERENCE:",
        "",
        "MANDATORY REQUIREMENTS:",
        "1. ONLY search for the EXACT drug name, manufacturer, and generic name provided",
        "2. ONLY search for data from the SPECIFIC month and year provided",
        "3. If no data exists for that exact month/year, clearly state this",
        "4. Do NOT search for similar drugs or different time periods",
        "",
        "YOUR RESEARCH FOCUS:",
        "1. Market size, revenue, and growth trends for the SPECIFIC drug",
        "2. Competitor analysis and market share data for that EXACT time period",
        "3. Pricing strategies and market access for the specified drug",
        "4. Market forecasts specifically mentioning the target drug",
        "5. Regulatory approvals affecting the specified drug in that time frame",
        "",
        STRUCTURED_OUTPUT_INSTRUCTIONS,
        "",
        "YOUR CATEGORY: Market Research",
        "YOUR SUB-CATEGORIES: Revenue Analysis, Market Share, Pricing Strategy, Growth Trends, Market Access"
    ],
    markdown=True,
)

# 2. Clinical Trials Research Agent
clinical_trials_agent = Agent(
    name="Clinical Trials Research Specialist", 
    model=OpenAIChat(id="gpt-4o"),
    tools=[tavily_tools],
    instructions=[
        "You are a clinical trials research specialist with STRICT INPUT ADHERENCE:",
        "",
        "MANDATORY REQUIREMENTS:",
        "1. ONLY search for trials involving the EXACT drug name provided",
        "2. ONLY search for trial data from the SPECIFIC month and year provided",
        "3. Include manufacturer name in searches to avoid confusion with similar drugs",
        "4. If using generic name, ensure it matches the specified drug exactly",
        "",
        "YOUR RESEARCH FOCUS:",
        "1. Clinical trials for the SPECIFIC drug in the target time period",
        "2. Trial results and data published/updated in that exact month/year",
        "3. FDA submissions and regulatory filings for that specific time frame",
        "4. Trial phase updates and recruitment status from that period",
        "5. Principal investigator announcements for the specified drug/time",
        "",
        STRUCTURED_OUTPUT_INSTRUCTIONS,
        "",
        "YOUR CATEGORY: Clinical Trials",
        "YOUR SUB-CATEGORIES: Phase I Results, Phase II Results, Phase III Results, FDA Submissions, Trial Recruitment, Endpoints Analysis"
    ],
    markdown=True,
)

# 3. Copay & Insurance Coverage Agent
copay_coverage_agent = Agent(
    name="Drug Coverage & Copay Specialist",
    model=OpenAIChat(id="gpt-4o"),
    tools=[tavily_tools],
    instructions=[
        "You are a drug coverage and copay research specialist with STRICT INPUT ADHERENCE:",
        "",
        "MANDATORY REQUIREMENTS:",
        "1. ONLY research coverage for the EXACT drug name provided",
        "2. ONLY search for coverage changes/updates from the SPECIFIC month and year",
        "3. Include manufacturer name to distinguish from similar drugs",
        "4. Focus exclusively on the specified time period for coverage data",
        "",
        "YOUR RESEARCH FOCUS:",
        "1. Insurance formulary changes for the specific drug in target month/year",
        "2. Copay assistance program updates from that exact time period",
        "3. Medicare/Medicaid coverage policy changes for that drug/timeframe",
        "4. Prior authorization requirement updates in the specified period",
        "5. Patient access program announcements from that month/year",
        "",
        STRUCTURED_OUTPUT_INSTRUCTIONS,
        "",
        "YOUR CATEGORY: Coverage & Access",
        "YOUR SUB-CATEGORIES: Formulary Updates, Copay Programs, Medicare Coverage, Prior Authorization, Patient Access Programs"
    ],
    markdown=True,
)

# 4. Breakthrough Drugs & Innovation Agent
breakthrough_agent = Agent(
    name="Drug Breakthrough & Innovation Specialist",
    model=OpenAIChat(id="gpt-4o"), 
    tools=[tavily_tools],
    instructions=[
        "You are a breakthrough drugs and innovation specialist with STRICT INPUT ADHERENCE:",
        "",
        "MANDATORY REQUIREMENTS:",
        "1. ONLY search for breakthrough designations for the EXACT drug specified",
        "2. ONLY search for announcements/designations from the SPECIFIC month and year",
        "3. Include manufacturer name to ensure correct drug identification",
        "4. Focus exclusively on the specified drug and time period",
        "",
        "YOUR RESEARCH FOCUS:",
        "1. FDA breakthrough therapy designations for the specific drug in target period",
        "2. Orphan drug designations announced in that exact month/year",
        "3. Accelerated approval pathway updates for the specified drug/timeframe",
        "4. Scientific publication mentions of the drug from that time period",
        "5. Innovation awards or recognition for the specific drug in that timeframe",
        "",
        STRUCTURED_OUTPUT_INSTRUCTIONS,
        "",
        "YOUR CATEGORY: Innovation & Breakthroughs",
        "YOUR SUB-CATEGORIES: Breakthrough Designation, Orphan Status, Fast Track, Priority Review, Scientific Publications"
    ],
    markdown=True,
)

# 5. Regulatory & Compliance Agent
regulatory_agent = Agent(
    name="Drug Regulatory & Compliance Specialist",
    model=OpenAIChat(id="gpt-4o"),
    tools=[tavily_tools],
    instructions=[
        "You are a pharmaceutical regulatory specialist with STRICT INPUT ADHERENCE:",
        "",
        "MANDATORY REQUIREMENTS:",
        "1. ONLY search for regulatory updates for the EXACT drug specified",
        "2. ONLY search for regulatory actions from the SPECIFIC month and year",
        "3. Include manufacturer name in all searches for precise identification",
        "4. Focus exclusively on official regulatory sources for that timeframe",
        "",
        "YOUR RESEARCH FOCUS:",
        "1. FDA approvals/rejections for the specific drug in target month/year",
        "2. Regulatory guidance updates affecting the drug in that period",
        "3. Safety communications specific to the drug from that timeframe",
        "4. Manufacturing compliance issues for the drug/manufacturer in that period",
        "5. REMS program updates for the specific drug in target timeframe",
        "",
        STRUCTURED_OUTPUT_INSTRUCTIONS,
        "",
        "YOUR CATEGORY: Regulatory Affairs",
        "YOUR SUB-CATEGORIES: FDA Approvals, Safety Communications, Compliance Issues, REMS Updates, Regulatory Guidance"
    ],
    markdown=True,
)

# 6. Adverse Events & Safety Agent  
safety_agent = Agent(
    name="Drug Safety & Adverse Events Specialist",
    model=OpenAIChat(id="gpt-4o"),
    tools=[tavily_tools],
    instructions=[
        "You are a drug safety and adverse events specialist with STRICT INPUT ADHERENCE:",
        "",
        "MANDATORY REQUIREMENTS:",
        "1. ONLY search for safety data for the EXACT drug specified",
        "2. ONLY search for safety reports from the SPECIFIC month and year provided",
        "3. Include manufacturer name to distinguish from other similar drugs",
        "4. Focus exclusively on the specified time period for safety data",
        "",
        "YOUR RESEARCH FOCUS:",
        "1. Adverse event reports for the specific drug in target month/year",
        "2. FDA safety communications about the drug from that exact period",
        "3. Drug recall notices for the specific drug/manufacturer in that timeframe",
        "4. Safety profile updates or label changes from that period",
        "5. Pharmacovigilance data specific to the drug from that month/year",
        "",
        STRUCTURED_OUTPUT_INSTRUCTIONS,
        "",
        "YOUR CATEGORY: Safety & Adverse Events",
        "YOUR SUB-CATEGORIES: Adverse Event Reports, Safety Alerts, Drug Recalls, Label Changes, Pharmacovigilance"
    ],
    markdown=True,
)

# 7. Competitive Intelligence Agent
competitive_intel_agent = Agent(
    name="Drug Competitive Intelligence Specialist", 
    model=OpenAIChat(id="gpt-4o"),
    tools=[tavily_tools],
    instructions=[
        "You are a pharmaceutical competitive intelligence specialist with STRICT INPUT ADHERENCE:",
        "",
        "MANDATORY REQUIREMENTS:",
        "1. ONLY search for competitive intelligence about the EXACT drug specified",
        "2. ONLY search for competitive developments from the SPECIFIC month and year",
        "3. Include manufacturer name to ensure correct drug identification",
        "4. Focus on competitive activities in the specified therapeutic area and timeframe",
        "",
        "YOUR RESEARCH FOCUS:",
        "1. Competitor drug launches targeting the same indication in that period",
        "2. Patent challenges or generic competition announcements for that timeframe",
        "3. Biosimilar developments affecting the specific drug in target period",
        "4. Partnership or licensing deals involving the drug from that month/year",
        "5. Market positioning changes for the drug in the specified timeframe",
        "",
        STRUCTURED_OUTPUT_INSTRUCTIONS,
        "",
        "YOUR CATEGORY: Competitive Intelligence",
        "YOUR SUB-CATEGORIES: Competitor Launches, Generic Competition, Biosimilars, Partnerships, Market Positioning"
    ],
    markdown=True,
)

# ========== SUPPORT AGENTS WITH STRUCTURED OUTPUT ==========

# Knowledge Synthesis Agent (Table Format)
knowledge_agent = Agent(
    name="Pharmaceutical Knowledge Synthesizer",
    model=OpenAIChat(id="gpt-4o-mini"),
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
        "YOUR CATEGORY: Research Synthesis",
        "YOUR SUB-CATEGORIES: Data Integration, Cross-Analysis, Gap Analysis, Key Insights, Summary Findings",
        "",
        "Take all the research agent outputs and consolidate them into additional table rows with synthesis insights."
    ],
    markdown=True,
)

# Content Analyzer Agent (Table Format)
content_analyzer = Agent(
    name="Pharmaceutical Content Analyzer",
    model=OpenAIChat(id="gpt-4o-mini"),
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
        "YOUR CATEGORY: Content Analysis",
        "YOUR SUB-CATEGORIES: Data Quality Assessment, Trend Analysis, Risk Analysis, Opportunity Assessment, Strategic Implications",
        "",
        "Analyze the consolidated research and present insights as additional table rows."
    ],
    markdown=True,
)

# Validation Agent (Table Format)
validation_agent = Agent(
    name="Research Validation Specialist",
    model=OpenAIChat(id="gpt-4o-mini"),
    tools=[tavily_tools],
    instructions=[
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
        "YOUR CATEGORY: Validation & Quality",
        "YOUR SUB-CATEGORIES: Data Verification, Source Validation, Accuracy Check, Completeness Assessment, Quality Score",
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
            all_table_rows.append(extract_content(result))
        
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
        all_table_rows.append(extract_content(synthesis))
        
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
        all_table_rows.append(extract_content(analysis))
        
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
        all_table_rows.append(extract_content(validation))
        
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