"""
Test script for multi-tools drug research
Tests with Exa, Trafilatura, DuckDuckGo tools enabled
"""
import sys
import traceback

from multi_tools_search import DrugResearchInput, InputDrivenDrugResearchWorkflow

# Test parameters
test_input = DrugResearchInput(
    drug_name="Dupixent",
    manufacturer="Sanofi/Regeneron",
    generic_name="dupilumab",
    target_month="October",
    target_year="2025",
    therapeutic_area="atopic dermatitis"
)

print("🔬 MULTI-TOOLS DRUG RESEARCH SYSTEM 🔬")
print("📊 Testing with Dupixent (dupilumab) - October 2025")
print(f"\n📋 Research Parameters:")
print(f"   Drug: {test_input.drug_name}")
print(f"   Generic: {test_input.generic_name}")
print(f"   Manufacturer: {test_input.manufacturer}")
print(f"   Period: {test_input.target_month} {test_input.target_year}")
print(f"   Therapeutic Area: {test_input.therapeutic_area}")
print("\n🚀 Starting multi-tool research workflow...")
print("🛠️  Tools Enabled: Tavily, Exa, Trafilatura, DuckDuckGo")
print()

try:
    # Execute research workflow
    workflow = InputDrivenDrugResearchWorkflow()
    result = workflow.run(test_input)

    print("\n" + "="*80)
    print("RESEARCH COMPLETE!")
    print("="*80)
    print(result)
    
    print("\n📁 Check individual agent outputs in: agent_outputs/")
    print("📄 Check CSV file for final results")
    
except Exception as e:
    print(f"\n❌ ERROR: {str(e)}")
    print("\nTraceback:")
    traceback.print_exc()
    sys.exit(1)

