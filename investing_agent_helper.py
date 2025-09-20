# -*- coding: utf-8 -*-
"""Investing Agent.ipynb
Original file is located at
    https://colab.research.google.com/drive/1GJv0LRheNdvo567SvyEOLoqpJrwMhNPb
"""
from google.cloud.aiplatform_v1beta1 import GenerateContentResponse

# !pip install --quiet google-cloud-storage google-cloud-aiplatform google-cloud-bigquery PyPDF2 fpdf2
# !gcloud config set project tuning-machines

"""## Project Config"""

PROJECT_ID = 'tuning-machines'
# COMPANY_ANALYSED = ""


"""## Imports"""
import json
from typing import Type, TypeVar, Dict, List, Optional
# import vertexai
# from vertexai.generative_models import GenerativeModel, GenerationResponse
from pydantic import BaseModel, ValidationError, Field
from google import genai
from google.genai import types
import asyncio
from dotenv import load_dotenv
import os

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set.")
client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_ID = "gemini-2.5-flash-lite"


"""## Load (and Classify) Input Files"""
from fpdf import FPDF
import PyPDF2
import pandas as pd

def convert_txt_to_pdf(root_location: str, text_file_path: str):
    try:
        text_file_path = os.path.join(root_location, text_file_path)
        with open(text_file_path, 'r', encoding='utf-8') as f:
            text_content = f.read()

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(180, 10, text=text_content)

        file_name = os.path.basename(text_file_path)
        pdf_file_name = file_name.replace('.txt', '.pdf')
        pdf_output_path = os.path.join(root_location, pdf_file_name)
        os.remove(os.path.join(root_location, text_file_path))

        pdf.output(pdf_output_path)
        print(f"Converted {text_file_path} to {pdf_output_path}")

    except FileNotFoundError:
        print(f"Error: File not found at {text_file_path}")
    except Exception as e:
        print(f"Error converting {text_file_path}: {e}")
def fetch_all_files(file_folder_path):
    if os.path.exists(file_folder_path):
        file_list = []
        for root_location, _, files in os.walk(file_folder_path):
            company_name = os.path.basename(root_location)
            for file in files:
                if file.endswith('.txt'):
                    convert_txt_to_pdf(root_location, file)
                file_list.append({'company': company_name, 'filename': file, 'filepath': os.path.join(root_location, file)})

        if file_list:
            files_dataframe = pd.DataFrame(file_list)
            return files_dataframe
        else:
            print(f"No files found in the folder: {file_folder_path}")
            return None
    else:
        print(f"Folder not found: {file_folder_path}")
        return None
def extract_text_from_file(filepath):
    text = ""
    try:
        if filepath.lower().endswith('.pdf'):
            with open(filepath, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page_num in range(len(reader.pages)):
                    text += reader.pages[page_num].extract_text() or ""
        elif filepath.lower().endswith('.txt'):
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            text = "Unsupported file type"
    except FileNotFoundError:
        text = "Error: File not found"
    except Exception as e:
        text = f"Error reading file: {e}"
    return text
def classify_by_financial_terms(text):
    financial_keywords = ["revenue", "profit", "expenses", "EBITDA", "valuation", "financials", "balance sheet", "income statement", "cash flow"]
    text_lower = text.lower()
    for keyword in financial_keywords:
        if keyword in text_lower:
            return True
    return False
def run_classifier_on_files(files_dataframe):
  files_dataframe['classification'] = 'other'

  for index, row in files_dataframe.iterrows():
      filename = row['filename'].lower()
      if 'pitch deck' in filename or 'pitchdeck' in filename:
          files_dataframe.loc[index, 'classification'] = 'pitchdeck'
      elif 'investor deck' in filename or 'investordeck' in filename:
          files_dataframe.loc[index, 'classification'] = 'pitchdeck'
      elif 'transcript' in filename:
          files_dataframe.loc[index, 'classification'] = 'transcript'
      elif 'financial' in filename or 'financials' in filename:
          files_dataframe.loc[index, 'classification'] = 'financials'
      elif 'linkedin' in filename or 'linkedin_profile' in filename:
          files_dataframe.loc[index, 'classification'] = 'linkedin'

  files_dataframe['content'] = files_dataframe['filepath'].apply(extract_text_from_file)

  for index, row in files_dataframe.iterrows():
      if row['classification'] == 'other':
          if classify_by_financial_terms(row['content']):
              files_dataframe.loc[index, 'classification'] = 'financials'
  return files_dataframe

def fetch_all_required_files(folder_path, company_name, classification_list = []):
    files_df = fetch_all_files(folder_path)
    grouped_files_df = run_classifier_on_files(files_dataframe=files_df).groupby('company')
    company_files_df = grouped_files_df.get_group(company_name)
    if not classification_list:
        pitchdeck_and_other_pdfs = company_files_df[
            (company_files_df['filepath'].str.lower().str.endswith('.pdf'))
        ]
    else:
        pitchdeck_and_other_pdfs = company_files_df[
            (company_files_df['filepath'].str.lower().str.endswith('.pdf')) &
            (company_files_df['classification'].isin(classification_list))
            ]

    return pitchdeck_and_other_pdfs['filepath'].to_list()
    # return get_files_with_classification(grouped_files_df, company_name=company_name)

"""# Execution"""

"""## Orchestrator Implementation"""

"""### Output Schemas
**Agent 1: Founder Analyst**
*   **Primary Goal:** Analyze the founding team's composition, skills, and experience.
*   **Required Inputs:**
    *   `pitch_deck_text`: Full text extracted from the pitch deck.
    *   `founder_linkedin_urls` (Optional): A list of URLs for deeper background checks.

**Agent 2: Industry Definer**
*   **Primary Goal:** Define the startup's true industry based on its activities, not just its claims.
*   **Required Inputs:**
    *   `pitch_deck_text`: Full text from the pitch deck.
    *   `company_website_text`: Text scraped from the company's website.
    *   `investing_thesis`: A string describing the VC fund's investment thesis.

**Agent 3: Product Analyst**
*   **Primary Goal:** Deconstruct the product's value proposition and market fit.
*   **Required Inputs:**
    *   `pitch_deck_text`: Full text from the pitch deck.
    *   `product_demo_transcript` (Optional): Transcript from a product demo video or call.
"""

class FounderAnalysis(BaseModel):
    founder_count: int
    key_strengths: List[str]
    identified_gaps: List[str]
    are_gaps_relevant_for_industry: bool
    are_strengths_relevant_for_industry: bool
    deal_breaker_skill_missing: bool
    summary: str

class PorterFiveForces(BaseModel):    #Only used as a part of the larger Industry Analysis JSON
    bargaining_power_of_suppliers: str
    bargaining_power_of_buyers: str
    threat_of_new_entrants: str
    threat_of_substitutes: str
    rivalry_among_existing_competitors: str

class IndustryAnalysis(BaseModel):
    claimed_industry: str
    activity_based_industry: str
    is_coherent_with_claims: bool
    porter_five_forces_summary: PorterFiveForces
    is_aligned_with_thesis: bool
    summary: str

class ProductAnalysis(BaseModel):
    core_product_offering: str
    problem_solved: str
    value_proposition_qualitative: str
    value_proposition_quantitative: str
    direct_substitutes: List[str]
    summary: str

"""**Agent 4: Externalities Analyst**
*   **Primary Goal:** Identify external (PESTLE) risks and threats.
*   **Required Inputs:**
    *   `industry_analysis`: The `IndustryAnalysis` object from Agent 2.
    *   `product_analysis`: The `ProductAnalysis` object from Agent 3.
    *   `macroeconomic_data`: A summary of the current economic climate.

**Agent 5: Competition Analyst**
*   **Primary Goal:** Analyze the competitive landscape and the startup's differentiation.
*   **Required Inputs:**
    *   `product_analysis`: The `ProductAnalysis` object from Agent 3.
    *   `market_analysis_from_deck`: The market section from the initial Gemini extraction.

**Agent 6: Financial Viability Analyst**
*   **Primary Goal:** Assess the financial model and market size for viability.
*   **Required Inputs:**
    *   `pitch_deck_financials`: Tables and text related to financials from the deck.
    *   `product_analysis`: The `ProductAnalysis` object from Agent 3.
    *   `competition_analysis`: The `CompetitionAnalysis` object from Agent 5.

**Agent 7: VC Synergy Analyst**
*   **Primary Goal:** Evaluate the startup's fit within the VC's existing portfolio.
*   **Required Inputs:**
    *   `founder_analysis`: The `FounderAnalysis` object from Agent 1.
    *   `externalities_analysis`: The `ExternalitiesAnalysis` object from Agent 4.
    *   `vc_portfolio_data`: A structured description of the fund's current portfolio companies.
"""
class PestleSensitivities(BaseModel): # Only used in ExternalitiesAnalysis
    political: List[str]
    economic: List[str]
    social: List[str]
    technological: List[str]
    legal: List[str]
    environmental: List[str]

class ExternalitiesAnalysis(BaseModel):
    pestle_sensitivities: PestleSensitivities
    key_threats: List[str]
    imminent_threats_based_on_macro_climate: List[str]
    existential_threat_identified: bool
    summary: str

class CompetitionAnalysis(BaseModel):
    direct_competitors: List[str]
    best_alternative_solution: str
    competitive_advantage: str
    net_positive_for_decision_maker: bool
    switching_costs_analysis: str
    summary: str

# class FinancialAnalysis(BaseModel):
#     market_size_TAM_in_INR: float
#     required_market_share_for_3yr_amortization: float
#     is_required_market_share_rational: bool
#     unit_economics_summary: str
#     summary: str

# Beginning Financial Analysis
class MarketSize(BaseModel):
    tam: float = Field(description="Total Addressable Market value in USD.")
    source: str = Field(description="Where the TAM figure was found (e.g., 'Pitch Deck, slide 8', 'Analyst Estimate').")
    rationale: str = Field(description="Brief justification for the TAM figure used, especially if estimated.")

class UnitEconomics(BaseModel):
    revenue_model: str = Field(description="The primary revenue model (e.g., 'B2B SaaS - Per Seat', 'Transactional', 'Marketplace Take Rate').")
    price_per_unit: float = Field(description="Revenue per unit (e.g., per user per month, per transaction).")
    variable_cost_per_unit: float = Field(description="Variable cost (COGS) to deliver one unit.")
    contribution_margin_per_unit: float = Field(description="Calculated as (Price - Variable Cost).")
    customer_acquisition_cost_cac: float = Field(description="Estimated cost to acquire one new customer.")

class ThreeYearViability(BaseModel):
    annual_fixed_costs: float = Field(description="Estimated annual fixed costs (burn rate) from salaries, rent, etc.")
    one_time_development_costs: float = Field(description="Estimated one-time R&D or development costs to be amortized.")
    total_costs_to_amortize: float = Field(description="Calculated as (Annual Fixed Costs * 3) + One-Time Development Costs.")
    required_units_to_sell_in_3_years: int = Field(description="Calculated as Total Costs to Amortize / Contribution Margin per Unit.")
    required_annual_revenue_at_year_3: float = Field(description="The annualized revenue needed to meet the 3-year goal.")
    required_market_share: float = Field(description="The required percentage of the TAM. Calculated as Required Annual Revenue / TAM.")

class FinancialAnalysis(BaseModel):
    market_size: MarketSize
    unit_economics: UnitEconomics
    three_year_viability_check: ThreeYearViability
    is_rational_assessment: str = Field(description="A final judgment on whether the required market share is rational and achievable, with a brief explanation.")
    summary: str = Field(description="A high-level executive summary of the financial viability.")
    
# Beginning Synergy Analysis
class SynergyDetail(BaseModel):     ## Only used in SynergyAnalysis
    portfolio_co: str
    synergy: str

class SynergyAnalysis(BaseModel):
    potential_synergies: List[SynergyDetail]
    solves_identified_skill_gap: bool
    solves_identified_external_threat: bool
    summary: str

"""**Combined Agent - Define the Shared State**
"""
# --- Define the Shared State ---
class AnalysisSharedState(BaseModel):
    """A single object to hold all analysis results."""
    # source_pitch_deck_name: str
    company_analysed: str
    source_pitch_deck_urls: List[str]
    vc_portfolio_information: Optional[List[str]] = None
    investing_thesis: Optional[str] = None
    founder_linkedin_urls: List[str] = []

    # Outputs from parallel agents
    founder_analysis: Optional[FounderAnalysis] = None
    industry_analysis: Optional[IndustryAnalysis] = None
    product_analysis: Optional[ProductAnalysis] = None

    # Outputs from sequential agents
    externalities_analysis: Optional[ExternalitiesAnalysis] = None
    competition_analysis: Optional[CompetitionAnalysis] = None
    financial_analysis: Optional[FinancialAnalysis] = None
    synergy_analysis: Optional[SynergyAnalysis] = None

"""### Agent Functions (Trial 1)

"""

# Generic Type for Pydantic models
T = TypeVar('T', bound=BaseModel)

async def invoke_gemini_agent(
    agent_name: str,
    prompt: str,
    output_model: Type[T],
    agent_system_instruction: Optional[str] = ""
) -> T:
    """
    A robust helper to invoke the Gemini model and parse the output into a Pydantic model.
    """
    print(f"Agent {agent_name}: Starting analysis...")

    global_system_instructions = """
    You are a world-class specialist analyst working as part of a multi-agent system to evaluate a startup for a top-tier Venture Capital firm. Your analysis must be objective, critical, and based **exclusively** on the information provided in the prompt.

    **Universal Rules:**
    1.  **No Hallucination:** Do not invent data, figures, or founder experiences. If information is missing from the provided text, you must state that it is not available rather than making an assumption.
    2.  **Stick to Your Role:** Only perform the analysis requested in your specific instructions. Do not analyze aspects assigned to other agents.
    3.  **JSON Output Only:** Your final output MUST be a single, valid JSON object. Do not include any explanatory text, markdown formatting (like ` ```json `), or apologies before or after the JSON object.
    ***
    """

    try:
        response : GenerateContentResponse = await client.aio.models.generate_content(
            model=MODEL_ID,
            # contents= research_pdf + [prompt],
            contents = prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature = 0.3,
                max_output_tokens=65535,
                response_schema = output_model.model_json_schema(),
                system_instruction=[types.Part.from_text(text=global_system_instructions + agent_system_instruction)]
            )
        )
        stopping_phrase = json.loads(response.model_dump_json())["candidates"][0]["finish_reason"]
        if stopping_phrase != 'STOP':
          print(f"Current stopping phrase {stopping_phrase}")

        # Clean and parse the JSON response
        json_text = response.text.strip().replace("```json", "").replace("```", "")
        json_data = json.loads(json_text)

        # Validate the data against the Pydantic model
        validated_output = output_model.model_validate(json_data)

        print(f"Agent {agent_name}: Analysis complete and validated.")
        return validated_output

    except json.JSONDecodeError as e:
        print(f"ERROR in Agent {agent_name}: Failed to decode JSON. Response text: {response.text}")
        raise ValueError(f"Agent {agent_name} received invalid JSON from API.") from e
    except ValidationError as e:
        print(f"ERROR in Agent {agent_name}: Pydantic validation failed. Errors: {e.errors()}")
        raise ValueError(f"Agent {agent_name} API response did not match schema.") from e
    except Exception as e:
        print(f"ERROR in Agent {agent_name}: An unexpected error occurred: {e}")
        raise


# --- Full-Fledged Agent Functions ---

async def run_agent_1_founders(source_pitch_deck_urls: List[str]) -> FounderAnalysis:
    prompt = f"""
    You are a Venture Capital Analyst specializing in team dynamics.
    Analyze the 'team' section of the provided pitch deck text.
    Based *only* on the text below, answer the following questions:
    a) How many founders are there?
    b) Are the founders' skills complementary? What is the degree of overlap?
    c) What crucial business skills (e.g., technical, sales, operations) are covered?
    d) What crucial skills appear to be missing?
    e) Are there any obvious deal-breakers in the team's composition?


    Return your analysis as a JSON object matching the FounderAnalysis schema.
    """
    prompt = fetch_input_pdf(prompt_pdf_input= source_pitch_deck_urls) + [prompt]

    agent_system_instruction = """
      #### **Agent 1: Founder Analyst**
      **Your Persona:** You are a **Founder Analyst**. You specialize in evaluating team dynamics, skill sets, and founder-market fit.
      **Your Framework:** Your analysis focuses on the complementarity of skills, identifying crucial strengths, and flagging any deal-breaking gaps in the founding team.
      **Your Central Question:** "Is this the right team to solve this problem in this market?"
    """

    return await invoke_gemini_agent("1 (Founders)", prompt, FounderAnalysis, agent_system_instruction)

async def run_agent_2_industry(source_pitch_deck_urls: List[str], thesis: str) -> IndustryAnalysis:
    prompt = f"""
    You are a Market Research Analyst. Your task is to define the startup's industry.
    1.  Identify the industry the company *claims* to be in.
    2.  Based on the company's actual activities and product, define a more precise, activity-based industry.
    3.  Assess if the activity-based definition is coherent with their claims.
    4.  Provide a brief Porter's Five Forces analysis for the activity-based industry.
    5.  Determine if this industry aligns with the provided VC Investment Thesis.

    VC Investment Thesis: "{thesis}"

    Pitch Deck: Attached to prompt

    Return your analysis as a JSON object matching the IndustryAnalysis schema.
    """
    prompt = fetch_input_pdf(prompt_pdf_input= source_pitch_deck_urls) + [prompt]
    agent_system_instruction = """
      #### **Agent 2: Industry Definer**
      **Your Persona:** You are a **Market Research Analyst**. You excel at seeing beyond marketing claims to define a company's true operational industry.
      **Your Framework:** You will use the 'activity-based' principle to define the industry and then apply Porter's Five Forces framework to assess its attractiveness.
      **Your Central Question:** "Is this an attractive industry for our fund to be in, and does it align with our thesis?"
    """

    return await invoke_gemini_agent("2 (Industry)", prompt, IndustryAnalysis, agent_system_instruction)

async def run_agent_3_product(source_pitch_deck_urls: List[str]) -> ProductAnalysis:
    prompt = f"""
    You are a Product Manager. Analyze the product described in the pitch deck.
    Focus on the core value proposition. Answer these questions:
    a) What is the exact product or service being offered?
    b) What is the primary problem this product solves for the customer?
    c) Describe the value addition in qualitative terms (e.g., "saves time", "improves accuracy").
    d) Describe the value addition in quantitative terms if mentioned (e.g., "reduces costs by 20%").
    e) What are the direct substitutes or alternatives the customer is using today?

    Pitch Deck: Attached to prompt

    Return your analysis as a JSON object matching the ProductAnalysis schema.
    """
    prompt = fetch_input_pdf(prompt_pdf_input= source_pitch_deck_urls) + [prompt]
    agent_system_instruction = """
      #### **Agent 3: Product Analyst**
      **Your Persona:** You are a **Senior Product Manager**. You are an expert at deconstructing a product to its core value proposition for the customer.
      **Your Framework:** You will analyze the problem-solution fit, the qualitative and quantitative value propositions, and the existing alternatives the customer uses.
      **Your Central Question:** "Does this product solve a real, painful problem in a way that is significantly better than the alternatives?"
    """

    return await invoke_gemini_agent("3 (Product)", prompt, ProductAnalysis, agent_system_instruction)

async def run_agent_4_externalities(state: AnalysisSharedState) -> ExternalitiesAnalysis:
    prompt = f"""
    You are a Risk Analyst. Your task is to perform a PESTLE analysis.
    Given the defined industry and product, identify potential external risks.
    - Industry: {state.industry_analysis.activity_based_industry}
    - Product: {state.product_analysis.core_product_offering}
    - Current Macro Climate: Assume a climate of high interest rates and geopolitical instability.

    1.  Identify Political, Economic, Social, Technological, Legal, and Environmental sensitivities.
    2.  List the top 3-5 key threats to the business.
    3.  Based on the macro climate, which of these are imminent threats?
    4.  Are any of these threats existential?

    Return your analysis as a JSON object matching the ExternalitiesAnalysis schema.
    """
    agent_system_instruction = """
      #### **Agent 4: Externalities Analyst**
      **Your Persona:** You are a **Geopolitical and Economic Risk Analyst**. You identify external threats that a startup's founders may have overlooked.
      **Your Framework:** You will conduct a PESTLE (Political, Economic, Social, Technological, Legal, Environmental) analysis to identify key sensitivities and existential threats.
      **Your Central Question:** "What external forces could kill this company, regardless of how well it executes?"
    """

    return await invoke_gemini_agent("4 (Externalities)", prompt, ExternalitiesAnalysis, agent_system_instruction)

async def run_agent_5_competition(state: AnalysisSharedState) -> CompetitionAnalysis:
    prompt = f"""
    You are a Competitive Strategy Analyst.
    Analyze the competitive landscape based on the provided context.
    - Product: {state.product_analysis.core_product_offering}
    - Direct Substitutes Identified: {state.product_analysis.direct_substitutes}
    - Pitch Deck: Attached to prompt

    1.  Who are the main direct competitors?
    2.  What is the best alternative solution for the customer (could be a competitor or a manual process)?
    3.  What is this company's primary competitive advantage?
    4.  From the decision-maker's perspective, is there a clear net positive in switching?
    5.  Briefly analyze the switching costs (time, money, effort).

    Return your analysis as a JSON object matching the CompetitionAnalysis schema.
    """
    prompt = fetch_input_pdf(prompt_pdf_input= state.source_pitch_deck_urls) + [prompt]
    agent_system_instruction = """
      #### **Agent 5: Competition Analyst**
      **Your Persona:** You are a **Competitive Strategy Analyst**. You understand that competition is not just about direct rivals but also about alternative solutions.
      **Your Framework:** You will analyze the competitive landscape, determine the best alternative solution from the customer's perspective, and assess the current product's defensible competitive advantage and switching costs.
      **Your Central Question:** "Why will this company win against all other ways the customer can solve this problem?"
    """

    return await invoke_gemini_agent("5 (Competition)", prompt, CompetitionAnalysis, agent_system_instruction)

async def run_agent_6_financials(state: AnalysisSharedState) -> FinancialAnalysis:
    prompt = f"""
    You are a Financial Analyst. Assess the financial viability based on the deck.
    - Market Info: {state.industry_analysis.summary}
    - Pitch Deck Text: Attached to prompt

    1.  Extract the Total Addressable Market (TAM) value. If not present, estimate it based on the industry.
    2.  Based on the financials (costs, pricing), estimate the percentage of the TAM the company needs to capture in 3 years to amortize its costs and be profitable.
    3.  Is this required market share percentage rational and achievable?
    4.  Summarize the unit economics (e.g., LTV/CAC, margins) if data is available.
    
    **Your Step-by-Step Task:**
    You must perform the following analysis in order.
    
    **Step 1: Determine Market Size (TAM).**
    - Find the Total Addressable Market (TAM) figure in the pitch deck.
    - Note the source (e.g., which slide).
    - If not present, estimate the TAM based on the industry and provide a clear rationale for your estimation.
    
    **Step 2: Deconstruct the Revenue Model.**
    - Identify the core revenue model (e.g., SaaS, transactional).
    - Find the price per unit (e.g., price per seat per month).
    
    **Step 3: Calculate Unit Economics.**
    - Find or estimate the variable costs (COGS) to deliver one unit.
    - Calculate the Contribution Margin per Unit (Price - COGS).
    - Find or estimate the Customer Acquisition Cost (CAC).
    
    **Step 4: Analyze the Cost Structure.**
    - Find or estimate the annual fixed costs (burn rate), which includes salaries, rent, and G&A.
    - Find or estimate any significant one-time development costs mentioned.
    
    **Step 5: Perform the 3-Year Viability Check.**
    - This is a critical sanity check to see if the business can become self-sustaining.
    - First, calculate the `Total Costs to Amortize` over 3 years using the formula: `(Annual Fixed Costs * 3) + One-Time Development Costs`.
    - Second, calculate the `Required Units to Sell in 3 Years` using the formula: `Total Costs to Amortize / Contribution Margin per Unit`.
    - Third, calculate the `Required Annual Revenue at Year 3` based on the number of units that need to be sold.
    - Finally, calculate the `Required Market Share` percentage using the formula: `Required Annual Revenue / TAM`.
    
    **Step 6: Make a Rationality Assessment.**
    - Based on the final `Required Market Share` percentage, provide a concluding judgment. Is capturing this much of the market in 3 years plausible, ambitious, or unrealistic for a startup in this industry? Justify your answer.
    
    Now, perform your step-by-step analysis and return your analysis as a JSON object matching the FinancialAnalysis schema.
    """

    prompt = fetch_input_pdf(prompt_pdf_input= state.source_pitch_deck_urls) + [prompt]
    agent_system_instruction = """
      #### **Agent 6: Financial Analyst**
      **Your Persona:** You are a pragmatic **Financial Analyst** for a top-tier VC firm. You are deeply skeptical of projections 
        and believe in first-principles analysis. Your primary job is to sanity-check the business model's viability. 
        You are skeptical of vanity metrics and focus on the fundamental viability of a business model.
      **Your Framework:** You will analyze the unit economics, market size (TAM), and the rationality of the market share required for profitability.
      **Your Central Question:** "Can this company realistically make enough money to become a venture-scale business?"

      **Global Rules:**
      1.  **No Hallucination:** If a specific number (e.g., CAC, fixed costs) is not in the text, you must state that you are estimating it based on industry standards for a company of this type and stage. **Always show your work.**
      2.  **JSON Output Only:** Your final output MUST be a single, valid JSON object matching the `FinancialAnalysis` schema.
    """
    return await invoke_gemini_agent("6 (Financials)", prompt, FinancialAnalysis, agent_system_instruction)

async def run_agent_7_synergies(state: AnalysisSharedState, portfolio_data: List[str]) -> SynergyAnalysis:
    prompt = f"""
    You are a Venture Capital Partner. Your goal is to find synergies with our existing portfolio.
    - Identified Founder Skill Gaps: {state.founder_analysis.identified_gaps}
    - Identified External Threats: {state.externalities_analysis.key_threats}
    - Our VC Portfolio: {'Not Available' if portfolio_data == [] else portfolio_data}

    1.  What specific, actionable synergies exist between this startup and our portfolio companies? (e.g., "Cross-sell Product A to Portfolio Co. B's customer base").
    2.  Can a connection to our portfolio companies help close the identified skill gaps?
    3.  Can our portfolio help mitigate any of the identified external threats?

    Return your analysis as a JSON object matching the SynergyAnalysis schema.
    """
    agent_system_instruction = """
      #### **Agent 7: VC Synergy Analyst**
      **Your Persona:** You are a **VC Portfolio Strategist**. You have deep knowledge of our fund's existing portfolio companies and strategic goals.
      **Your Framework:** Your analysis must be actionable. You will identify concrete, cross-portfolio opportunities for sales, technology, or talent sharing.
      **Your Central Question:** "Does this startup make our entire portfolio stronger, and can we uniquely help them succeed?"
    """

    return await invoke_gemini_agent("7 (Synergies)", prompt, SynergyAnalysis, agent_system_instruction)

"""### Overall State"""

# --- 4. Create the Orchestrator ---
async def run_investment_analysis(pitch_deck_urls: List[str], company_name: str, api_key = "AIzaSyClrXb0hufBK6_aZ_PbAmp48Pvszeft2DE"):
    """
    Orchestrates the multi-agent analysis of a startup pitch deck.
    """
    # Initialize the shared state
    API_KEY = api_key

    state = AnalysisSharedState(company_analysed = company_name,
                                source_pitch_deck_urls=pitch_deck_urls)
    print(f"Pitch Deck URLs: {pitch_deck_urls}")
    print(f"State Currently: {state}")

    print("--- Starting Parallel Analysis Phase (Agents 1, 2, 3) ---")

    # Run agents 1, 2, and 3 concurrently
    parallel_tasks = [
        run_agent_1_founders(state.source_pitch_deck_urls),
        run_agent_2_industry(state.source_pitch_deck_urls, "No specific thesis, all companies are a good fit"),
        run_agent_3_product(state.source_pitch_deck_urls),
    ]

    results = await asyncio.gather(*parallel_tasks)

    # Update the shared state with the results
    state.founder_analysis = results[0]
    state.industry_analysis = results[1]
    state.product_analysis = results[2]

    print("\n--- Parallel Phase Complete. Starting Sequential Analysis ---")

    # Run agents 4, 5, 6, 7 sequentially
    state.externalities_analysis = await run_agent_4_externalities(state)
    print("Agent 4: Complete.")
    state.competition_analysis = await run_agent_5_competition(state)
    print("Agent 5: Complete.")
    state.financial_analysis = await run_agent_6_financials(state)
    print("Agent 6: Complete.")
    state.synergy_analysis = await run_agent_7_synergies(state, portfolio_data= state.vc_portfolio_information)
    print("Agent 7: Complete.")

    print("\n--- Full Analysis Complete ---")
    return state


def fetch_input_pdf(prompt_pdf_input):
  research_pdf = []
  for each_pdf in prompt_pdf_input:
    research_pdf.append(
        client.files.upload(
      file=each_pdf,
      config=dict(mime_type='application/pdf')
      )
    )

  return research_pdf


if __name__ == "__main__":
    # Example of running the orchestrator
    # Use await instead of asyncio.run() in Colab
    # final_state = await run_investment_analysis(pitch_deck_urls = get_files_with_classfication(grouped_files_df, ['pitchdeck']))

    # final_state = await run_investment_analysis(pitch_deck_urls = get_files_with_classification(grouped_files_df))
    company_name = "Fabpad"
    folder_path = 'C:\\Users\\Home\\Desktop\\GCP Hackathon\\tuning-machines-ai\\company-analysis-data-tuning-machines'

    final_state = asyncio.run(run_investment_analysis(pitch_deck_urls = fetch_all_required_files(folder_path, company_name),
                                                      company_name = company_name))
    print(final_state.model_dump_json())
    # print(final_state.model_dump_json(indent=2))

