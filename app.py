import streamlit as st
import os
import io
from duckduckgo_search import DDGS
from google import genai
from pydantic import BaseModel, Field
from typing import Literal
from pypdf import PdfReader

# --- Configuration ---
API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_KEY_HERE")
client = genai.Client(api_key=API_KEY)

class EvaluationResult(BaseModel):
    status: Literal[
        "Application Cleared: Ready to Proceed",
        "Criteria Unmet: Ineligible Based on Current Details",
        "Action Required: Missing or Incomplete Documentation",
        "Escalated: Manual Reviewer Verification Needed"
    ]
    established_points: list[str] = Field(description="What was proven")
    unestablished_points: list[str] = Field(description="What is missing")
    issues: list[str] = Field(description="Discrepancies or unreadable data")
    can_proceed: bool
    next_action: str

st.set_page_config(page_title="ClearGov Fast-Track", layout="wide")
st.title("ClearGov: Smart Evidence Evaluator")

role_query = st.text_input("What public service or job are you applying for?")

if st.button("Search Requirements"):
    with st.spinner("Searching the web for official criteria..."):
        results = DDGS().text(f"mandatory eligibility requirements and documents for {role_query}", max_results=3)
        search_text = "\n".join([r["body"] for r in results])
        
        prompt = f"Extract a bulleted list of mandatory documents and criteria for '{role_query}' based on this web search: {search_text}"
        reqs = client.models.generate_content(model="gemini-3.6-flash", contents=prompt).text
        st.session_state["reqs"] = reqs

if "reqs" in st.session_state:
    st.info("**Found Requirements:**\n" + st.session_state["reqs"])
    
    uploaded_files = st.file_uploader("Upload your PDF documents", type=["pdf"], accept_multiple_files=True)
    
    if uploaded_files and st.button("Evaluate Application"):
        with st.spinner("AI is reasoning over your evidence..."):
            evidence_text = ""
            for file in uploaded_files:
                try:
                    file_bytes = file.getvalue()
                    if not file_bytes:
                        st.error(f"The file {file.name} is completely empty (0 bytes)!")
                        continue
                        
                    reader = PdfReader(io.BytesIO(file_bytes))
                    evidence_text += f"\n--- Document: {file.name} ---\n"
                    evidence_text += "".join([p.extract_text() or "" for p in reader.pages])
                    
                except Exception as e:
                    st.error(f"Could not read '{file.name}'. It might not be a valid PDF. Error: {e}")
                    continue
            
            if not evidence_text.strip():
                st.warning("No readable text was found in your uploaded documents. Please upload a valid PDF with text.")
                st.stop()
            
            system_instruction = """
            You are a strict application evaluator. Compare the submitted evidence against the requirements. 
            You must output your decision using the structured JSON format provided.
            """
            eval_prompt = f"Requirements:\n{st.session_state['reqs']}\n\nSubmitted Evidence:\n{evidence_text}"
            
            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=eval_prompt,
                config={
                    "system_instruction": system_instruction,
                    "response_mime_type": "application/json",
                    "response_schema": EvaluationResult,
                },
            )
            
            result = EvaluationResult.model_validate_json(response.text)
            
            st.divider()
            st.header(f"Status: {result.status}")
            st.subheader(f"Can Proceed: {'✅ Yes' if result.can_proceed else '❌ No'}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.success("**Successfully Established**")
                for p in result.established_points: st.write(f"- {p}")
            with col2:
                st.warning("**Missing or Problematic**")
                for p in result.unestablished_points + result.issues: st.write(f"- {p}")
                
            st.info(f"**Next Action:** {result.next_action}")
