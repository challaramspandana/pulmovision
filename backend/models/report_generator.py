import os
import google.generativeai as genai

class RadiologyReportGenerator:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel("gemini-1.5-flash")
        else:
            self.model = None

    def generate_report(self, patient_info, current_findings, progression_status, retrieved_cases):
        """
        Generates a structured medical report combining AI predictions,
        progression analysis, and retrieved clinical cases.
        """
        prompt = f"""
        You are an expert radiologist assisting a clinician. Generate a structured radiology report based on the following patient data:

        PATIENT DETAILS:
        - Age: {patient_info.get('age', 'N/A')}
        - Gender: {patient_info.get('gender', 'N/A')}
        - Clinical Symptoms: {patient_info.get('symptoms', 'N/A')}

        AI CHEST X-RAY FINDINGS:
        - Primary Predicted Pathology: {current_findings.get('predicted_disease')}
        - AI Confidence Score: {current_findings.get('confidence')}%

        DISEASE PROGRESSION ANALYSIS:
        - Status compared to prior scan: {progression_status}

        SIMILAR HISTORICAL CASES (RAG RETRIEVAL):
        {self._format_cases(retrieved_cases)}

        Please generate a professional radiology report with the following sections:
        1. EXAMINATION FINDINGS
        2. COMPARISON & PROGRESSION ANALYSIS
        3. CLINICAL IMPRESSION
        4. RECOMMENDED NEXT STEPS
        """

        if self.model:
            try:
                response = self.model.generate_content(prompt)
                return response.text
            except Exception as e:
                return self._fallback_report(patient_info, current_findings, progression_status, retrieved_cases)
        else:
            return self._fallback_report(patient_info, current_findings, progression_status, retrieved_cases)

    def _format_cases(self, cases):
        formatted = ""
        for idx, c in enumerate(cases, 1):
            formatted += f"Case {idx}: {c['summary']} (Diagnosis: {c['metadata'].get('diagnosis')})\n"
        return formatted

    def _fallback_report(self, patient_info, current_findings, progression_status, retrieved_cases):
        """Structured template fallback if no API key is provided."""
        return f"""
======================================================================
                     PULMOVISION RADIOLOGY REPORT
======================================================================
PATIENT AGE/GENDER: {patient_info.get('age', 'N/A')} / {patient_info.get('gender', 'N/A')}
SYMPTOMS: {patient_info.get('symptoms', 'N/A')}

1. EXAMINATION FINDINGS:
   - Primary AI Observation: {current_findings.get('predicted_disease')} 
   - Detection Confidence: {current_findings.get('confidence')}%

2. COMPARISON & PROGRESSION ANALYSIS:
   - Status vs. Previous Radiograph: {progression_status}

3. SIMILAR CASE CONTEXT:
   - Retrieved past cases indicate comparable radiological patterns of {current_findings.get('predicted_disease')}.

4. CLINICAL IMPRESSION:
   - Findings are consistent with {current_findings.get('predicted_disease')}. The overall progression status is marked as {progression_status}.

5. RECOMMENDATION:
   - Clinical correlation with patient lab results and symptoms is advised.
======================================================================
        """

if __name__ == "__main__":
    generator = RadiologyReportGenerator()
    
    sample_patient = {"age": 45, "gender": "Male", "symptoms": "Cough, fever, short breath"}
    sample_findings = {"predicted_disease": "Pneumonia", "confidence": 88.5}
    sample_progression = "Worsening by 8.2% compared to prior scan"
    sample_retrieved = [
        {"summary": "Patient showed lower lobe opacity treated with antibiotics.", "metadata": {"diagnosis": "Pneumonia"}}
    ]

    report = generator.generate_report(sample_patient, sample_findings, sample_progression, sample_retrieved)
    print(report)