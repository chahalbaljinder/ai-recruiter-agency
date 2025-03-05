from typing import Dict, Any
import json
import google.generativeai as genai
import os


class BaseAgent:
    def __init__(self, name: str, instructions: str):
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.name = name
        self.instructions = instructions
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash")

    async def run(self, messages: list) -> Dict[str, Any]:
        """Default run method to be overridden by child classes"""
        raise NotImplementedError("Subclasses must implement run()")

    def _query_gemini(self, prompt: str) -> str:
        """Query Gemini model with the given prompt"""
        try:
            response = self.model.generate_content(
                self.instructions + prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=2000,
                ),
            )
            return response.text
        except Exception as e:
            print(f"Error querying Gemini: {str(e)}")
            raise

    def _parse_json_safely(self, text: str) -> Dict[str, Any]:
        """Safely parse JSON from text, handling potential errors"""
        try:
            # Try to find JSON-like content between curly braces
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                json_str = text[start : end + 1]
                return json.loads(json_str)
            return {"error": "No JSON content found"}
        except json.JSONDecodeError:
            return {"error": "Invalid JSON content"}
