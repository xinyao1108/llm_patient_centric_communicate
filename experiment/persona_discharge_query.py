#!/usr/bin/env python3
"""
Query ChatGPT with different persona combinations for discharge summary comprehension.
Tests multiple questions across different personas.
"""

import os
import json
import itertools
from openai import OpenAI
from anthropic import Anthropic
from typing import List, Dict, Any, Optional
from datetime import datetime


class PersonaDischargeQueryEngine:
    def __init__(self, openai_api_key: str = None, anthropic_api_key: str = None):
        """Initialize the engine with API keys for OpenAI and/or Anthropic."""
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")

        if not self.openai_api_key and not self.anthropic_api_key:
            raise ValueError(
                "At least one API key required. Set OPENAI_API_KEY and/or ANTHROPIC_API_KEY."
            )

        self.openai_client = OpenAI(api_key=self.openai_api_key) if self.openai_api_key else None
        self.anthropic_client = Anthropic(api_key=self.anthropic_api_key) if self.anthropic_api_key else None

        # Define all questions.
        #
        # Some questions reuse the same Q-number across discharge summaries but
        # the answer options vary by case (e.g., Q6 in DS2 asks about
        # "when to call the doctor" rather than the diagnosis name). To handle
        # this, each entry below is a dict mapping DS id -> question text;
        # the "default" key is used for any DS not explicitly overridden.
        self.questions = {
            "Q1": {
                "default": "Please rate your understanding level of this discharge instruction.\nA.Very clear\nB. Somewhat clear\nC. Not clear at all",
            },
            "Q2": {
                "default": "Do you know the name of all your medications? \nA.Yes\nB. I don't know\nC. Not provided",
            },
            "Q3": {
                "default": "Do you know your diagnosis? \nA.Yes\nB. I don't know\nC. Not provided",
            },
            "Q4": {
                "default": "Do you know the common side effects of all your medications? \nA.Yes\nB. I don't know\nC. Not provided",
            },
            "Q5": {
                "default": "Are there other prescriptions given besides the medication? \nA. Yes\nB. I don't know\nC. Not provided",
            },
            "Q6": {
                # DS1, DS3, DS4: diagnosis identification
                "default": "Do you know what kind of condition you have mentioned in the discharge instructions?\nA. Stomach disease\nB. Ulcer in the duodenum\nC. Wearing Tegaderm\nD. Keep splint\nE. I don't know",
                # DS2: "when to call the doctor" -- different stem and options
                "DS2": "What are some situations that you have to call Dr. ___?\nA. Fever > 101\nB. Chest Pain\nC. Stomach Pain / diarrhea\nD. Shorten of breath\nE. All of the above\nF. I don't know",
            },
            "Q7": {
                # DS1 and DS4: 4-option treatment identification
                "default": "Do you know what kind of treatment you need to follow based on the discharge instructions?\nA. Nothing\nB. Take a new medication\nC. See a doctor again\nD. I don't know",
                # DS2: 6-option "activities prohibited" multi-select
                "DS2": "Do you know what kind of treatment you need to follow based on the discharge instructions?\nA. Nothing\nB. Swimming in 4 weeks\nC. Consuming narcotics without stool softeners\nD. Driving while under influence of narcotics\nE. Consuming Alcohol\nF. I don't know",
                # DS3: 5-option orthopedic-specific variant
                "DS3": "Do you know what kind of treatment you need to follow based on the discharge instructions?\nA. Take a new medication\nB. See a doctor again\nC. Non-weight-bearing left lower extremity\nD. Nothing\nE. I don't know",
            },
            "Q8": {
                "default": "Are there any activities or foods you need to avoid?\nA. Avoid fruit\nB. Avoid strenuous exercise\nC. Others\nD. I don't know",
            },
            "Q9": {
                "default": "Is there anything about your discharge instructions that is unclear or worrying you?\nA. Medication schedule\nB. Follow-up appointments\nC. Symptoms to watch for\nD. Dietary restrictions\nE. Activity limitations\nF. Other, please specify\nG. No, it's very clear",
            },
            "Q10": {
                "default": "Please rate the difficulty in understanding this discharge instruction.\nA. Extremely easy\nB. Somewhat easy\nC. Neither easy nor difficult\nD. Somewhat difficult\nE. Extremely difficult",
            },
        }

    def get_question_text(self, q_id: str, ds_id: str) -> str:
        """Return the question text for (q_id, ds_id), falling back to default."""
        q_entry = self.questions[q_id]
        return q_entry.get(ds_id, q_entry["default"])

        # Define discharge summaries
        self.discharge_summaries = {
            "DS1": "You were admitted and found to have an ulcer in the duodenum. To help this heal, we are prescribing new medications (pantoprazole). Please be sure to take this until you are seen in follow-up.",

            "DS2": """Call Dr. xxx if experience:
-Take stool softeners with narcotics
-Fever > 101 or chills
-Increased shortness of breath or cough
-Chest pain
-You may shower. No swimming for 4 weeks
-No driving while taking narcotics.""",

            "DS3": """Keep splint/dressing on until follow-up
Keep splint clean & dry at all times
Follow up with Dr. xxx in xxx days
Wean off of narcotics
Take aspirin for 2 weeks
Physical Therapy:
NWB left lower extremity
Leave splint on until follow-up""",

            "DS4": "Ok to shower today but wear tegaderm dressing over the drain site. No heavy lifting. Return to ED for anything that concerns you."
        }

    def build_persona_prompt(
        self,
        age: str,
        gender: str,
        education: str,
        ethnicity: str,
        doctor_visit: str,
        er_visit_frequency: str,
        discharge_summary: str,
        question: str,
        model: str = "o1-mini",
        reasoning_instruction: str = "Explain with your reasoning, then provide the letter."
    ) -> str:
        """Build the complete prompt with persona and discharge summary."""

        prompt = f"""You are a {age} {gender} with {education} education level, you from {ethnicity} race, you visit doctor {doctor_visit}, and visit emergency room {er_visit_frequency}. {reasoning_instruction}

You will be reading a Discharge Summary written by a clinician for a patient. After reading the Discharge Summary, I will ask you a multiple-choice question. Your job is to think as someone with your background and choose the correct answer by selecting the letter of the answer (e.g., A, B, C, etc.).

Here is the Discharge Summary:

{discharge_summary}

Now, answer the following question:

{question}"""

        return prompt

    def query(
        self,
        prompt: str,
        model: str = "o1-mini",
        temperature: float = 1,
        max_completion_tokens: int = 500,
        provider: str = "openai"
    ) -> Dict[str, Any]:
        """Send a single query to the specified provider."""
        if provider == "anthropic":
            return self._query_anthropic(prompt, model, temperature, max_completion_tokens)
        return self._query_openai(prompt, model, temperature, max_completion_tokens)

    def _query_openai(
        self,
        prompt: str,
        model: str,
        temperature: float,
        max_completion_tokens: int
    ) -> Dict[str, Any]:
        """Send a single query to OpenAI."""
        if not self.openai_client:
            return {"success": False, "error": "OpenAI API key not configured"}
        try:
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are responding as the persona described in the prompt."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_completion_tokens=max_completion_tokens
            )

            return {
                "success": True,
                "response": response.choices[0].message.content,
                "model": model,
                "_tokens": response.usage.total_tokens,
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def _query_anthropic(
        self,
        prompt: str,
        model: str,
        temperature: float,
        max_completion_tokens: int
    ) -> Dict[str, Any]:
        """Send a single query to Anthropic."""
        if not self.anthropic_client:
            return {"success": False, "error": "Anthropic API key not configured"}
        try:
            response = self.anthropic_client.messages.create(
                model=model,
                system="You are responding as the persona described in the prompt.",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_completion_tokens
            )

            total_tokens = response.usage.input_tokens + response.usage.output_tokens
            return {
                "success": True,
                "response": response.content[0].text,
                "model": model,
                "_tokens": total_tokens,
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def run_full_experiment(
        self,
        persona_variations: Dict[str, List[str]],
        discharge_summary_ids: List[str] = None,
        question_ids: List[str] = None,
        model: str = "o1-mini",
        temperature: float = 1,
        max_personas: Optional[int] = None,
        enable_reasoning: bool = False,
        provider: str = "openai"
    ) -> List[Dict[str, Any]]:
        """
        Run the full experiment: all persona combinations x all DS x all questions.

        Args:
            persona_variations: Dict with persona attributes and their possible values
            discharge_summary_ids: List of DS IDs to test (default: all)
            question_ids: List of question IDs to test (default: all)
            model: ChatGPT model to use
            temperature: Sampling temperature
            max_personas: Limit number of persona combinations (for testing)
            enable_reasoning: Whether to ask model to provide reasoning (default: False)

        Returns:
            List of all results
        """
        # Default to all DS and questions
        if discharge_summary_ids is None:
            discharge_summary_ids = list(self.discharge_summaries.keys())
        if question_ids is None:
            question_ids = list(self.questions.keys())

        # Generate all persona combinations
        required_keys = ['age', 'gender', 'education', 'ethnicity', 'doctor_visit', 'er_visit_frequency']
        missing_keys = [k for k in required_keys if k not in persona_variations]
        if missing_keys:
            raise ValueError(f"Missing required persona keys: {missing_keys}")

        keys = required_keys
        values = [persona_variations[k] for k in keys]
        all_persona_combos = list(itertools.product(*values))

        # Limit personas if specified
        if max_personas and len(all_persona_combos) > max_personas:
            print(f"Limiting to first {max_personas} of {len(all_persona_combos)} persona combinations")
            all_persona_combos = all_persona_combos[:max_personas]

        total_queries = len(all_persona_combos) * len(discharge_summary_ids) * len(question_ids)
        print(f"="*80)
        print(f"EXPERIMENT CONFIGURATION")
        print(f"="*80)
        print(f"Personas: {len(all_persona_combos)}")
        print(f"Discharge Summaries: {len(discharge_summary_ids)} ({', '.join(discharge_summary_ids)})")
        print(f"Questions: {len(question_ids)} ({', '.join(question_ids)})")
        print(f"Total queries: {total_queries}")
        print(f"Model: {model}")
        print(f"="*80)

        all_results = []
        query_count = 0
        total_tokens_used = 0  # Track tokens for summary

        for persona_combo in all_persona_combos:
            persona = dict(zip(keys, persona_combo))

            for ds_id in discharge_summary_ids:
                discharge_summary = self.discharge_summaries[ds_id]

                for q_id in question_ids:
                    question = self.get_question_text(q_id, ds_id)
                    query_count += 1

                    # Determine reasoning instruction based on enable_reasoning parameter
                    if enable_reasoning:
                        reasoning_instruction = "Explain with your reasoning, then provide the letter."
                    else:
                        reasoning_instruction = "Answer with only letter"

                    # Build prompt
                    prompt = self.build_persona_prompt(
                        age=persona['age'],
                        gender=persona['gender'],
                        education=persona['education'],
                        ethnicity=persona['ethnicity'],
                        doctor_visit=persona['doctor_visit'],
                        er_visit_frequency=persona['er_visit_frequency'],
                        discharge_summary=discharge_summary,
                        question=question,
                        model=model,
                        reasoning_instruction=reasoning_instruction
                    )

                    # Query
                    print(f"\n[{query_count}/{total_queries}] {ds_id} | {q_id} | {persona}")
                    result = self.query(prompt, model=model, temperature=temperature, provider=provider)

                    # Track tokens for summary
                    if result.get('success', True):
                        total_tokens_used += result.get('_tokens', 0)

                    # Add metadata
                    result['persona'] = persona
                    result['discharge_summary_id'] = ds_id
                    result['question_id'] = q_id
                    result['timestamp'] = datetime.now().isoformat()

                    # Only include success when False
                    if result.get('success') is True:
                        result.pop('success', None)

                    # Remove internal token tracking before adding to results
                    result.pop('_tokens', None)

                    all_results.append(result)

                    if result.get('success', True):
                        print(f"  ✓ {result['response'][:80]}...")
                    else:
                        print(f"  ✗ Error: {result['error']}")

        # Store total tokens in results metadata for save_results
        if all_results:
            all_results[0]['_summary_tokens'] = total_tokens_used

        return all_results

    def run_specific_combinations(
        self,
        test_cases: List[Dict[str, Any]],
        model: str = "gpt-5-mini",
        temperature: float = 1,
        enable_reasoning: bool = False,
        provider: str = "openai"
    ) -> List[Dict[str, Any]]:
        """
        Run specific test combinations.

        Args:
            test_cases: List of dicts with keys: persona, ds_id, question_id
            model: ChatGPT model to use
            temperature: Sampling temperature
            enable_reasoning: Whether to ask model to provide reasoning (default: False)

        Returns:
            List of results
        """
        results = []
        total = len(test_cases)
        total_tokens_used = 0  # Track tokens for summary

        print(f"Running {total} specific test cases...")

        for i, test_case in enumerate(test_cases, 1):
            persona = test_case['persona']
            ds_id = test_case['ds_id']
            q_id = test_case['question_id']

            discharge_summary = self.discharge_summaries[ds_id]
            question = self.get_question_text(q_id, ds_id)

            # Determine reasoning instruction based on enable_reasoning parameter
            if enable_reasoning:
                reasoning_instruction = "Explain with your reasoning, then provide the letter."
            else:
                reasoning_instruction = "Answer with only letter"

            # Build prompt
            prompt = self.build_persona_prompt(
                age=persona['age'],
                gender=persona['gender'],
                education=persona['education'],
                ethnicity=persona['ethnicity'],
                doctor_visit=persona['doctor_visit'],
                er_visit_frequency=persona['er_visit_frequency'],
                discharge_summary=discharge_summary,
                question=question,
                model=model,
                reasoning_instruction=reasoning_instruction
            )

            # Query
            print(f"\n[{i}/{total}] {ds_id} | {q_id} | {persona}")
            result = self.query(prompt, model=model, temperature=temperature, provider=provider)

            # Track tokens for summary
            if result.get('success', True):
                total_tokens_used += result.get('_tokens', 0)

            # Add metadata
            result['persona'] = persona
            result['discharge_summary_id'] = ds_id
            result['question_id'] = q_id
            result['timestamp'] = datetime.now().isoformat()

            # Only include success when False
            if result.get('success') is True:
                result.pop('success', None)

            # Remove internal token tracking before adding to results
            result.pop('_tokens', None)

            results.append(result)

            if result.get('success', True):
                print(f"  ✓ {result['response'][:80]}...")
            else:
                print(f"  ✗ Error: {result['error']}")

        # Store total tokens in results metadata for save_results
        if results:
            results[0]['_summary_tokens'] = total_tokens_used

        return results

    def save_results(self, results: List[Dict[str, Any]], output_file: str = "results.json"):
        """Save results to JSON file with summary."""
        # Extract total tokens from metadata before saving
        total_tokens = results[0].pop('_summary_tokens', 0) if results else 0

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n{'='*80}")
        print(f"SUMMARY")
        print(f"{'='*80}")
        print(f"Results saved to: {output_file}")

        successful = sum(1 for r in results if r.get('success', True))

        print(f"Successful queries: {successful}/{len(results)}")
        print(f"Total tokens used: {total_tokens}")
        print(f"{'='*80}")


def main():
    """Example usage - run a small test."""

    try:
        engine = PersonaDischargeQueryEngine()
    except ValueError as e:
        print(f"Error: {e}")
        print("\nSet your OpenAI API key:")
        print("  export OPENAI_API_KEY='your-api-key-here'")
        return

    # Define persona variations for testing
    persona_variations = {
        'age': ['25', '65'],  # Young vs older
        'gender': ['male', 'female'],
        'education': ['high', 'low'],
        'ethnicity': ['White', 'Black'],
        'doctor_visit': ['High', 'Low'],
        'er_visit_frequency': ['High', 'Low']
    }

    # Example 1: Test one DS with a few questions
    print("\n" + "="*80)
    print("EXAMPLE 1: Testing DS1 with Q1, Q2, Q3")
    print("="*80)

    results = engine.run_full_experiment(
        persona_variations=persona_variations,
        discharge_summary_ids=['DS1'],
        question_ids=['Q1', 'Q2', 'Q3'],
        model="o1-mini",
        temperature=1,
        max_personas=2  # Limit to 2 personas for quick test
    )

    engine.save_results(results, "test_results.json")

    # Example 2: Test specific combinations
    print("\n" + "="*80)
    print("EXAMPLE 2: Testing specific combinations")
    print("="*80)

    test_cases = [
        {
            'persona': {
                'age': '25',
                'gender': 'female',
                'education': 'high',
                'ethnicity': 'Asian',
                'doctor_visit': 'High',
                'er_visit_frequency': 'Low'
            },
            'ds_id': 'DS1',
            'question_id': 'Q1'
        },
        {
            'persona': {
                'age': '65',
                'gender': 'male',
                'education': 'low',
                'ethnicity': 'White',
                'doctor_visit': 'Low',
                'er_visit_frequency': 'High'
            },
            'ds_id': 'DS2',
            'question_id': 'Q10'
        }
    ]

    results2 = engine.run_specific_combinations(
        test_cases=test_cases,
        model="o1-mini",
        temperature=1.0
    )

    engine.save_results(results2, "specific_test_results.json")


if __name__ == "__main__":
    main()
