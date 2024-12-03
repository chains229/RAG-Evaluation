from pydantic import BaseModel
import google.generativeai as genai
import instructor
from deepeval.models import DeepEvalBaseLLM

class CustomGemini(DeepEvalBaseLLM):
    """
    Define Gemini as a Judge for deepeval
    """
    def __init__(self):
        self.model = genai.GenerativeModel("models/gemini-1.5-flash-002")

    def load_model(self):
        return self.model

    def generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        client = self.load_model()
        instructor_client = instructor.from_gemini(
            client=client,
            mode=instructor.Mode.GEMINI_JSON,
        )
        resp = instructor_client.messages.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            response_model=schema,
        )
        return resp

    async def a_generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        return self.generate(prompt, schema)

    def get_model_name(self):
        return "Gemini 1.5"

def eval_steps(level: str):
    "Contains the evaluation steps for each level. This will be used for reference-required metrics"
    if level == "Remember":
        return ["Check if the generated answer is correct and directly relevant to the user query"]
    elif level == "Understand":
        return ["Assess how thoroughly the candidate summarization captures the key points and essential information from the ground truth summarization",
                "Determine whether the candidate summarization accurately represents the facts and details provided in the ground truth without introducing errors",
                "Evaluate how effectively the candidate summarization rephrases the information from the ground truth while maintaining the original meaning",
                "Judge the overall similarity in content and meaning between the candidate summarization and the ground truth summarization"
        ]
    elif level == "Apply":
        return ["Assess how effectively the Candidate Solution applies relevant knowledge to address the specific scenario in the Query",
                "Evaluate whether the Candidate Solution covers all critical aspects and requirements outlined in the Query, as addressed by the Expert Solution",
                "Determine if the information provided in the Candidate Solution is factually correct and aligns with the Expert Solution without introducing errors",
                "Judge whether the Candidate Solution offers practical and actionable advice appropriate for the scenario"
        ]
    elif level == "Analyze":
        return ["Assess how well the Candidate Response incorporates a range of perspectives as reflected in the Partial Answers.",
                "Determine whether the Candidate Response acknowledges that the question is debatable, indicating that differing viewpoints exist",
                "Judge how well the Candidate Response aligns with the core ideas in the Partial Answers, reflecting factual accuracy and relevance to the perspectives provided"
        ]
    elif level == "Evaluate":
        return ["Assess how well the Candidate Response incorporates a range of perspectives as reflected in the Partial Answers.",
                "Determine whether the Candidate Response acknowledges that the question is debatable, indicating that differing viewpoints exist",
                "Judge how well the Candidate Response aligns with the core ideas in the Partial Answers, reflecting factual accuracy and relevance to the perspectives provided"
        ]
    else:
        return ["Evaluate whether the candidate presents the paragraph according to one of the acceptable structures: deductive, inductive, general-specific-general, chain, or parallel structure.",
                "Assess if the candidate accurately identifies the main issue for discussion",
                "Ensure proper Vietnamese spelling and grammar. Points may be deducted for excessive spelling or grammatical errors.",
                "Evaluate the depth of thought, originality in expression, and the candidate’s ability to draw from personal knowledge and experience to discuss the issue"
        ]
    
