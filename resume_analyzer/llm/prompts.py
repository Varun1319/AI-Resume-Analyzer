# llm/prompts.py

ANSWER_TEMPLATE = """
You are an assistant that analyzes resumes carefully.

Context:
\"\"\"{context}\"\"\"

Instruction:
{prompt}

Answer clearly and professionally.
"""


def summary_prompt(chunks_str: str) -> str:
    return (
        "Provide a **comprehensive and well-structured summary** of this resume. "
        "Include key skills, educational highlights, technical expertise, and achievements. "
        "Also add a concluding note on overall candidate suitability. "
        f"Resume content:\n\n{chunks_str}"
    )


def strength_prompt(chunks_str: str) -> str:
    return (
        "Analyze this resume and write a **detailed list of the candidate strengths**. "
        "Include technical strengths, soft skills, leadership abilities, teamwork, achievements, "
        "and any unique differentiating factors. Give **short examples** from the text wherever possible. "
        f"Resume content:\n\n{chunks_str}"
    )


def weakness_prompt(chunks_str: str) -> str:
    return (
        "Critically analyze this resume to identify **weak areas or gaps** in terms of formatting, "
        "missing keywords, or lack of quantifiable results. Suggest **specific improvements** to make "
        "it more impactful for recruiters or ATS. "
        f"Resume content:\n\n{chunks_str}"
    )


def job_title_prompt(chunks_str: str) -> str:
    return (
        "Based on this resume, list **5–10 most suitable job roles or titles**, categorized by domain "
        "(e.g., Software Engineering, Data, Product, etc.) and include **a one-line reason** why each "
        "fits the profile. "
        f"Resume content:\n\n{chunks_str}"
    )


def build_prompts(resume_text: str) -> dict[str, str]:
    return {
        "summary": summary_prompt(resume_text),
        "strengths": strength_prompt(resume_text),
        "weaknesses": weakness_prompt(resume_text),
        "job_titles": job_title_prompt(resume_text),
    }
