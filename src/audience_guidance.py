AUDIENCE_GUIDANCE_BY_SENIORITY = {
    "intern": (
        "The user is an intern. Explain the answer step by step, define important "
        "terms briefly, and avoid assuming prior domain knowledge. Keep every claim "
        "grounded in the retrieved context."
    ),
    "junior": (
        "The user is junior-level. Use a practical, structured explanation, call out "
        "non-obvious terms, and include concrete next steps when the context supports "
        "them."
    ),
    "mid-level": (
        "The user is mid-level. Use a balanced technical explanation, assume basic "
        "professional familiarity, and focus on decisions, dependencies, and important "
        "details from the context."
    ),
    "senior": (
        "The user is senior-level. Be concise, emphasize constraints, risks, trade-offs, "
        "and source-specific details, and skip basic explanations unless the question "
        "asks for them."
    ),
}
PROJECT_MANAGER_ROLE_NAMES = {
    "project manager",
    "proje yoneticisi",
}
AUDIENCE_GUIDANCE_BY_ROLE = {
    "project_manager": (
        "The user's role is Project Manager. Focus on product quality, user impact, "
        "acceptance criteria, risks, dependencies, scope, timeline, and decision "
        "points. Avoid low-level implementation details such as chunking, embedding "
        "generation, vector-store internals, or model plumbing unless the user "
        "explicitly asks for them. Translate technical details into product "
        "implications and concrete actions when the context supports them."
    ),
}


def build_audience_guidance(user_profile):
    if not user_profile:
        return ""

    if isinstance(user_profile, str):
        seniority = user_profile
        role = ""
    else:
        seniority = user_profile.get("seniority", "")
        role = user_profile.get("role", "")

    normalized_seniority = str(seniority).strip().casefold()
    guidance_parts = []

    role_guidance = build_role_guidance(role)
    if role_guidance:
        guidance_parts.append(role_guidance)

    seniority_guidance = AUDIENCE_GUIDANCE_BY_SENIORITY.get(normalized_seniority)
    if seniority_guidance:
        guidance_parts.append(seniority_guidance)

    return "\n".join(guidance_parts)


def build_role_guidance(role):
    normalized_role = normalize_role_name(role)
    if any(role_name in normalized_role for role_name in PROJECT_MANAGER_ROLE_NAMES):
        return AUDIENCE_GUIDANCE_BY_ROLE["project_manager"]
    return ""


def normalize_role_name(role):
    translation = str.maketrans(
        {
            "ç": "c",
            "ğ": "g",
            "ı": "i",
            "ö": "o",
            "ş": "s",
            "ü": "u",
            "Ç": "c",
            "Ğ": "g",
            "İ": "i",
            "I": "i",
            "Ö": "o",
            "Ş": "s",
            "Ü": "u",
        }
    )
    return str(role).strip().translate(translation).casefold()
