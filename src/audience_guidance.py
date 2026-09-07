AUDIENCE_GUIDANCE_BY_SENIORITY = {
    "intern": (
        "The user's seniority is Intern. Adapt the answer for someone still building "
        "context in this domain. Explain the core idea clearly, define terms that are "
        "needed for this specific answer, and connect facts to simple examples or "
        "workflows when the retrieved context supports it. Do not dilute or omit "
        "important details; make them easier to follow."
    ),
    "junior": (
        "The user's seniority is Junior. Give a practical and structured answer that "
        "helps the user understand what matters, why it matters, and what to do next "
        "when the retrieved context supports next steps. Briefly clarify non-obvious "
        "terms or dependencies without over-explaining familiar basics."
    ),
    "mid-level": (
        "The user's seniority is Mid-level. Assume working familiarity with the domain "
        "and focus on the relationships between facts, dependencies, constraints, "
        "trade-offs, ownership, and implementation or operational implications. Keep "
        "the answer balanced: enough detail to act, not unnecessary background."
    ),
    "senior": (
        "The user's seniority is Senior. Be concise and decision-oriented. Emphasize "
        "constraints, risks, trade-offs, exceptions, source-specific details, and "
        "implications for ownership, architecture, delivery, governance, or operations "
        "when they are supported by the retrieved context. Include basics only when "
        "they are necessary to answer the question."
    ),
}
PROJECT_MANAGER_ROLE_NAMES = {
    "project manager",
    "proje yoneticisi",
}
GENERIC_ROLE_GUIDANCE = (
    "The user's role is {role}. Adapt the explanation to that role's likely "
    "responsibilities and domain context without hiding relevant technical details. "
    "When useful, connect concepts to familiar workflows, systems, integrations, "
    "handoffs, controls, risks, or operational outcomes for that role. Keep all "
    "role-based framing grounded in the retrieved context and avoid unsupported "
    "assumptions about the user's department."
)
PROJECT_MANAGER_ROLE_GUIDANCE = (
    "For a Project Manager, frame the answer around goals, scope, stakeholders, "
    "dependencies, acceptance criteria, delivery impact, risks, decisions, and next "
    "actions. Preserve technical facts when they matter, but explain why they affect "
    "planning, coordination, quality, cost, timeline, or user/business outcomes."
)


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
    role_label = format_role_label(role)
    if not role_label:
        return ""

    guidance = [GENERIC_ROLE_GUIDANCE.format(role=role_label)]
    normalized_role = normalize_role_name(role)
    if any(role_name in normalized_role for role_name in PROJECT_MANAGER_ROLE_NAMES):
        guidance.append(PROJECT_MANAGER_ROLE_GUIDANCE)
    return "\n".join(guidance)


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


def format_role_label(role):
    return " ".join(str(role or "").strip().split())[:80]
