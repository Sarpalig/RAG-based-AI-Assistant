from src.audience_guidance import build_audience_guidance, build_role_guidance


def test_project_manager_role_gets_product_quality_guidance():
    guidance = build_role_guidance("Project Manager")

    assert "The user's role is Project Manager" in guidance
    assert "goals, scope, stakeholders" in guidance
    assert "Preserve technical facts" in guidance
    assert "planning, coordination, quality" in guidance


def test_project_manager_role_matches_variants():
    role_variants = [
        "Senior Project Manager",
        "project manager",
        "Proje Yoneticisi",
        "Proje Yöneticisi",
    ]

    for role in role_variants:
        guidance = build_role_guidance(role)

        assert f"The user's role is {role}" in guidance
        assert "For a Project Manager" in guidance


def test_non_project_manager_role_gets_general_role_guidance_only():
    guidance = build_role_guidance("SAP Consultant")

    assert "The user's role is SAP Consultant" in guidance
    assert "systems, integrations" in guidance
    assert "For a Project Manager" not in guidance


def test_project_manager_guidance_combines_with_seniority_guidance():
    guidance = build_audience_guidance(
        {"role": "Project Manager", "seniority": "Senior"}
    )

    assert "goals, scope, stakeholders" in guidance
    assert "Preserve technical facts" in guidance
    assert "The user's seniority is Senior" in guidance
    assert "decision-oriented" in guidance


def test_seniority_guidance_scales_detail_without_department_assumptions():
    intern_guidance = build_audience_guidance({"seniority": "Intern"})
    junior_guidance = build_audience_guidance({"seniority": "Junior"})
    mid_guidance = build_audience_guidance({"seniority": "Mid-level"})
    senior_guidance = build_audience_guidance({"seniority": "Senior"})

    assert "still building context" in intern_guidance
    assert "Do not dilute or omit important details" in intern_guidance
    assert "what matters, why it matters" in junior_guidance
    assert "working familiarity" in mid_guidance
    assert "decision-oriented" in senior_guidance
    assert "chunking" not in senior_guidance
    assert "embedding" not in senior_guidance


def test_unknown_profile_has_no_guidance():
    assert build_audience_guidance(None) == ""
    assert build_audience_guidance({"seniority": "Lead"}) == ""
