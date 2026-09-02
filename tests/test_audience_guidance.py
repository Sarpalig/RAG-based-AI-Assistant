from src.audience_guidance import build_audience_guidance, build_role_guidance


def test_project_manager_role_gets_product_quality_guidance():
    guidance = build_role_guidance("Project Manager")

    assert "The user's role is Project Manager" in guidance
    assert "Focus on product quality" in guidance
    assert "Avoid low-level implementation details" in guidance
    assert "chunking, embedding generation" in guidance


def test_project_manager_role_matches_variants():
    role_variants = [
        "Senior Project Manager",
        "project manager",
        "Proje Yoneticisi",
        "Proje Yöneticisi",
    ]

    for role in role_variants:
        assert "The user's role is Project Manager" in build_role_guidance(role)


def test_non_project_manager_role_does_not_get_project_manager_guidance():
    assert build_role_guidance("Backend Developer") == ""


def test_project_manager_guidance_combines_with_seniority_guidance():
    guidance = build_audience_guidance(
        {"role": "Project Manager", "seniority": "Senior"}
    )

    assert "Focus on product quality" in guidance
    assert "Avoid low-level implementation details" in guidance
    assert "The user is senior-level" in guidance


def test_unknown_profile_has_no_guidance():
    assert build_audience_guidance(None) == ""
    assert build_audience_guidance({"role": "Backend Developer", "seniority": "Lead"}) == ""
