import app


def use_in_memory_profile_store(monkeypatch):
    app.st.session_state.clear()
    app.st.session_state.profile_store = app.normalize_profile_store(
        {"active_profile_id": None, "profiles": []}
    )

    def fake_save_profile_store(store):
        app.st.session_state.profile_store = app.normalize_profile_store(store)

    monkeypatch.setattr(app, "save_profile_store", fake_save_profile_store)


def test_password_hash_verification_accepts_only_matching_password():
    password_hash = app.hash_password("secret-pass")

    assert password_hash != "secret-pass"
    assert app.verify_password("secret-pass", password_hash)
    assert not app.verify_password("wrong-pass", password_hash)


def test_normalize_profile_store_adds_passwordless_mock_profiles():
    store = app.normalize_profile_store({"active_profile_id": None, "profiles": []})
    profiles_by_id = {profile["id"]: profile for profile in store["profiles"]}

    assert "mock-intern" in profiles_by_id
    assert "mock-junior" in profiles_by_id
    assert "mock-mid-level" in profiles_by_id
    assert "mock-senior" in profiles_by_id
    assert "mock-project-manager" in profiles_by_id
    assert profiles_by_id["mock-project-manager"]["role"] == "Proje Yöneticisi"
    assert profiles_by_id["mock-project-manager"]["password_hash"] == ""
    assert profiles_by_id["mock-project-manager"]["is_mock"] is True


def test_mock_profiles_can_activate_without_password():
    profile = app.normalize_profile(
        {
            "id": "mock-project-manager",
            "display_name": "Test Proje Yöneticisi",
            "role": "Proje Yöneticisi",
            "seniority": "Senior",
            "is_mock": True,
        }
    )

    assert app.can_activate_profile(profile)


def test_mock_profiles_are_hidden_from_visible_profile_list():
    store = app.normalize_profile_store({"active_profile_id": None, "profiles": []})

    assert store["profiles"]
    assert app.visible_profiles(store["profiles"]) == []


def test_profile_initials_uses_first_two_name_parts():
    assert app.profile_initials("Ada Lovelace") == "AL"
    assert app.profile_initials("Ada") == "A"
    assert app.profile_initials("") == "P"


def test_active_user_profile_ignores_mock_profile(monkeypatch):
    use_in_memory_profile_store(monkeypatch)
    app.st.session_state.profile_store["active_profile_id"] = "mock-junior"

    assert app.active_user_profile() is None


def test_primary_user_profile_returns_first_visible_profile_when_none_is_active():
    password_hash = app.hash_password("correct-pass")
    store = app.normalize_profile_store(
        {
            "active_profile_id": None,
            "profiles": [
                {
                    "id": "real-user",
                    "display_name": "Real User",
                    "role": "Geliştirici",
                    "seniority": "Senior",
                    "password_hash": password_hash,
                }
            ],
        }
    )
    app.st.session_state.profile_store = store

    assert app.primary_user_profile()["id"] == "real-user"


def test_normal_profiles_require_matching_password_to_activate():
    profile = app.normalize_profile(
        {
            "id": "real-user",
            "display_name": "Real User",
            "role": "Proje Yöneticisi",
            "seniority": "Senior",
            "password_hash": app.hash_password("correct-pass"),
        }
    )

    assert app.can_activate_profile(profile, "correct-pass")
    assert not app.can_activate_profile(profile, "wrong-pass")
    assert not app.can_activate_profile(profile)


def test_save_profile_requires_password_for_new_normal_profile(monkeypatch):
    use_in_memory_profile_store(monkeypatch)

    try:
        app.save_profile(
            display_name="Real User",
            role="Geliştirici",
            seniority="Junior",
            password="",
        )
    except ValueError as exc:
        assert "şifre belirlenmesi zorunludur" in str(exc)
    else:
        raise AssertionError("Expected password validation error.")


def test_save_profile_hashes_password_and_activates_profile(monkeypatch):
    use_in_memory_profile_store(monkeypatch)

    profile = app.save_profile(
        display_name="Real User",
        role="Proje Yöneticisi",
        seniority="Senior",
        password="correct-pass",
    )

    saved_profile = app.find_profile(profile["id"])
    assert app.st.session_state.profile_store["active_profile_id"] == profile["id"]
    assert saved_profile["password_hash"]
    assert saved_profile["password_hash"] != "correct-pass"
    assert app.verify_password("correct-pass", saved_profile["password_hash"])
