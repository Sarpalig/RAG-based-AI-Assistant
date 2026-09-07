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
    assert profiles_by_id["mock-project-manager"]["role"] == "Project Manager"
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


def test_role_label_returns_display_label_for_known_role():
    assert app.role_label("Project Manager") == "Proje Yöneticisi"
    assert app.role_label("SAP Consultant") == "SAP Danışmanı"


def test_role_options_preserve_existing_custom_role():
    options = app.role_options_for_current_value("Legacy Role")

    assert "Backend Developer" in options
    assert "Developer" not in options
    assert "Legacy Role" in options


def test_normalize_profile_migrates_old_general_developer_role():
    profile = app.normalize_profile(
        {
            "display_name": "Real User",
            "role": "Geliştirici",
            "seniority": "Junior",
        }
    )

    assert profile["role"] == "Backend Developer"


def test_normalize_profile_migrates_turkish_project_manager_role():
    profile = app.normalize_profile(
        {
            "display_name": "Real User",
            "role": "Proje Yöneticisi",
            "seniority": "Senior",
        }
    )

    assert profile["role"] == "Project Manager"


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
    app.st.session_state.profile_logged_out = False

    assert app.primary_user_profile()["id"] == "real-user"


def test_primary_user_profile_returns_none_after_logout():
    password_hash = app.hash_password("correct-pass")
    app.st.session_state.profile_logged_out = True
    app.st.session_state.profile_store = app.normalize_profile_store(
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

    assert app.primary_user_profile() is None


def test_open_new_profile_form_prepares_creation_state():
    app.st.session_state.editing_profile_id = "real-user"
    app.st.session_state.profile_form_open = False
    app.st.session_state.confirm_delete_profile_id = "real-user"

    app.open_new_profile_form()

    assert app.st.session_state.editing_profile_id is None
    assert app.st.session_state.profile_form_open is True
    assert app.st.session_state.confirm_delete_profile_id is None


def test_logout_profile_clears_active_profile_and_chat(monkeypatch):
    use_in_memory_profile_store(monkeypatch)
    app.st.session_state.profile_store["active_profile_id"] = "real-user"
    app.st.session_state.messages = [{"role": "user", "content": "Merhaba"}]
    app.st.session_state.editing_profile_id = "real-user"
    app.st.session_state.profile_form_open = True
    app.st.session_state.confirm_delete_profile_id = "real-user"
    app.st.session_state.profile_logged_out = False

    app.logout_profile()

    assert app.st.session_state.profile_store["active_profile_id"] is None
    assert app.st.session_state.messages == []
    assert app.st.session_state.editing_profile_id is None
    assert app.st.session_state.profile_form_open is False
    assert app.st.session_state.confirm_delete_profile_id is None
    assert app.st.session_state.profile_logged_out is True


def test_find_visible_profile_by_name_matches_without_exposing_profile_choices():
    password_hash = app.hash_password("correct-pass")
    profiles = app.visible_profiles(
        app.normalize_profile_store(
            {
                "active_profile_id": None,
                "profiles": [
                    {
                        "id": "real-user",
                        "display_name": "Lionel Messi",
                        "role": "Geliştirici",
                        "seniority": "Senior",
                        "password_hash": password_hash,
                    }
                ],
            }
        )["profiles"]
    )

    assert app.find_visible_profile_by_name("  lionel   messi  ", profiles)["id"] == "real-user"
    assert app.find_visible_profile_by_name("Cristiano Ronaldo", profiles) is None


def test_authenticate_profile_requires_matching_name_and_password():
    password_hash = app.hash_password("correct-pass")
    profiles = app.visible_profiles(
        app.normalize_profile_store(
            {
                "active_profile_id": None,
                "profiles": [
                    {
                        "id": "real-user",
                        "display_name": "Lionel Messi",
                        "role": "Geliştirici",
                        "seniority": "Senior",
                        "password_hash": password_hash,
                    }
                ],
            }
        )["profiles"]
    )

    assert app.authenticate_profile("Lionel Messi", "correct-pass", profiles)["id"] == "real-user"
    assert app.authenticate_profile("Lionel Messi", "wrong-pass", profiles) is None
    assert app.authenticate_profile("Unknown User", "correct-pass", profiles) is None


def test_recent_chat_history_keeps_last_user_turns_with_assistant_replies():
    messages = [
        {"role": "user", "content": "Question 1"},
        {"role": "assistant", "content": "Answer 1"},
        {"role": "user", "content": "Question 2"},
        {"role": "assistant", "content": "Answer 2"},
        {"role": "user", "content": "Question 3"},
        {"role": "assistant", "content": "Answer 3"},
    ]

    history = app.recent_chat_history(messages, max_user_turns=2)

    assert history == [
        {"role": "user", "content": "Question 2"},
        {"role": "assistant", "content": "Answer 2"},
        {"role": "user", "content": "Question 3"},
        {"role": "assistant", "content": "Answer 3"},
    ]


def test_recent_chat_history_ignores_empty_and_unknown_messages():
    messages = [
        {"role": "system", "content": "Hidden"},
        {"role": "user", "content": "   "},
        {"role": "assistant", "content": "Useful answer"},
        {"role": "user", "content": "Useful question"},
    ]

    history = app.recent_chat_history(messages, max_user_turns=1)

    assert history == [
        {"role": "user", "content": "Useful question"},
    ]


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
    assert saved_profile["role"] == "Project Manager"
    assert saved_profile["password_hash"]
    assert saved_profile["password_hash"] != "correct-pass"
    assert app.verify_password("correct-pass", saved_profile["password_hash"])
