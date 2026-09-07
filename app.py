import os
import hmac
import html
import hashlib
import json
import secrets
from pathlib import Path
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv

from src.generation import LLMError
from src.loaders import DocumentLoaderError
from src.rag_pipeline import RagPipeline
from src.vector_store import create_chroma_client, list_indexed_documents


DEFAULT_COLLECTION_NAME = "rag_documents"
PROFILE_STORE_PATH = Path(__file__).resolve().parent / "data" / "user_profiles.json"
OLLAMA_MODEL_OPTIONS = ["qwen3.5:9b", "gemma4:e4b"]
PAGE_OPTIONS = ["Sohbet", "Profil", "Belgeler"]
CHAT_MEMORY_USER_TURNS = 5
NAVIGATION_ITEMS = [
    {"page": "Sohbet", "label": "Sohbet", "icon": "+"},
    {"page": "Profil", "label": "Profil", "icon": "☰"},
    {"page": "Belgeler", "label": "Belgeler", "icon": "⌕"},
]
SENIORITY_OPTIONS = ["Intern", "Junior", "Mid-level", "Senior"]
SENIORITY_LABELS = {
    "Intern": "Stajyer",
    "Junior": "Junior",
    "Mid-level": "Orta seviye",
    "Senior": "Senior",
}
ROLE_OPTIONS = [
    "Backend Developer",
    "Frontend Developer",
    "Full Stack Developer",
    "Data Engineer",
    "Data Scientist",
    "Machine Learning Engineer",
    "MLOps Engineer",
    "BI Analyst",
    "Data Analyst",
    "Business Analyst",
    "SAP Consultant",
    "Integration Consultant",
    "Project Manager",
    "Product Owner",
    "DevOps/SRE",
    "Security Specialist",
    "QA Engineer",
]
ROLE_LABELS = {
    "Backend Developer": "Backend Developer",
    "Frontend Developer": "Frontend Developer",
    "Full Stack Developer": "Full Stack Developer",
    "Data Engineer": "Data Engineer",
    "Data Scientist": "Data Scientist",
    "Machine Learning Engineer": "ML Engineer",
    "MLOps Engineer": "MLOps Engineer",
    "BI Analyst": "BI Analyst",
    "Data Analyst": "Data Analyst",
    "Business Analyst": "Business Analyst",
    "SAP Consultant": "SAP Danışmanı",
    "Integration Consultant": "Entegrasyon Danışmanı",
    "Project Manager": "Proje Yöneticisi",
    "Product Owner": "Product Owner",
    "DevOps/SRE": "DevOps / SRE",
    "Security Specialist": "Güvenlik Uzmanı",
    "QA Engineer": "QA Engineer",
}
PASSWORD_HASH_ITERATIONS = 120000
DEFAULT_PROFILE_STORE = {
    "active_profile_id": None,
    "profiles": [],
}
EMPTY_PROFILE_FORM = {
    "display_name": "",
    "role": ROLE_OPTIONS[0],
    "seniority": "Junior",
}
MOCK_PROFILES = [
    {
        "id": "mock-intern",
        "display_name": "Test Stajyer",
        "role": "Backend Developer",
        "seniority": "Intern",
        "is_mock": True,
    },
    {
        "id": "mock-junior",
        "display_name": "Test Junior",
        "role": "Backend Developer",
        "seniority": "Junior",
        "is_mock": True,
    },
    {
        "id": "mock-mid-level",
        "display_name": "Test Orta Seviye",
        "role": "Backend Developer",
        "seniority": "Mid-level",
        "is_mock": True,
    },
    {
        "id": "mock-senior",
        "display_name": "Test Senior",
        "role": "Backend Developer",
        "seniority": "Senior",
        "is_mock": True,
    },
    {
        "id": "mock-project-manager",
        "display_name": "Test Proje Yöneticisi",
        "role": "Project Manager",
        "seniority": "Senior",
        "is_mock": True,
    },
]


@st.cache_resource
def get_pipeline(ollama_model):
    return RagPipeline(ollama_model=ollama_model)


def configured_ollama_model():
    load_dotenv()
    env_model = os.getenv("OLLAMA_MODEL", OLLAMA_MODEL_OPTIONS[0])
    if env_model in OLLAMA_MODEL_OPTIONS:
        return env_model
    return OLLAMA_MODEL_OPTIONS[0]


def selected_ollama_model():
    return st.session_state.get("selected_ollama_model", configured_ollama_model())


def selected_page():
    return st.session_state.get("selected_page", "Sohbet")


def hash_password(password):
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        PASSWORD_HASH_ITERATIONS,
    ).hex()
    return f"pbkdf2_sha256${PASSWORD_HASH_ITERATIONS}${salt}${digest}"


def verify_password(password, password_hash):
    if not password or not password_hash:
        return False

    try:
        algorithm, iterations, salt, expected_digest = password_hash.split("$", 3)
        iterations = int(iterations)
    except (AttributeError, TypeError, ValueError):
        return False

    if algorithm != "pbkdf2_sha256":
        return False

    actual_digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        iterations,
    ).hex()
    return hmac.compare_digest(actual_digest, expected_digest)


def is_mock_profile(profile):
    if not profile:
        return False
    return bool(profile.get("is_mock"))


def requires_password(profile):
    return not is_mock_profile(profile)


def normalize_profile(profile):
    seniority = profile.get("seniority", EMPTY_PROFILE_FORM["seniority"])
    if seniority not in SENIORITY_OPTIONS:
        seniority = EMPTY_PROFILE_FORM["seniority"]

    profile_id = profile.get("id") or uuid4().hex
    is_mock = bool(profile.get("is_mock", False))
    return {
        "id": str(profile_id),
        "display_name": str(profile.get("display_name", "")).strip(),
        "role": normalize_role_value(profile.get("role")),
        "seniority": seniority,
        "password_hash": "" if is_mock else str(profile.get("password_hash", "")),
        "is_mock": is_mock,
    }


def merge_mock_profiles(profiles):
    profiles_by_id = {profile["id"]: profile for profile in profiles}
    for mock_profile in MOCK_PROFILES:
        if mock_profile["id"] not in profiles_by_id:
            profiles_by_id[mock_profile["id"]] = normalize_profile(mock_profile)
    return list(profiles_by_id.values())


def normalize_profile_store(store):
    if not isinstance(store, dict):
        store = DEFAULT_PROFILE_STORE.copy()

    profiles = [
        normalize_profile(profile)
        for profile in store.get("profiles", [])
        if isinstance(profile, dict)
    ]
    profiles = merge_mock_profiles(profiles)
    profile_ids = {profile["id"] for profile in profiles}
    active_profile_id = store.get("active_profile_id")
    if active_profile_id not in profile_ids:
        active_profile_id = None

    return {
        "active_profile_id": active_profile_id,
        "profiles": profiles,
    }


def load_profile_store():
    if not PROFILE_STORE_PATH.exists():
        return DEFAULT_PROFILE_STORE.copy()

    try:
        data = json.loads(PROFILE_STORE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return DEFAULT_PROFILE_STORE.copy()

    return normalize_profile_store(data)


def save_profile_store(store):
    normalized_store = normalize_profile_store(store)
    PROFILE_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROFILE_STORE_PATH.write_text(
        json.dumps(normalized_store, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    st.session_state.profile_store = normalized_store


def find_profile(profile_id):
    if not profile_id:
        return None

    for profile in st.session_state.get("profile_store", {}).get("profiles", []):
        if profile.get("id") == profile_id:
            return profile

    return None


def active_user_profile():
    store = st.session_state.get("profile_store", DEFAULT_PROFILE_STORE)
    profile = find_profile(store.get("active_profile_id"))
    if is_mock_profile(profile) or (profile and not profile.get("password_hash")):
        return None
    return profile


def current_user_profile():
    profile = active_user_profile()
    if not profile:
        return None

    return {
        "display_name": profile["display_name"],
        "role": profile["role"],
        "seniority": profile["seniority"],
    }


def profile_label(profile):
    name = profile.get("display_name") or "İsimsiz profil"
    role = role_label(profile.get("role"))
    seniority = seniority_label(profile.get("seniority", "Genel"))
    if role:
        return f"{name} - {role} ({seniority})"
    return f"{name} ({seniority})"


def seniority_label(seniority):
    return SENIORITY_LABELS.get(seniority, seniority)


def role_label(role):
    return ROLE_LABELS.get(role, role or "Rol belirtilmedi")


def normalize_role_value(role):
    role = str(role or "").strip()
    if role in {"Developer", "Geliştirici", "GeliÅŸtirici"}:
        return "Backend Developer"
    if role in {"Proje Yöneticisi", "Proje YÃ¶neticisi"}:
        return "Project Manager"
    return role


def role_options_for_current_value(role):
    role = normalize_role_value(role)
    if role and role not in ROLE_OPTIONS:
        return ROLE_OPTIONS + [role]
    return ROLE_OPTIONS


def profile_initials(display_name):
    name_parts = [part for part in str(display_name).strip().split() if part]
    if not name_parts:
        return "P"
    initials = "".join(part[0] for part in name_parts[:2])
    return initials.upper()


def can_activate_profile(profile, password=None):
    if not profile:
        return False
    if is_mock_profile(profile):
        return True
    return verify_password(password or "", profile.get("password_hash", ""))


def visible_profiles(profiles):
    return [
        profile
        for profile in profiles
        if not is_mock_profile(profile) and profile.get("password_hash")
    ]


def normalize_profile_name(display_name):
    return " ".join(str(display_name).strip().casefold().split())


def find_visible_profile_by_name(display_name, profiles=None):
    store = st.session_state.get("profile_store", DEFAULT_PROFILE_STORE.copy())
    candidate_profiles = profiles or visible_profiles(store.get("profiles", []))
    normalized_name = normalize_profile_name(display_name)
    if not normalized_name:
        return None

    for profile in candidate_profiles:
        if normalize_profile_name(profile.get("display_name", "")) == normalized_name:
            return profile
    return None


def authenticate_profile(display_name, password, profiles=None):
    profile = find_visible_profile_by_name(display_name, profiles)
    if not profile:
        return None
    if not can_activate_profile(profile, password):
        return None
    return profile


def primary_user_profile():
    active_profile = active_user_profile()
    if active_profile:
        return active_profile
    if st.session_state.get("profile_logged_out"):
        return None

    store = st.session_state.get("profile_store", DEFAULT_PROFILE_STORE.copy())
    profiles = visible_profiles(store.get("profiles", []))
    if profiles:
        return profiles[0]
    return None


def set_active_profile(profile_id):
    store = st.session_state.get("profile_store", DEFAULT_PROFILE_STORE.copy())
    save_profile_store(
        {
            "active_profile_id": profile_id,
            "profiles": store.get("profiles", []),
        }
    )


def open_new_profile_form():
    st.session_state.editing_profile_id = None
    st.session_state.profile_form_open = True
    st.session_state.confirm_delete_profile_id = None


def logout_profile():
    set_active_profile(None)
    st.session_state.messages = []
    st.session_state.editing_profile_id = None
    st.session_state.profile_form_open = False
    st.session_state.confirm_delete_profile_id = None
    st.session_state.profile_logged_out = True


def load_pipeline(show_status=False, ollama_model=None):
    model = ollama_model or selected_ollama_model()

    if not show_status:
        pipeline = get_pipeline(model)
        if not hasattr(pipeline, "list_indexed_documents"):
            get_pipeline.clear()
            pipeline = get_pipeline(model)
        return pipeline

    with st.status("RAG işlem hattı hazırlanıyor...", expanded=False) as status:
        status.write("Gömme modeli ve vektör veritabanı yükleniyor.")
        pipeline = get_pipeline(model)
        if not hasattr(pipeline, "list_indexed_documents"):
            get_pipeline.clear()
            pipeline = get_pipeline(model)
        status.update(label="RAG işlem hattı hazır.", state="complete")

    return pipeline


def initialize_session_state():
    if "indexed_documents" not in st.session_state:
        st.session_state.indexed_documents = {}
    if "indexed_documents_loaded" not in st.session_state:
        st.session_state.indexed_documents_loaded = False
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "selected_ollama_model" not in st.session_state:
        st.session_state.selected_ollama_model = configured_ollama_model()
    if "selected_page" not in st.session_state:
        st.session_state.selected_page = "Sohbet"
    if "document_notice" not in st.session_state:
        st.session_state.document_notice = None
    if "profile_notice" not in st.session_state:
        st.session_state.profile_notice = None
    if "profile_store" not in st.session_state:
        st.session_state.profile_store = load_profile_store()
    if "editing_profile_id" not in st.session_state:
        st.session_state.editing_profile_id = None
    if "profile_form_open" not in st.session_state:
        st.session_state.profile_form_open = False
    if "confirm_delete_profile_id" not in st.session_state:
        st.session_state.confirm_delete_profile_id = None
    if "profile_logged_out" not in st.session_state:
        st.session_state.profile_logged_out = False


def load_indexed_documents_on_startup():
    if st.session_state.indexed_documents_loaded:
        return

    try:
        load_dotenv()
        chroma_dir = os.getenv("CHROMA_DIR", "chroma_db")
        client = create_chroma_client(chroma_dir)
        collection = client.get_or_create_collection(name=DEFAULT_COLLECTION_NAME)
        documents = list_indexed_documents(collection)
        st.session_state.indexed_documents = {
            document["document_hash"]: document
            for document in documents
        }
        st.session_state.indexed_documents_loaded = True
    except Exception as exc:
        st.session_state.indexed_documents_loaded = True
        st.warning(f"Yüklü belge listesi okunamadı: {exc}")


def remember_index_result(result):
    document_hash = result.get("document_hash")
    if not document_hash:
        return

    st.session_state.indexed_documents[document_hash] = {
        "file_name": result.get("file_name", "Bilinmeyen dosya"),
        "chunk_count": result.get("chunk_count", 0),
        "skipped": result.get("skipped", False),
    }


def refresh_indexed_documents(rag):
    documents = rag.list_indexed_documents()
    st.session_state.indexed_documents = {
        document["document_hash"]: document
        for document in documents
    }


def document_status_label(document):
    if document["skipped"]:
        return "Zaten yüklü"
    if document["chunk_count"] > 0:
        return "Hazır"
    return "Metin bulunamadı"


def show_indexed_documents():
    if not st.session_state.indexed_documents:
        st.caption("Henüz yüklenen belge yok.")
        return

    for document in st.session_state.indexed_documents.values():
        status = document_status_label(document)
        st.markdown(f"**{document['file_name']}**")
        st.caption(f"{status} - {document['chunk_count']} parça")


def index_uploaded_files(uploaded_files):
    if not uploaded_files:
        st.warning("Lütfen önce en az bir belge yükleyin.")
        return

    try:
        rag = load_pipeline(show_status=True)
    except Exception as exc:
        st.error(f"İşlem hattı başlatılırken hata oluştu: {exc}")
        return

    with st.spinner("Belgeler hazırlanıyor..."):
        for uploaded_file in uploaded_files:
            try:
                result = rag.index_file(uploaded_file, uploaded_file.name)
                remember_index_result(result)

                if result["skipped"]:
                    st.info(f"{result['file_name']} zaten yüklü.")
                elif result["chunk_count"] > 0:
                    st.success(
                        f"{result['file_name']} hazırlandı "
                        f"({result['chunk_count']} parça)."
                    )
                else:
                    st.warning(
                        f"{result['file_name']} içinde okunabilir metin bulunamadı."
                    )
            except DocumentLoaderError as exc:
                st.error(f"{uploaded_file.name}: {exc}")
            except Exception as exc:
                st.error(f"{uploaded_file.name}: Belge hazırlanırken hata oluştu: {exc}")

    refresh_indexed_documents(rag)


def refresh_document_list():
    try:
        rag = load_pipeline(show_status=True)
        refresh_indexed_documents(rag)
        st.success("Yüklenen belge listesi yenilendi.")
    except Exception as exc:
        st.error(f"Belge listesi yenilenirken hata oluştu: {exc}")


def delete_indexed_document(document_hash):
    document = st.session_state.indexed_documents.get(document_hash)
    file_name = document.get("file_name", "Belge") if document else "Belge"

    try:
        rag = load_pipeline(show_status=True)
        rag.delete_document(document_hash)
        refresh_indexed_documents(rag)
        st.session_state.document_notice = {
            "type": "success",
            "message": f"{file_name} indeksten kaldırıldı.",
        }
    except Exception as exc:
        st.error(f"{file_name} silinirken hata oluştu: {exc}")


def recent_chat_history(messages, max_user_turns=CHAT_MEMORY_USER_TURNS):
    if max_user_turns <= 0:
        return []

    selected_messages = []
    user_turns = 0
    for message in reversed(messages):
        role = message.get("role")
        content = str(message.get("content", "")).strip()
        if role not in {"user", "assistant"} or not content:
            continue

        selected_messages.append({"role": role, "content": content})
        if role == "user":
            user_turns += 1
            if user_turns >= max_user_turns:
                break

    return list(reversed(selected_messages))


def answer_question(query, ollama_model=None):
    if not query.strip():
        raise ValueError("Lütfen önce bir soru yazın.")

    try:
        rag = load_pipeline(ollama_model=ollama_model)
        chat_history = recent_chat_history(st.session_state.get("messages", []))
        with st.spinner("Yanıt hazırlanıyor..."):
            return rag.answer_query(
                query,
                user_profile=current_user_profile(),
                chat_history=chat_history,
            )
    except LLMError as exc:
        raise LLMError(f"LLM hatası: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"Soru yanıtlanırken hata oluştu: {exc}") from exc


def render_citations(citations, expanded=False):
    if citations:
        with st.expander("Kaynaklar", expanded=expanded):
            for citation in citations:
                st.write(f"- {citation}")
    else:
        st.info("Bu yanıt için gösterilecek kaynak bulunamadı.")


def render_message(message):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        citations = message.get("citations") or []
        if citations:
            render_citations(citations, expanded=False)


def render_chat():
    if not st.session_state.messages:
        st.info(
            "Belgeler hazır olduğunda aşağıdaki alandan soru sorabilirsiniz. "
            "Yanıtlar yalnızca indekslenen doküman parçalarına dayanır."
        )

    for message in st.session_state.messages:
        render_message(message)


def render_chat_settings():
    settings = st.columns([2, 3, 1])
    current_model = selected_ollama_model()
    selected_index = (
        OLLAMA_MODEL_OPTIONS.index(current_model)
        if current_model in OLLAMA_MODEL_OPTIONS
        else 0
    )

    with settings[0]:
        st.selectbox(
            "Ollama modeli",
            options=OLLAMA_MODEL_OPTIONS,
            index=selected_index,
            key="selected_ollama_model",
        )

    with settings[1]:
        active_profile = active_user_profile()
        if active_profile:
            st.markdown(f"**Aktif profil:** {profile_label(active_profile)}")
            st.caption("Yanıt anlatımı aktif profile göre uyarlanır.")
        else:
            st.markdown("**Aktif profil:** Yok")
            st.caption("Yanıtlar genel seviyede üretilir.")

    with settings[2]:
        st.write("")
        if st.button("Sohbeti temizle", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


def render_sidebar():
    with st.sidebar:
        st.markdown(
            """
            <style>
            [data-testid="stSidebar"] {
                background-color: #20211f;
            }
            [data-testid="stSidebar"] div.stButton > button {
                width: 100%;
                justify-content: flex-start;
                border: 0;
                border-radius: 8px;
                padding: 0.82rem 0.85rem;
                background: transparent;
                color: #f4f4f2;
                font-size: 1.14rem;
                font-weight: 600;
            }
            [data-testid="stSidebar"] div.stButton > button:hover {
                background: #2a2a27;
                color: #ffffff;
            }
            [data-testid="stSidebar"] div.stButton > button[kind="primary"] {
                background: #2b2926;
                color: #ffffff;
            }
            [data-testid="stSidebar"] div.stButton > button[kind="primary"]:hover {
                background: #302e2b;
                color: #ffffff;
            }
            [data-testid="stSidebar"] div.stButton > button:focus {
                box-shadow: none;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        for item in NAVIGATION_ITEMS:
            is_active = selected_page() == item["page"]
            if st.button(
                f"{item['icon']}  {item['label']}",
                key=f"nav_{item['page']}",
                type="primary" if is_active else "secondary",
                use_container_width=True,
            ):
                st.session_state.selected_page = item["page"]
                st.rerun()


def render_document_upload():
    st.subheader("Belge ekle")
    uploaded_files = st.file_uploader(
        "PDF, DOCX veya Markdown dosyası seçin",
        type=["pdf", "docx", "md"],
        accept_multiple_files=True,
    )

    actions = st.columns([1, 1, 4])
    with actions[0]:
        if st.button("Belgeleri hazırla", type="primary", use_container_width=True):
            index_uploaded_files(uploaded_files)
    with actions[1]:
        if st.button("Listeyi yenile", use_container_width=True):
            refresh_document_list()


def render_document_list():
    st.subheader("İndekslenen belgeler")

    if not st.session_state.indexed_documents:
        st.info("Henüz indekslenen belge yok.")
        return

    header = st.columns([4, 1, 1, 1])
    header[0].markdown("**Dosya**")
    header[1].markdown("**Tür**")
    header[2].markdown("**Parça**")
    header[3].markdown("**İşlem**")

    for document_hash, document in st.session_state.indexed_documents.items():
        row = st.columns([4, 1, 1, 1])
        row[0].write(document.get("file_name", "Bilinmeyen dosya"))
        row[1].write(document.get("document_type", "-"))
        row[2].write(document.get("chunk_count", 0))
        if row[3].button(
            "Sil",
            key=f"delete_{document_hash}",
            use_container_width=True,
        ):
            delete_indexed_document(document_hash)
            st.rerun()


def render_documents_page():
    st.title("Belgeler")
    st.caption("RAG indeksine belge ekleyin, mevcut belgeleri görüntüleyin ve gerekmeyenleri kaldırın.")
    notice = st.session_state.get("document_notice")
    if notice:
        if notice["type"] == "success":
            st.success(notice["message"])
        else:
            st.info(notice["message"])
        st.session_state.document_notice = None
    render_document_upload()
    st.divider()
    render_document_list()


def save_profile(display_name, role, seniority, password="", profile_id=None):
    store = st.session_state.get("profile_store", DEFAULT_PROFILE_STORE.copy())
    profiles = store.get("profiles", [])
    existing_profile = find_profile(profile_id)

    if existing_profile and is_mock_profile(existing_profile):
        raise ValueError("Bu profil düzenlenemez.")

    if not str(display_name).strip():
        raise ValueError("Profil için ad soyad bilgisi zorunludur.")

    existing_password_hash = ""
    if existing_profile:
        existing_password_hash = existing_profile.get("password_hash", "")

    if not password and not existing_password_hash:
        raise ValueError("Profil oluşturmak için şifre belirlenmesi zorunludur.")

    password_hash = hash_password(password) if password else existing_password_hash
    normalized_profile = normalize_profile(
        {
            "id": profile_id or uuid4().hex,
            "display_name": display_name,
            "role": role,
            "seniority": seniority,
            "password_hash": password_hash,
            "is_mock": False,
        }
    )

    updated_profiles = []
    profile_found = False
    for profile in profiles:
        if profile.get("id") == normalized_profile["id"]:
            updated_profiles.append(normalized_profile)
            profile_found = True
        else:
            updated_profiles.append(profile)

    if not profile_found:
        updated_profiles.append(normalized_profile)

    save_profile_store(
        {
            "active_profile_id": normalized_profile["id"],
            "profiles": updated_profiles,
        }
    )
    return normalized_profile


def activate_profile(profile_id, password=None):
    store = st.session_state.get("profile_store", DEFAULT_PROFILE_STORE.copy())
    if profile_id is not None:
        profile = find_profile(profile_id)
        if not can_activate_profile(profile, password):
            return False

    save_profile_store(
        {
            "active_profile_id": profile_id,
            "profiles": store.get("profiles", []),
        }
    )
    return True


def delete_profile(profile_id):
    store = st.session_state.get("profile_store", DEFAULT_PROFILE_STORE.copy())
    profile = find_profile(profile_id)
    if profile and is_mock_profile(profile):
        raise ValueError("Bu profil silinemez.")

    profiles = [
        profile
        for profile in store.get("profiles", [])
        if profile.get("id") != profile_id
    ]
    active_profile_id = store.get("active_profile_id")
    if active_profile_id == profile_id:
        active_profile_id = None

    save_profile_store(
        {
            "active_profile_id": active_profile_id,
            "profiles": profiles,
        }
    )


def render_profile_page():
    st.title("Profilim")
    st.caption(
        "Profil bilgilerinizi yönetin ve yanıtların ihtiyaçlarınıza uygun şekilde hazırlanmasını sağlayın."
    )

    notice = st.session_state.get("profile_notice")
    if notice:
        if notice["type"] == "success":
            st.success(notice["message"])
        else:
            st.info(notice["message"])
        st.session_state.profile_notice = None

    profile = primary_user_profile()
    if profile and st.session_state.get("profile_store", {}).get("active_profile_id") != profile["id"]:
        set_active_profile(profile["id"])

    if profile:
        render_profile_summary(profile)
    else:
        profiles = visible_profiles(
            st.session_state.get("profile_store", DEFAULT_PROFILE_STORE.copy()).get(
                "profiles",
                [],
            )
        )
        if profiles:
            render_profile_login(profiles)
            if st.button("Yeni profil oluştur", use_container_width=True):
                open_new_profile_form()
                st.rerun()
        else:
            st.info("Henüz bir profil oluşturulmadı.")
            open_new_profile_form()

    if st.session_state.profile_form_open:
        st.divider()
        render_profile_form(profile)

    st.divider()
    


def render_profile_summary(profile):
    display_name = html.escape(profile.get("display_name") or "İsimsiz profil")
    role = html.escape(role_label(profile.get("role")))
    seniority = html.escape(seniority_label(profile.get("seniority", "Genel")))
    initials = html.escape(profile_initials(profile.get("display_name", "")))

    st.markdown(
        f"""
        <style>
        .profile-card {{
            max-width: 760px;
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 14px;
            overflow: hidden;
            background: #171a1f;
            color: #f4f4f2;
            box-shadow: 0 18px 45px rgba(0, 0, 0, 0.32);
            margin: 1.25rem 0 1rem;
        }}
        .profile-card__cover {{
            height: 130px;
            background:
                linear-gradient(135deg, rgba(35, 38, 42, 0.98), rgba(11, 62, 59, 0.94)),
                radial-gradient(circle at 18% 18%, rgba(255, 255, 255, 0.10), transparent 34%);
        }}
        .profile-card__body {{
            display: flex;
            gap: 1.4rem;
            padding: 0 2rem 2rem;
        }}
        .profile-card__avatar {{
            width: 104px;
            height: 104px;
            display: flex;
            align-items: center;
            justify-content: center;
            flex: 0 0 auto;
            margin-top: -52px;
            border: 5px solid #171a1f;
            border-radius: 16px;
            background: #2f3a40;
            color: #f4f4f2;
            font-size: 2rem;
            font-weight: 800;
            letter-spacing: 0;
        }}
        .profile-card__content {{
            min-width: 0;
            padding-top: 1.25rem;
        }}
        .profile-card__name {{
            font-size: 1.65rem;
            line-height: 1.2;
            font-weight: 800;
            margin-bottom: 0.35rem;
        }}
        .profile-card__role {{
            color: #b7bfbd;
            font-size: 1rem;
            font-weight: 600;
            margin-bottom: 1rem;
        }}
        .profile-card__meta {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.75rem;
        }}
        .profile-card__meta span {{
            display: inline-flex;
            align-items: center;
            min-height: 2rem;
            padding: 0.35rem 0.7rem;
            border-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.10);
            background: #24282d;
            color: #e8ecea;
            font-weight: 650;
        }}
        @media (max-width: 640px) {{
            .profile-card__body {{
                display: block;
                padding: 0 1.25rem 1.5rem;
            }}
            .profile-card__avatar {{
                width: 88px;
                height: 88px;
                margin-top: -44px;
                font-size: 1.55rem;
            }}
            .profile-card__name {{
                font-size: 1.35rem;
            }}
        }}
        </style>
        <div class="profile-card">
            <div class="profile-card__cover"></div>
            <div class="profile-card__body">
                <div class="profile-card__avatar">{initials}</div>
                <div class="profile-card__content">
                    <div class="profile-card__name">{display_name}</div>
                    <div class="profile-card__role">{role}</div>
                    <div class="profile-card__meta">
                        <span>Kıdem: {seniority}</span>
                        <span>Durum: Aktif</span>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    actions = st.columns([1, 1, 1, 3])
    with actions[0]:
        if st.button("Profili güncelle", type="primary", use_container_width=True):
            st.session_state.editing_profile_id = profile["id"]
            st.session_state.profile_form_open = True
            st.session_state.confirm_delete_profile_id = None
            st.rerun()
    with actions[1]:
        if st.button("Çıkış yap", use_container_width=True):
            logout_profile()
            st.session_state.profile_notice = {
                "type": "info",
                "message": "Oturum kapatıldı.",
            }
            st.rerun()
    with actions[2]:
        if st.button("Profili sil", use_container_width=True):
            st.session_state.confirm_delete_profile_id = profile["id"]
            st.session_state.profile_form_open = False
            st.rerun()

    if st.session_state.confirm_delete_profile_id == profile["id"]:
        st.warning(
            "Profilinizi silmek üzeresiniz. Bu işlem geri alınamaz ve profil bilgileriniz kaldırılır."
        )
        confirm_actions = st.columns([1, 1, 4])
        with confirm_actions[0]:
            delete_confirmed = st.button(
                "Evet, profili sil",
                type="primary",
                use_container_width=True,
            )
        with confirm_actions[1]:
            delete_cancelled = st.button("Vazgeç", use_container_width=True)

        if delete_cancelled:
            st.session_state.confirm_delete_profile_id = None
            st.rerun()

        if delete_confirmed:
            try:
                delete_profile(profile["id"])
                st.session_state.editing_profile_id = None
                st.session_state.profile_form_open = True
                st.session_state.confirm_delete_profile_id = None
                st.session_state.profile_logged_out = False
                st.session_state.profile_notice = {
                    "type": "info",
                    "message": "Profil silindi.",
                }
            except ValueError as exc:
                st.session_state.profile_notice = {
                    "type": "info",
                    "message": str(exc),
                }
            st.rerun()


def render_profile_login(profiles):
    st.info("Profil bilgilerinizi görüntülemek için lütfen şifrenizle giriş yapın.")

    with st.form("profile_login_form"):
        display_name = st.text_input(
            "Ad soyad",
            max_chars=80,
        )
        password = st.text_input(
            "Şifrenizi girin",
            type="password",
            max_chars=120,
        )
        submitted = st.form_submit_button("Giriş yap", type="primary")

    if submitted:
        profile = authenticate_profile(display_name, password, profiles)
        if profile:
            set_active_profile(profile["id"])
            st.session_state.profile_logged_out = False
            st.session_state.profile_form_open = False
            st.session_state.editing_profile_id = None
            st.session_state.confirm_delete_profile_id = None
            st.session_state.profile_notice = {
                "type": "success",
                "message": "Giriş yapıldı.",
            }
        else:
            st.session_state.profile_notice = {
                "type": "info",
                "message": "Ad soyad veya şifre hatalı.",
            }
        st.rerun()


def render_profile_form(profile):
    editing_profile = find_profile(st.session_state.editing_profile_id)
    if not editing_profile:
        editing_profile = profile or EMPTY_PROFILE_FORM

    update_existing_profile = bool(profile and editing_profile.get("id") == profile["id"])
    seniority_index = (
        SENIORITY_OPTIONS.index(editing_profile["seniority"])
        if editing_profile.get("seniority") in SENIORITY_OPTIONS
        else SENIORITY_OPTIONS.index(EMPTY_PROFILE_FORM["seniority"])
    )
    role_options = role_options_for_current_value(editing_profile.get("role"))
    role_index = (
        role_options.index(editing_profile["role"])
        if editing_profile.get("role") in role_options
        else role_options.index(EMPTY_PROFILE_FORM["role"])
    )

    st.subheader("Profil bilgileri")
    with st.form("profile_form"):
        display_name = st.text_input(
            "Ad soyad",
            value=editing_profile.get("display_name", ""),
            max_chars=80,
        )
        role = st.selectbox(
            "Rol / ekip",
            options=role_options,
            index=role_index,
            format_func=role_label,
        )
        seniority = st.selectbox(
            "Kıdem seviyesi",
            options=SENIORITY_OPTIONS,
            index=seniority_index,
            format_func=seniority_label,
        )
        password_label = (
            "Yeni şifrenizi girin"
            if update_existing_profile
            else "Şifrenizi belirleyin"
        )
        password = st.text_input(
            password_label,
            type="password",
            max_chars=120,
        )

        form_actions = st.columns([1, 1, 4])
        with form_actions[0]:
            submitted = st.form_submit_button(
                "Kaydet" if update_existing_profile else "Profil oluştur",
                type="primary",
                use_container_width=True,
            )
        with form_actions[1]:
            cancelled = st.form_submit_button(
                "Vazgeç",
                use_container_width=True,
            )

    if cancelled:
        st.session_state.profile_form_open = False
        st.session_state.editing_profile_id = None
        st.session_state.confirm_delete_profile_id = None
        st.rerun()

    if submitted:
        try:
            saved_profile = save_profile(
                display_name=display_name,
                role=role,
                seniority=seniority,
                password=password,
                profile_id=editing_profile.get("id") if update_existing_profile else None,
            )
            st.session_state.editing_profile_id = None
            st.session_state.profile_form_open = False
            st.session_state.confirm_delete_profile_id = None
            st.session_state.profile_logged_out = False
            st.session_state.profile_notice = {
                "type": "success",
                "message": f"{profile_label(saved_profile)} başarıyla kaydedildi.",
            }
        except ValueError as exc:
            st.session_state.profile_notice = {
                "type": "info",
                "message": str(exc),
            }
        st.rerun()


def render_chat_page():
    st.title("Kurumsal Doküman Asistanı")
    st.caption("Kaynaklı yanıt üreten RAG tabanlı doküman sohbeti.")
    render_chat_settings()
    st.divider()

    render_chat()

    query = st.chat_input("Dokümanlar hakkında soru sorun")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
        render_assistant_response(query)


def render_assistant_response(query):
    with st.chat_message("assistant"):
        try:
            answer, citations = answer_question(query)
            st.markdown(answer)
            render_citations(citations, expanded=True)
        except Exception as exc:
            answer = str(exc)
            citations = []
            st.error(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "citations": citations,
        }
    )


def main():
    st.set_page_config(page_title="Kurumsal Doküman Asistanı", layout="wide")
    initialize_session_state()
    load_indexed_documents_on_startup()

    render_sidebar()

    if selected_page() == "Belgeler":
        render_documents_page()
    elif selected_page() == "Profil":
        render_profile_page()
    else:
        render_chat_page()


if __name__ == "__main__":
    main()
