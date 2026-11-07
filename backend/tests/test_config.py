from config import Settings, get_settings


def test_defaults_load():
    s = get_settings()
    assert s.embedding_model == "all-MiniLM-L6-v2"
    assert s.chunk_overlap < s.chunk_size
    assert s.top_k >= 1


def test_overrides_do_not_mutate_original():
    s = get_settings()
    t = s.with_overrides(top_k=9)
    assert t.top_k == 9
    assert s.top_k == get_settings().top_k  # original untouched (frozen dataclass copy)


def test_public_dict_hides_paths_and_urls():
    d = get_settings().public_dict()
    blob = str(d)
    assert "/" not in blob.replace("all-MiniLM-L6-v2", "")
    assert "ollama_base_url" not in d
    assert "data_dir" not in d


def test_relative_sqlite_resolves_to_absolute(tmp_path):
    s = Settings().with_overrides(database_url="sqlite:///data/x.db")
    assert s.sqlalchemy_url().startswith("sqlite:////")
