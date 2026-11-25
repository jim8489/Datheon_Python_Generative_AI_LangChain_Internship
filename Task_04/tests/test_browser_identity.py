from core.browser_identity import (
    create_browser_token,
    get_or_create_browser_id,
    load_signing_key,
    verify_browser_token,
)


class FakeCookieController:
    def __init__(self, initial=None):
        self.cookies = dict(initial or {})
        self.set_calls = []

    def get(self, name):
        return self.cookies.get(name)

    def set(self, name, value, **options):
        self.cookies[name] = value
        self.set_calls.append((name, value, options))


def test_signed_browser_token_round_trips():
    key = b"k" * 32
    browser_id = "a" * 32
    token = create_browser_token(browser_id, key, issued_at=1_700_000_000)

    assert verify_browser_token(
        token, key, now=1_700_000_100, ttl_seconds=1_000
    ) == browser_id


def test_tampered_or_wrong_key_token_is_rejected():
    token = create_browser_token("a" * 32, b"k" * 32, issued_at=1_700_000_000)

    assert verify_browser_token(token + "x", b"k" * 32, now=1_700_000_100) is None
    assert verify_browser_token(token, b"z" * 32, now=1_700_000_100) is None


def test_expired_and_future_tokens_are_rejected():
    token = create_browser_token("b" * 32, b"k" * 32, issued_at=1_000)

    assert verify_browser_token(token, b"k" * 32, now=2_000, ttl_seconds=100) is None
    assert verify_browser_token(token, b"k" * 32, now=900, ttl_seconds=100) is None


def test_identity_is_reused_from_valid_cookie():
    key = b"k" * 32
    existing_id = "c" * 32
    token = create_browser_token(existing_id, key)
    controller = FakeCookieController({"documind_browser": token})

    browser_id = get_or_create_browser_id(
        controller, key, secure=True, cookie_name="documind_browser", ttl_days=365
    )

    assert browser_id == existing_id
    assert controller.set_calls == []


def test_invalid_cookie_is_replaced_with_signed_cookie():
    key = b"k" * 32
    controller = FakeCookieController({"documind_browser": "tampered"})

    browser_id = get_or_create_browser_id(
        controller, key, secure=False, cookie_name="documind_browser", ttl_days=365
    )

    assert len(browser_id) == 32
    int(browser_id, 16)
    assert len(controller.set_calls) == 1
    name, token, options = controller.set_calls[0]
    assert name == "documind_browser"
    assert verify_browser_token(token, key) == browser_id
    assert options["secure"] is False
    assert options["same_site"] == "strict"
    assert options["max_age"] == 365 * 24 * 60 * 60


def test_signing_key_is_stable_and_created_with_private_mode(tmp_path):
    key_path = tmp_path / "data" / "signing.key"

    first = load_signing_key(key_path=key_path, configured_key=None)
    second = load_signing_key(key_path=key_path, configured_key=None)

    assert first == second
    assert len(first) == 32
    assert key_path.exists()


def test_configured_signing_key_must_be_long_enough():
    try:
        load_signing_key(configured_key="too-short")
    except ValueError as exc:
        assert "at least 32 bytes" in str(exc)
    else:
        raise AssertionError("Short signing keys must be rejected")


def test_initial_request_cookie_is_used_when_component_cache_is_empty():
    key = b"k" * 32
    existing_id = "d" * 32
    token = create_browser_token(existing_id, key)
    controller = FakeCookieController()

    browser_id = get_or_create_browser_id(
        controller,
        key,
        secure=True,
        cookie_name="documind_browser",
        ttl_days=365,
        fallback_token=token,
    )

    assert browser_id == existing_id
    assert controller.set_calls == []
