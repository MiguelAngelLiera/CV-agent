"""
Cubre los casos de prueba pedidos por el reto:
* pregunta directa
* pregunta sin evidencia en el CV
* input vacío/demasiado largo
* auth ausente/inválida
* conversación multi-turno.
"""

import os
import sys
import types
from unittest.mock import MagicMock
import json

os.environ["AGENT_API_KEY"] = "test-key"
os.environ["GEMINI_API_KEY"] = "fake-gemini-key"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient
from google.genai import errors as genai_errors

from app import main

client = TestClient(main.app)
HEADERS = {"Authorization": "Bearer test-key"}


def _mock_completion(text="Respuesta simulada del agente."):
    usage = types.SimpleNamespace(
        prompt_token_count=50, candidates_token_count=20, total_token_count=70
    )
    return types.SimpleNamespace(text=text, usage_metadata=usage)


def test_missing_api_key_rejected():
    r = client.post("/v1/responses", json={"input": "Hola"})
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "missing_api_key"


def test_invalid_api_key_rejected():
    r = client.post(
        "/v1/responses",
        json={"input": "Hola"},
        headers={"Authorization": "Bearer wrong"},
    )
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "invalid_api_key"


def test_empty_input_rejected():
    r = client.post("/v1/responses", json={"input": ""}, headers=HEADERS)
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "empty_input"


def test_oversized_input_rejected():
    huge = "a" * (main.MAX_INPUT_CHARS + 1)
    r = client.post("/v1/responses", json={"input": huge}, headers=HEADERS)
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "input_too_long"


def test_direct_question_returns_grounded_answer(monkeypatch):
    main.client.models.generate_content = MagicMock(
        return_value=_mock_completion("Tengo experiencia con RAG en BBVA.")
    )
    r = client.post(
        "/v1/responses",
        json={"input": "¿Qué experiencia tienes con RAG?"},
        headers=HEADERS,
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "completed"
    assert body["output"][0]["content"][0]["type"] == "output_text"
    assert "id" in body and body["id"].startswith("resp_")


def test_unanswerable_question_is_handled_gracefully(monkeypatch):
    main.client.models.generate_content = MagicMock(
        return_value=_mock_completion(
            "Eso no está en la información que tengo del perfil de Miguel."
        )
    )
    r = client.post(
        "/v1/responses", json={"input": "¿Cuál es tu color favorito?"}, headers=HEADERS
    )
    assert r.status_code == 200
    assert "no está" in r.json()["output"][0]["content"][0]["text"].lower()


def test_multi_turn_conversation_uses_previous_response_id(monkeypatch):
    main.client.models.generate_content = MagicMock(
        return_value=_mock_completion("Primera respuesta.")
    )
    r1 = client.post(
        "/v1/responses", json={"input": "Cuéntame de tu experiencia."}, headers=HEADERS
    )
    resp_id = r1.json()["id"]

    captured = {}

    def fake_generate(**kwargs):
        captured["contents"] = list(kwargs["contents"])  # snapshot
        return _mock_completion("Ese proyecto usó LangGraph.")

    main.client.models.generate_content = fake_generate
    r2 = client.post(
        "/v1/responses",
        json={"input": "¿Y qué framework usaste ahí?", "previous_response_id": resp_id},
        headers=HEADERS,
    )
    assert r2.status_code == 200
    assert len(captured["contents"]) == 3 


def test_unknown_previous_response_id_rejected():
    r = client.post(
        "/v1/responses",
        json={"input": "hola", "previous_response_id": "resp_does_not_exist"},
        headers=HEADERS,
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "unknown_previous_response_id"


def test_unexpected_internal_error_returns_500(monkeypatch):
    def raise_generic_error(*args, **kwargs):
        raise RuntimeError("algo inesperado sin código de proveedor")

    main.client.models.generate_content = raise_generic_error
    r = client.post("/v1/responses", json={"input": "hola"}, headers=HEADERS)
    assert r.status_code == 500
    assert r.json()["error"]["code"] == "internal_error"


def test_input_as_message_list_with_parts_array(monkeypatch):
    """Cubre el formato que manda la plataforma: content como lista
    de partes [{"type":"input_text","text":"..."}] en vez de un string plano."""
    main.client.models.generate_content = MagicMock(return_value=_mock_completion("Respuesta con contenido estructurado."))
    r = client.post(
        "/v1/responses",
        json={"input": [{"role": "user", "content": [{"type": "input_text", "text": "Hola, cuéntame de ti"}]}]},
        headers=HEADERS,
    )
    assert r.status_code == 200
    assert r.json()["status"] == "completed"


def test_malformed_input_returns_open_responses_error_shape():
    r = client.post("/v1/responses", json={"input": [{"role": "not-a-valid-role", "content": "hola"}]}, headers=HEADERS)
    assert r.status_code == 400
    body = r.json()
    assert "error" in body and "detail" not in body
    assert body["error"]["code"] == "validation_error"


def test_streaming_events():
    """Cada línea 'data:' debe traer su propio campo 'type' — algunos
    clientes (incluida la plataforma del reto) lo leen del payload, no solo
    del nombre de evento SSE."""
    main.client.models.generate_content = MagicMock(return_value=_mock_completion("Hola, esta es una respuesta de prueba."))
    with client.stream(
        "POST",
        "/v1/responses",
        json={"input": "Cuéntame sobre tu experiencia", "stream": True},
        headers=HEADERS,
    ) as r:
        assert r.status_code == 200
        raw = "".join(r.iter_text())

    data_lines = [line.removeprefix("data: ") for line in raw.splitlines() if line.startswith("data: ")]
    assert len(data_lines) >= 3 
    types_seen = set()
    for line in data_lines:
        payload = json.loads(line)
        assert "type" in payload, f"evento sin 'type': {payload}"
        types_seen.add(payload["type"])
    assert types_seen == {"response.created", "response.output_text.delta", "response.completed"}
