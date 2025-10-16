import json

from tools.extract import ZoneAnalysis, format_user_prompt, parse_analysis_payload


def test_parse_analysis_payload_valid():
    payload = {
        "pair": "GBPUSD",
        "bias": "buy",
        "entry": 1.2675,
        "stop": 1.2630,
        "target": 1.2800,
        "zone_type": "DBR",
        "eq_alignment": "buy",
        "enhancer_score": 8.5,
        "comment": "Valid test payload",
    }
    text = json.dumps(payload)
    analysis = parse_analysis_payload(text)
    assert isinstance(analysis, ZoneAnalysis)
    assert analysis.pair == "GBPUSD"
    assert analysis.enhancer_score == 8.5


def test_format_user_prompt_strips_empty_segments():
    prompt = format_user_prompt(["primary", "", " secondary "])
    assert prompt == "primary\n\nsecondary"
