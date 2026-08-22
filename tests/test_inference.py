from vlm.inference import vlm_answer, SatQueryVLM
from vlm.schemas import VLMRequest

def test_satquery_vlm_generate():
    model = SatQueryVLM(model_name="dummy")
    req = VLMRequest(images=["dummy.jpg"], question="Test")
    res = model.generate(req)
    assert "Dummy response" in res.answer
    assert res.confidence == 0.99

def test_vlm_answer_global():
    res = vlm_answer(images=["dummy.jpg"], question="Test Q")
    assert "Test Q" in res["answer"]
    assert res["confidence"] == 0.99
