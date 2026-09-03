import asyncio
from types import SimpleNamespace

import pytest

from models.schemas import QuizQuestion, QuizOption, GenerateQuizRequest, GenerateQuestionsRequest
from services.question_quality_service import QuestionQualityService, QuestionQualityError, arithmetic_value
from services.question_service import QuestionService
from config.settings import settings


def mcq(stem, choices, correct=0, explanation="Generator answer must not reach the blind solver"):
    return QuizQuestion(question=stem, options=[QuizOption(text=text, isCorrect=i == correct) for i, text in enumerate(choices)], explanation=explanation)


def reported_questions():
    return [
        mcq("لدالة من الشكل y=ax^2 إذا كانت قيمتها عند x=2 تساوي 12 وعند x=-1 تساوي 1. أوجد قيمة a.", ["3", "2", "1", "4"]),
        mcq("للدالة f(x)=x^2-4x. احسب قيمة f(2).", ["-4", "0", "-? ", "-? "]),
        mcq("إذا كانت الدالة f(x)=3x-5، فما قيمة x التي تحقق f(x)=7؟", ["4", "3", "2", "5"], 1),
        mcq("اوجد تقاطع الدالة y=x^2-9 مع محور السينات (أي عند y=0).", ["x=3 و x=-3", "x=9 و x=-9", "x=0 فقط", "x=±√9 فقط"]),
        mcq("لدينا الدالة: f(x)=2x+3. احسب f(-4).", ["-5", "-1", "5", "11"], 1),
    ]


def solve(index, answer="4", expression=None, valid=True):
    return {"index": index, "valid": valid, "answer": answer, "explanation": "Independent verified solution", "reason": "Contradictory premises" if not valid else "", "numeric_expression": expression}


def verdict(index, selected=0, valid=True):
    return {"index": index, "valid": valid, "correct_option": selected, "reason": "Equivalent correct options" if not valid else ""}


def test_reported_five_questions_reject_three_and_correct_two_without_answer_leakage():
    gate = QuestionQualityService()
    calls = []

    async def review(stage, payload, *args):
        calls.append((stage, payload))
        if stage == "solve":
            assert {q["index"] for q in payload} == {0, 2, 3, 4}
            assert all(set(q) == {"index", "question", "type"} for q in payload)
            return {0: solve(0, valid=False), 2: solve(2, "4", "(7+5)/3"),
                    3: solve(3, "x=3 or x=-3"), 4: solve(4, "-5", "2*(-4)+3")}
        assert all("isCorrect" not in str(q) for q in payload)
        return {2: verdict(2), 3: verdict(3, valid=False), 4: verdict(4)}

    gate._review = review
    approved, rejected = asyncio.run(gate.review_batch(reported_questions(), [], "Arabic", True))
    assert set(rejected) == {0, 1, 3}
    assert [index for index, q in approved] == [2, 4]
    assert [next(o.text for o in q.options if o.isCorrect) for _, q in approved] == ["4", "-5"]
    assert all(q.explanation == "Independent verified solution" for _, q in approved)


@pytest.mark.parametrize("expression,answer", [("(7+5)/3", 4), ("2*(-4)+3", -5), ("2**2-4*2", -4), ("0.1+0.2", "3/10")])
def test_exact_arithmetic(expression, answer):
    assert str(arithmetic_value(expression)) == str(answer)


@pytest.mark.parametrize("expression", ["__import__('os').system('whoami')", "x+1", "2**100000", "1/0", "True", "[1,2]", "9**9**9"])
def test_arithmetic_rejects_unsafe_or_unbounded_expressions(expression):
    with pytest.raises((ValueError, ArithmeticError)):
        arithmetic_value(expression)


def test_numeric_check_overrides_judge_wrong_flag_and_rejects_missing_answer():
    gate = QuestionQualityService()
    async def review(stage, payload, *args):
        if stage == "solve":
            return {0: solve(0, "4", "(7+5)/3"), 1: solve(1, "25", "(7-2)*5")}
        return {0: verdict(0, 1), 1: verdict(1, 0)}
    gate._review = review
    approved, errors = asyncio.run(gate.review_batch([
        reported_questions()[2], mcq("س/5 + 2 = 7", ["10", "20", "5", "15"]),
    ], [], "Arabic", True))
    assert set(errors) == {1}
    assert approved[0][1].options[0].isCorrect


def test_replacements_only_for_rejected_questions_and_reviewed_again():
    gate = QuestionQualityService()
    good = mcq("3x=12", ["4", "3", "2", "1"])
    bad = mcq("f(2)?", ["-4", "0", "-?", "-?"])
    replacement = mcq("2x=8", ["4", "5", "6", "7"])
    reviewed = []
    async def review(stage, payload, *args):
        reviewed.append((stage, [q["question"] for q in payload]))
        return {q["index"]: solve(q["index"], "4", "4") if stage == "solve" else verdict(q["index"]) for q in payload}
    async def regenerate(count, errors, accepted):
        assert count == 1
        assert errors[0]["question"] == "f(2)?"
        assert accepted == [good.question]
        return [replacement]
    gate._review = review
    result = asyncio.run(gate.ensure([good, bad], 2, regenerate, [], "English", True))
    assert [q.question for q in result] == [good.question, replacement.question]
    assert ("solve", [replacement.question]) in reviewed


def test_exhaustion_returns_error_not_partial_or_unreviewed_questions():
    gate = QuestionQualityService()
    bad = mcq("f(2)?", ["-4", "0", "-?", "-?"])
    regenerations = []
    async def regenerate(*args):
        regenerations.append(args)
        return [bad]
    with pytest.raises(QuestionQualityError):
        asyncio.run(gate.ensure([bad], 1, regenerate, [], "Arabic", True))
    assert len(regenerations) == 2


def test_judge_outage_fails_closed():
    gate = QuestionQualityService()
    async def unavailable(**kwargs):
        raise TimeoutError("provider timeout")
    gate._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=unavailable)))
    with pytest.raises(QuestionQualityError, match="unavailable"):
        asyncio.run(gate.review_batch([reported_questions()[2]], [], "Arabic", True))


@pytest.mark.parametrize("text", ['{"items":[]}', '{"items":[{"index":99}]}', 'not-json'])
def test_malformed_judge_response_fails_closed(text):
    gate = QuestionQualityService()
    async def response(**kwargs):
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=text))])
    gate._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=response)))
    with pytest.raises(QuestionQualityError):
        asyncio.run(gate.review_batch([reported_questions()[2]], [], "Arabic", True))


@pytest.mark.parametrize("endpoint", ["quiz", "questions"])
def test_both_generation_paths_require_quality_gate(endpoint):
    service = QuestionService()
    called = []
    async def blocked(*args, **kwargs):
        called.append(True)
        raise QuestionQualityError("verification blocked")
    async def context(*args, **kwargs):
        return [{"text": "معادلات خطية", "metadata": {"language": "ar"}}]
    async def generation(*args, **kwargs):
        return []
    service._verify_question_set = blocked
    service._structured = generation
    service._get_quiz_context = context
    service._resolve_question_file_ids = lambda request: []
    service._retrieve_context_for_file_ids = context
    service.rag = SimpleNamespace(generate_structured_output=generation)
    with pytest.raises(QuestionQualityError):
        if endpoint == "quiz":
            asyncio.run(service.generate_quiz(GenerateQuizRequest(subject="رياضيات", numberOfQuestions=1)))
        else:
            asyncio.run(service.generate_questions(GenerateQuestionsRequest(questionsNumber=1)))
    assert called == [True]


def test_review_deadline_fails_closed(monkeypatch):
    gate = QuestionQualityService()
    monkeypatch.setattr(settings, "question_review_timeout_seconds", 0.01)
    async def slow(*args):
        await asyncio.sleep(1)
    gate.review_batch = slow
    with pytest.raises(QuestionQualityError, match="timed out"):
        asyncio.run(gate.ensure([reported_questions()[2]], 1, slow, [], "Arabic", True))


def test_review_chunks_large_sets_without_losing_indices():
    gate = QuestionQualityService()
    sizes = []
    async def review(stage, payload, *args):
        sizes.append(len(payload))
        return {q["index"]: solve(q["index"], "4", "4") if stage == "solve" else verdict(q["index"]) for q in payload}
    gate._review = review
    questions = [mcq(f"Question {i}: 2+2?", ["4", "3", "2", "1"]) for i in range(12)]
    approved, errors = asyncio.run(gate.review_batch(questions, [], "English", True))
    assert not errors
    assert [i for i, _ in approved] == list(range(12))
    assert sizes == [10, 10, 2, 2]


@pytest.mark.parametrize("question_type,answer", [("true_false", "false"), ("short_answer", "Independent answer")])
def test_non_mcq_answers_are_replaced_by_blind_solution(question_type, answer):
    from models.schemas import GeneratedQuestion
    gate = QuestionQualityService()
    async def review(stage, payload, *args):
        assert stage == "solve"
        assert "correctAnswer" not in payload[0]
        return {0: solve(0, answer)}
    gate._review = review
    question = GeneratedQuestion(question="Test question", type=question_type, difficulty="easy", correctAnswer="wrong")
    approved, errors = asyncio.run(gate.review_batch([question], [], "English", False))
    assert not errors
    assert approved[0][1].correctAnswer == answer


@pytest.mark.parametrize("endpoint", ["quiz", "questions"])
def test_api_reports_quality_failure_as_502(monkeypatch, endpoint):
    import main
    from fastapi import HTTPException
    async def blocked(*args, **kwargs):
        raise QuestionQualityError("Unable to verify")
    monkeypatch.setattr(main.question_service, "generate_quiz" if endpoint == "quiz" else "generate_questions", blocked)
    with pytest.raises(HTTPException) as failure:
        if endpoint == "quiz":
            asyncio.run(main.generate_quiz(GenerateQuizRequest(subject="Mathematics")))
        else:
            asyncio.run(main.generate_questions(GenerateQuestionsRequest()))
    assert failure.value.status_code == 502
