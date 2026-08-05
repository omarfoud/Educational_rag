import importlib

import pytest

from services.proctoring_service import ProctoringService

proctoring_module = importlib.import_module("services.proctoring_service")


@pytest.mark.asyncio
async def test_proctoring_frame_logs_client_side_signals(monkeypatch):
    saved = []

    def fake_save(event_id, session_id, student_id, event_type, confidence, details=None):
        event = {
            "id": event_id,
            "sessionId": session_id,
            "studentId": student_id,
            "eventType": event_type,
            "confidence": confidence,
            "details": details or {},
            "timestamp": "2026-08-06T00:00:00",
        }
        saved.append(event)
        return event

    monkeypatch.setattr(proctoring_module.database_service, "save_proctoring_event", fake_save)

    service = ProctoringService()
    result = await service.analyze_frame({
        "sessionId": "exam-session-1",
        "studentId": "student-1",
        "headPose": {"yaw": 35, "pitch": 2},
        "objects": [{"label": "cell phone", "confidence": 0.91}],
        "audioEnergy": 0.9,
    })

    assert result["sessionId"] == "exam-session-1"
    assert [event["eventType"] for event in result["events"]] == [
        "gaze_away",
        "object_detected",
        "abnormal_audio",
    ]
    assert len(saved) == 3


def test_proctoring_report_scores_session(monkeypatch):
    monkeypatch.setattr(
        proctoring_module.database_service,
        "get_proctoring_events",
        lambda session_id: [
            {"eventType": "object_detected", "confidence": 0.9},
            {"eventType": "multi_face", "confidence": 0.8},
            {"eventType": "gaze_away", "confidence": 0.7},
        ],
    )

    report = ProctoringService().build_report("exam-session-1")

    assert report["totalEvents"] == 3
    assert report["eventCounts"]["object_detected"] == 1
    assert report["riskLevel"] in {"medium", "high"}
