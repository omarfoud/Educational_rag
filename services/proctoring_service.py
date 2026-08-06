"""
Lightweight CV proctoring pipeline.

The browser sends sampled frames and optional client-side signals. The service
stores suspicious metadata only, not the video itself.
"""
from __future__ import annotations

import base64
import io
import uuid
from collections import Counter
from typing import Any

from services.database_service import database_service


HIGH_RISK_EVENTS = {"multi_face", "no_face", "object_detected"}
MEDIUM_RISK_EVENTS = {"gaze_away"}
SUSPICIOUS_OBJECTS = {
    "cell phone",
    "mobile phone",
    "phone",
    "book",
    "paper",
    "laptop",
    "tablet",
}


class ProctoringService:
    def __init__(self):
        self._cv2 = None
        self._face_cascade = None
        self._cv2_checked = False

    async def analyze_frame(self, payload: dict[str, Any]) -> dict[str, Any]:
        session_id = str(payload.get("sessionId") or payload.get("session_id") or "")
        student_id = str(payload.get("studentId") or payload.get("student_id") or "")
        if not session_id or not student_id:
            raise ValueError("sessionId and studentId are required")

        detected = []
        detected.extend(self._events_from_head_pose(payload.get("headPose") or payload.get("head_pose") or {}))
        detected.extend(self._events_from_objects(payload.get("objects") or []))
        detected.extend(self._events_from_audio(payload.get("audioEnergy") or payload.get("audio_energy")))

        image = self._decode_image(payload.get("image"))
        if image is not None:
            detected.extend(self._events_from_image(image))

        saved_events = [
            database_service.save_proctoring_event(
                event_id=str(uuid.uuid4()),
                session_id=session_id,
                student_id=student_id,
                event_type=event["eventType"],
                confidence=event["confidence"],
                details=event.get("details") or {},
            )
            for event in detected
        ]

        return {
            "sessionId": session_id,
            "events": saved_events,
            "status": "ok",
            "analyzer": "opencv" if self._get_cv2()[0] is not None else "heuristic",
        }

    def build_report(self, session_id: str) -> dict[str, Any]:
        events = database_service.get_proctoring_events(session_id)
        counts = Counter(event["eventType"] for event in events)
        score = 0.0
        for event in events:
            event_type = event["eventType"]
            confidence = float(event.get("confidence") or 0.0)
            if event_type in HIGH_RISK_EVENTS:
                score += 18.0 * confidence
            elif event_type in MEDIUM_RISK_EVENTS:
                score += 10.0 * confidence
            else:
                score += 5.0 * confidence
        risk_score = min(round(score, 2), 100.0)
        if risk_score >= 60:
            risk_level = "high"
        elif risk_score >= 25:
            risk_level = "medium"
        else:
            risk_level = "low"

        return {
            "sessionId": session_id,
            "totalEvents": len(events),
            "riskScore": risk_score,
            "riskLevel": risk_level,
            "eventCounts": dict(counts),
            "events": events,
        }

    def _events_from_head_pose(self, head_pose: dict[str, Any]) -> list[dict[str, Any]]:
        if not head_pose:
            return []
        yaw = abs(float(head_pose.get("yaw") or 0.0))
        pitch = abs(float(head_pose.get("pitch") or 0.0))
        explicit = bool(head_pose.get("gazeAway") or head_pose.get("gaze_away"))
        if explicit or yaw >= 28 or pitch >= 22:
            confidence = min(max((max(yaw / 45, pitch / 35, 0.65)), 0.0), 1.0)
            return [{
                "eventType": "gaze_away",
                "confidence": confidence,
                "details": {"yaw": yaw, "pitch": pitch},
            }]
        return []

    def _events_from_objects(self, objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
        events = []
        for obj in objects:
            label = str(obj.get("label") or obj.get("name") or "").strip().lower()
            confidence = float(obj.get("confidence") or obj.get("score") or 0.0)
            if label in SUSPICIOUS_OBJECTS and confidence >= 0.45:
                events.append({
                    "eventType": "object_detected",
                    "confidence": min(confidence, 1.0),
                    "details": {"label": label},
                })
        return events

    def _events_from_audio(self, audio_energy: Any) -> list[dict[str, Any]]:
        if audio_energy is None:
            return []
        energy = float(audio_energy)
        if energy >= 0.85:
            return [{
                "eventType": "abnormal_audio",
                "confidence": min(energy, 1.0),
                "details": {"audioEnergy": energy},
            }]
        return []

    def _events_from_image(self, image) -> list[dict[str, Any]]:
        cv2, cascade = self._get_cv2()
        if cv2 is None or cascade is None:
            return []

        import numpy as np

        frame = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
        faces = cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(45, 45))
        count = len(faces)
        if count == 0:
            return [{"eventType": "no_face", "confidence": 0.75, "details": {"faceCount": 0}}]
        if count > 1:
            return [{"eventType": "multi_face", "confidence": 0.9, "details": {"faceCount": count}}]
        return []

    def _decode_image(self, image_data: Any):
        if not image_data:
            return None
        from PIL import Image

        if isinstance(image_data, str):
            payload = image_data.split(",", 1)[1] if "," in image_data else image_data
            raw = base64.b64decode(payload)
        elif isinstance(image_data, bytes):
            raw = image_data
        else:
            return None
        return Image.open(io.BytesIO(raw))

    def _get_cv2(self):
        if self._cv2_checked:
            return self._cv2, self._face_cascade
        self._cv2_checked = True
        try:
            import cv2

            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            cascade = cv2.CascadeClassifier(cascade_path)
            if cascade.empty():
                return None, None
            self._cv2 = cv2
            self._face_cascade = cascade
        except Exception:
            self._cv2 = None
            self._face_cascade = None
        return self._cv2, self._face_cascade


proctoring_service = ProctoringService()
