import threading
import time
from pathlib import Path
from typing import Any, cast
from urllib.request import urlretrieve

import cv2
import mediapipe as mp


ANALYSIS_INTERVAL = 30
OBJECT_INTERVAL = 8
FATIGUE_CLOSED_SECONDS = 1.5
EYE_CLOSED_THRESHOLD = 0.65
SMILE_THRESHOLD = 0.35
MODEL_DIR = Path.home() / ".mediapipe" / "reconhecimento-facial"
FACE_LANDMARKER_MODEL = MODEL_DIR / "face_landmarker.task"
FACE_LANDMARKER_URL = (
	"https://storage.googleapis.com/mediapipe-models/face_landmarker/"
	"face_landmarker/float16/1/face_landmarker.task"
)
OBJECT_MODEL = MODEL_DIR / "yolo11n.pt"
OBJECT_MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt"
OBJECT_TRANSLATIONS = {"bottle": "GARRAFA", "mouse": "MOUSE", "cell phone": "CELULAR"}
EMOTION_TRANSLATIONS = {
	"angry": "Raiva",
	"disgust": "Nojo",
	"fear": "Medo",
	"happy": "Feliz",
	"sad": "Triste",
	"surprise": "Surpresa",
	"neutral": "Neutro",
}
GENDER_TRANSLATIONS = {"man": "Homem", "woman": "Mulher"}
RACE_TRANSLATIONS = {
	"asian": "Asiática",
	"indian": "Indiana",
	"black": "Negra",
	"white": "Branca",
	"middle eastern": "Oriente Médio",
	"latino hispanic": "Latina/hispânica",
}
COLOR_PANEL = (24, 30, 32)
COLOR_TEXT = (235, 240, 238)
COLOR_MUTED = (160, 174, 170)
COLOR_WOMAN = (190, 105, 255)
COLOR_MAN = (255, 150, 55)
COLOR_UNKNOWN = (150, 160, 160)
GENDER_COLORS = {"Mulher": COLOR_WOMAN, "Homem": COLOR_MAN}
COLOR_OBJECT = (255, 215, 70)
COLOR_ALERT = (40, 40, 245)


def ensure_model(model_path, model_url):
	if model_path.is_file() and model_path.stat().st_size > 0:
		return model_path

	model_path.parent.mkdir(parents=True, exist_ok=True)
	temporary_path = model_path.with_suffix(model_path.suffix + ".part")
	try:
		print(f"Baixando modelo {model_path.name}...")
		urlretrieve(model_url, temporary_path)
		if temporary_path.stat().st_size == 0:
			raise ValueError(f"O modelo baixado está vazio: {model_path.name}")
		temporary_path.replace(model_path)
	except Exception:
		temporary_path.unlink(missing_ok=True)
		raise
	return model_path


class FatigueMonitor:
	def __init__(self, closed_seconds=FATIGUE_CLOSED_SECONDS):
		self.closed_seconds = closed_seconds
		self.closed_since = None
		self.alert_active = False

	def update(self, face_blendshapes, now):
		closed_without_smile = False
		for face in face_blendshapes:
			values = {item.category_name: item.score for item in face}
			blink_left = values.get("eyeBlinkLeft", 0.0)
			blink_right = values.get("eyeBlinkRight", 0.0)
			smile_left = values.get("mouthSmileLeft", 0.0)
			smile_right = values.get("mouthSmileRight", 0.0)
			eyes_closed = (blink_left + blink_right) / 2 >= EYE_CLOSED_THRESHOLD
			smiling = (smile_left + smile_right) / 2 >= SMILE_THRESHOLD
			if eyes_closed and not smiling:
				closed_without_smile = True
				break

		if not closed_without_smile:
			self.closed_since = None
			self.alert_active = False
			return False, False

		if self.closed_since is None:
			self.closed_since = now

		alert_started = now - self.closed_since >= self.closed_seconds and not self.alert_active
		if alert_started:
			self.alert_active = True
		return self.alert_active, alert_started


def play_fatigue_alarm():
	try:
		import winsound

		winsound.Beep(1100, 450)
	except (ImportError, RuntimeError):
		print("ALERTA: olhos fechados sem sorriso. Faça uma pausa.")


def analyze_face_attributes(roi_color):
	from deepface import DeepFace

	analysis = DeepFace.analyze(
		img_path=roi_color,
		actions=["emotion", "age", "gender", "race"],
		enforce_detection=False,
		detector_backend="skip",
	)
	if isinstance(analysis, list):
		result = cast(dict[str, Any], analysis[0])
	else:
		result = cast(dict[str, Any], analysis)
	dominant_emotion = result["dominant_emotion"]
	confidence = float(result["emotion"][dominant_emotion])
	dominant_gender = str(result["dominant_gender"])
	dominant_race = str(result["dominant_race"])
	return {
		"emotion": EMOTION_TRANSLATIONS.get(dominant_emotion, dominant_emotion),
		"emotion_confidence": confidence,
		"age": int(round(float(result["age"]))),
		"gender": GENDER_TRANSLATIONS.get(dominant_gender.lower(), dominant_gender),
		"race": RACE_TRANSLATIONS.get(dominant_race.lower(), dominant_race),
	}


def count_estimated_genders(attributes_by_face):
	counts = {"Homem": 0, "Mulher": 0}
	for attributes in attributes_by_face:
		if attributes is not None and attributes.get("gender") in counts:
			counts[attributes["gender"]] += 1
	return counts


def draw_badge(frame, text, x, y, accent, scale=0.46):
	font = cv2.FONT_HERSHEY_SIMPLEX
	(text_width, text_height), baseline = cv2.getTextSize(text, font, scale, 1)
	padding_x, padding_y = 9, 6
	box_width = text_width + padding_x * 2 + 4
	box_height = text_height + baseline + padding_y * 2
	frame_height, frame_width = frame.shape[:2]
	x = min(max(8, x), max(8, frame_width - box_width - 8))
	y = min(max(8, y), max(8, frame_height - box_height - 8))

	cv2.rectangle(frame, (x, y), (x + box_width, y + box_height), COLOR_PANEL, -1)
	cv2.rectangle(frame, (x, y), (x + 3, y + box_height), accent, -1)
	cv2.putText(
		frame,
		text,
		(x + padding_x + 3, y + padding_y + text_height),
		font,
		scale,
		COLOR_TEXT,
		1,
		cv2.LINE_AA,
	)
	return box_height


def draw_camera_header(frame, face_count, gender_counts, object_count, fatigue_alert):
	header_height = 62
	shade = frame.copy()
	cv2.rectangle(shade, (0, 0), (frame.shape[1], header_height), COLOR_PANEL, -1)
	cv2.addWeighted(shade, 0.86, frame, 0.14, 0, frame)
	cv2.putText(
		frame,
		"ANALISE FACIAL  /  CAMERA ATIVA",
		(14, 22),
		cv2.FONT_HERSHEY_SIMPLEX,
		0.52,
		COLOR_TEXT,
		1,
		cv2.LINE_AA,
	)
	counts = (
		f"ROSTOS {face_count}  |  HOMENS {gender_counts['Homem']}  "
		f"MULHERES {gender_counts['Mulher']}  |  OBJETOS {object_count}"
	)
	cv2.putText(
		frame,
		counts,
		(14, 49),
		cv2.FONT_HERSHEY_SIMPLEX,
		0.43,
		COLOR_MUTED,
		1,
		cv2.LINE_AA,
	)
	status = "Q: SAIR"
	(text_width, _), _ = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
	cv2.putText(
		frame,
		status,
		(max(14, frame.shape[1] - text_width - 14), 22),
		cv2.FONT_HERSHEY_SIMPLEX,
		0.46,
		COLOR_MUTED,
		1,
		cv2.LINE_AA,
	)
	if fatigue_alert:
		alert_text = "ALERTA DE FADIGA  |  OLHOS FECHADOS"
		cv2.rectangle(frame, (0, header_height), (frame.shape[1], header_height + 32), COLOR_ALERT, -1)
		cv2.putText(
			frame,
			alert_text,
			(14, header_height + 22),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.55,
			(255, 255, 255),
			2,
			cv2.LINE_AA,
		)


def main():
	cascade_path = Path(cv2.__file__).resolve().parent / "data" / "haarcascade_frontalface_default.xml"
	face_cascade = cv2.CascadeClassifier(str(cascade_path))
	if face_cascade.empty():
		raise RuntimeError(f"Não foi possível carregar o classificador: {cascade_path}")

	face_landmarker_path = ensure_model(FACE_LANDMARKER_MODEL, FACE_LANDMARKER_URL)
	object_model_path = ensure_model(OBJECT_MODEL, OBJECT_MODEL_URL)
	landmarker_options = mp.tasks.vision.FaceLandmarkerOptions(
		base_options=mp.tasks.BaseOptions(model_asset_path=str(face_landmarker_path)),
		running_mode=mp.tasks.vision.RunningMode.VIDEO,
		output_face_blendshapes=True,
		num_faces=5,
	)
	face_landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(landmarker_options)
	from ultralytics import YOLO

	object_detector = YOLO(str(object_model_path))
	object_class_ids = [
		class_id
		for class_id, class_name in object_detector.names.items()
		if class_name in OBJECT_TRANSLATIONS
	]

	camera = cv2.VideoCapture(0)
	if not camera.isOpened():
		face_landmarker.close()
		raise RuntimeError("Não foi possível abrir a webcam.")

	print("Iniciando detecção facial. Pressione Q para sair.")
	frame_count = 0
	attributes_by_face = []
	detected_objects = []
	fatigue_monitor = FatigueMonitor()
	previous_timestamp_ms = 0
	try:
		while True:
			ret, frame = camera.read()
			if not ret:
				print("Não foi possível capturar imagem da webcam.")
				break

			# Espelha a imagem para acompanhar os movimentos como em um espelho.
			frame = cv2.flip(frame, 1)
			frame_count += 1

			# Tons de cinza facilitam a detecção pelo classificador Haar.
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			faces = face_cascade.detectMultiScale(
				gray,
				scaleFactor=1.1,
				minNeighbors=4,
				minSize=(30, 30),
			)
			if frame_count % OBJECT_INTERVAL == 0:
				try:
					prediction = cast(
						Any,
						next(
							iter(
								object_detector.predict(
									frame,
									classes=object_class_ids,
									conf=0.35,
									imgsz=416,
									device="cpu",
									verbose=False,
								)
							),
							None,
						),
					)
					detected_objects = []
					if prediction is not None:
						for box in prediction.boxes:
							class_id = int(box.cls.item())
							class_name = object_detector.names[class_id]
							x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
							detected_objects.append(
								(
									int(x1),
									int(y1),
									int(x2),
									int(y2),
									OBJECT_TRANSLATIONS[class_name],
									float(box.conf.item()),
								)
							)
				except Exception as error:
					print(f"Falha na detecção de objetos: {error}")
					detected_objects = []

			frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			media_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
			timestamp_ms = max(int(time.monotonic() * 1000), previous_timestamp_ms + 1)
			previous_timestamp_ms = timestamp_ms
			landmark_result = face_landmarker.detect_for_video(media_image, timestamp_ms)
			fatigue_alert, alert_started = fatigue_monitor.update(
				landmark_result.face_blendshapes or [],
				time.monotonic(),
			)
			if alert_started:
				threading.Thread(target=play_fatigue_alarm, daemon=True).start()

			if frame_count % ANALYSIS_INTERVAL == 0:
				attributes_by_face = []

			for face_index, (x, y, width, height) in enumerate(faces):
				roi_color = frame[y : y + height, x : x + width]
				if frame_count % ANALYSIS_INTERVAL == 0:
					try:
						attributes_by_face.append(analyze_face_attributes(roi_color))
					except Exception as error:
						print(f"Falha na análise facial: {error}")
						attributes_by_face.append(None)

				if face_index < len(attributes_by_face):
					attributes = attributes_by_face[face_index]
					if attributes is None:
						labels = [("ANALISE INDISPONIVEL", COLOR_UNKNOWN)]
					else:
						labels = [
							(
								f"EMOCAO  {attributes['emotion']}  {attributes['emotion_confidence']:.0f}%",
								COLOR_MUTED,
							),
							(f"IDADE APARENTE  {attributes['age']}", COLOR_MUTED),
							(
								f"GENERO ESTIMADO  {attributes['gender']}",
								GENDER_COLORS.get(attributes["gender"], COLOR_UNKNOWN),
							),
							(f"CATEGORIA RACIAL  {attributes['race']}", COLOR_MUTED),
						]
				else:
					attributes = None
					labels = [("ANALISANDO ATRIBUTOS...", COLOR_UNKNOWN)]

				gender_color = (
					GENDER_COLORS.get(attributes["gender"], COLOR_UNKNOWN)
					if attributes is not None
					else COLOR_UNKNOWN
				)
				cv2.rectangle(
					frame,
					(x, y),
					(x + width, y + height),
					gender_color,
					2,
				)

				badge_y = y + height + 7
				for label, accent in labels:
					badge_height = draw_badge(frame, label, x, badge_y, accent)
					badge_y += badge_height + 4

			if len(faces) == 0:
				attributes_by_face = []
			gender_counts = count_estimated_genders(attributes_by_face)

			for x1, y1, x2, y2, object_name, confidence in detected_objects:
				cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_OBJECT, 2)
				draw_badge(
					frame,
					f"{object_name} {confidence:.0%}",
					x1,
					max(64, y1 - 28),
					COLOR_OBJECT,
				)

			draw_camera_header(
				frame,
				len(faces),
				gender_counts,
				len(detected_objects),
				fatigue_alert,
			)
			cv2.imshow("Deteccao facial", frame)
			if cv2.waitKey(1) & 0xFF == ord("q"):
				break
	finally:
		camera.release()
		cv2.destroyAllWindows()
		face_landmarker.close()


if __name__ == "__main__":
	main()