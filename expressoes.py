from pathlib import Path
from typing import Any, cast

import cv2


ANALYSIS_INTERVAL = 30
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


def draw_camera_header(frame, face_count):
	header_height = 38
	shade = frame.copy()
	cv2.rectangle(shade, (0, 0), (frame.shape[1], header_height), COLOR_PANEL, -1)
	cv2.addWeighted(shade, 0.86, frame, 0.14, 0, frame)
	cv2.putText(
		frame,
		"ANALISE FACIAL  /  CAMERA ATIVA",
		(14, 24),
		cv2.FONT_HERSHEY_SIMPLEX,
		0.52,
		COLOR_TEXT,
		1,
		cv2.LINE_AA,
	)
	status = f"ROSTOS: {face_count}  |  Q: SAIR"
	(text_width, _), _ = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
	cv2.putText(
		frame,
		status,
		(max(14, frame.shape[1] - text_width - 14), 24),
		cv2.FONT_HERSHEY_SIMPLEX,
		0.46,
		COLOR_MUTED,
		1,
		cv2.LINE_AA,
	)


def main():
	cascade_path = Path(cv2.__file__).resolve().parent / "data" / "haarcascade_frontalface_default.xml"
	face_cascade = cv2.CascadeClassifier(str(cascade_path))
	if face_cascade.empty():
		raise RuntimeError(f"Não foi possível carregar o classificador: {cascade_path}")

	camera = cv2.VideoCapture(0)
	if not camera.isOpened():
		raise RuntimeError("Não foi possível abrir a webcam.")

	print("Iniciando detecção facial. Pressione Q para sair.")
	frame_count = 0
	attributes_by_face = []
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

			draw_camera_header(frame, len(faces))
			if len(faces) == 0:
				attributes_by_face = []

			cv2.imshow("Deteccao facial", frame)
			if cv2.waitKey(1) & 0xFF == ord("q"):
				break
	finally:
		camera.release()
		cv2.destroyAllWindows()


if __name__ == "__main__":
	main()