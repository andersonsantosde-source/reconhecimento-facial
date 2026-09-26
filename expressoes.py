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
				cv2.rectangle(
					frame,
					(x, y),
					(x + width, y + height),
					(0, 255, 0),
					2,
				)
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
						labels = ["Analise indisponivel"]
					else:
						labels = [
							f"Emocao: {attributes['emotion']} ({attributes['emotion_confidence']:.0f}%)",
							f"Idade aparente: {attributes['age']}",
							f"Genero estimado: {attributes['gender']}",
							f"Categoria racial estimada: {attributes['race']}",
						]
				else:
					labels = ["Analisando atributos..."]

				label_y = max(y - 10, 20)
				for label_index, label in enumerate(labels):
					cv2.putText(
						frame,
						label,
						(x, label_y + label_index * 20),
						cv2.FONT_HERSHEY_SIMPLEX,
						0.5,
						(0, 255, 0),
						2,
						cv2.LINE_AA,
					)

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