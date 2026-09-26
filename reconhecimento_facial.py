import subprocess
import sys
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
import playsound as playsound_module


PROJECT_DIR = Path(__file__).resolve().parent
FACE_ID = PROJECT_DIR / "IMG_0670.jpeg"
AUDIO_ACCESS_GRANTED = PROJECT_DIR / "acesso_liberado.wav"
AUDIO_ACCESS_DENIED = PROJECT_DIR / "acesso_negado.wav"
MODEL_DIR = Path.home() / ".mediapipe" / "reconhecimento-facial"
DETECTOR_MODEL = MODEL_DIR / "blaze_face_short_range.tflite"
RECOGNITION_MODEL = MODEL_DIR / "face_recognition_sface_2021dec.onnx"
DETECTOR_MODEL_URL = (
	"https://storage.googleapis.com/mediapipe-models/face_detector/"
	"blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)
RECOGNITION_MODEL_URL = (
	"https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/"
	"face_recognition_sface_2021dec.onnx"
)
COSINE_THRESHOLD = 0.363
_audio_process = None


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


def extract_face(detector, image):
	image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	media_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
	detections = detector.detect(media_image).detections or []
	if not detections:
		return None

	detection = max(
		detections,
		key=lambda item: item.bounding_box.width * item.bounding_box.height,
	)
	box = detection.bounding_box
	padding = int(max(box.width, box.height) * 0.08)
	x_start = max(0, box.origin_x - padding)
	y_start = max(0, box.origin_y - padding)
	x_end = min(image.shape[1], box.origin_x + box.width + padding)
	y_end = min(image.shape[0], box.origin_y + box.height + padding)

	face = image[y_start:y_end, x_start:x_end]
	return face if face.size else None


def extract_feature(recognizer, face):
	face_input = cv2.resize(face, (112, 112), interpolation=cv2.INTER_AREA)
	return recognizer.feature(face_input)


def play_access_audio(status):
	global _audio_process

	audio_path = {
		"ACESSO LIBERADO": AUDIO_ACCESS_GRANTED,
		"ACESSO NEGADO": AUDIO_ACCESS_DENIED,
	}.get(status)
	if audio_path is None:
		return

	try:
		if _audio_process is not None and _audio_process.poll() is None:
			_audio_process.terminate()
			_audio_process.wait(timeout=1)
		_audio_process = subprocess.Popen(
			[sys.executable, playsound_module.__file__, audio_path.name],
			cwd=PROJECT_DIR,
		)
	except Exception as error:
		print(f"Não foi possível reproduzir o áudio {audio_path.name}: {error}")


def main():
	if not FACE_ID.is_file():
		raise FileNotFoundError(f"Imagem de referência não encontrada: {FACE_ID}")

	reference_image = cv2.imread(str(FACE_ID))
	if reference_image is None:
		raise ValueError(f"Não foi possível abrir a imagem: {FACE_ID}")

	detector_path = ensure_model(DETECTOR_MODEL, DETECTOR_MODEL_URL)
	recognition_path = ensure_model(RECOGNITION_MODEL, RECOGNITION_MODEL_URL)
	detector_options = mp.tasks.vision.FaceDetectorOptions(
		base_options=mp.tasks.BaseOptions(model_asset_path=str(detector_path)),
		running_mode=mp.tasks.vision.RunningMode.IMAGE,
		min_detection_confidence=0.6,
	)
	face_detector = mp.tasks.vision.FaceDetector.create_from_options(detector_options)
	try:
		reference_face = extract_face(face_detector, reference_image)
		if reference_face is None:
			raise ValueError(
				"O MediaPipe não encontrou um rosto na imagem de referência. "
				"Use uma foto nítida, frontal e bem iluminada."
			)

		recognizer = cv2.FaceRecognizerSF.create(str(recognition_path), "")
		reference_feature = extract_feature(recognizer, reference_face)
		print("Imagem de referência carregada e rosto detectado pelo MediaPipe.")

		camera = cv2.VideoCapture(0)
		if not camera.isOpened():
			raise RuntimeError("Não foi possível abrir a câmera. Verifique a conexão e as permissões.")

		frame_count = 0
		status_acesso = "AGUARDANDO LEITURA"
		cor_status = (0, 200, 255)
		try:
			while True:
				success, frame = camera.read()
				if not success:
					print("Não foi possível capturar imagem da câmera.")
					break

				frame_count += 1
				if frame_count % 30 == 0:
					status_anterior = status_acesso
					try:
						face = extract_face(face_detector, frame)
						if face is None:
							status_acesso = "NENHUM ROSTO DETECTADO"
							cor_status = (0, 165, 255)
						else:
							feature = extract_feature(recognizer, face)
							score = recognizer.match(
								reference_feature,
								feature,
								cv2.FaceRecognizerSF_FR_COSINE,
							)
							if score >= COSINE_THRESHOLD:
								status_acesso = "ACESSO LIBERADO"
								cor_status = (0, 255, 0)
							else:
								status_acesso = "ACESSO NEGADO"
								cor_status = (0, 0, 255)
					except Exception:
						status_acesso = "ERRO DE LEITURA"
						cor_status = (0, 165, 255)

					if status_acesso != status_anterior:
						play_access_audio(status_acesso)

				cv2.putText(
					frame,
					status_acesso,
					(30, 50),
					cv2.FONT_HERSHEY_SIMPLEX,
					1.0,
					cor_status,
					2,
				)
				cv2.putText(
					frame,
					"Pressione Q para sair",
					(20, 75),
					cv2.FONT_HERSHEY_SIMPLEX,
					0.6,
					(255, 255, 255),
					2,
				)
				cv2.imshow("Reconhecimento facial", frame)

				if cv2.waitKey(1) & 0xFF == ord("q"):
					break
		finally:
			camera.release()
			cv2.destroyAllWindows()
	finally:
		face_detector.close()


if __name__ == "__main__":
	main()
