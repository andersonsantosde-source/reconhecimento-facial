import subprocess
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
import numpy as np
import playsound as playsound_module


PROJECT_DIR = Path(__file__).resolve().parent
FACE_ID = PROJECT_DIR / "IMG_0670.jpeg"
AUDIO_ACCESS_GRANTED = PROJECT_DIR / "acesso_liberado.wav"
AUDIO_ACCESS_DENIED = PROJECT_DIR / "acesso_negado.wav"
MODEL_DIR = Path.home() / ".mediapipe" / "reconhecimento-facial"
DETECTOR_MODEL = MODEL_DIR / "blaze_face_short_range.tflite"
RECOGNITION_MODEL = MODEL_DIR / "face_recognition_sface_2021dec.onnx"
USER_DATABASE = MODEL_DIR / "users.sqlite3"
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
UI_WIDTH = 1280
UI_HEIGHT = 800

COLOR_BACKGROUND = (18, 22, 24)
COLOR_SURFACE = (28, 34, 36)
COLOR_SURFACE_LIGHT = (37, 44, 46)
COLOR_BORDER = (63, 72, 73)
COLOR_TEXT = (232, 237, 233)
COLOR_MUTED = (145, 158, 155)
COLOR_ACCENT = (74, 190, 174)
COLOR_WARNING = (38, 176, 244)
COLOR_DANGER = (74, 76, 235)


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


def initialize_user_database():
	USER_DATABASE.parent.mkdir(parents=True, exist_ok=True)
	connection = sqlite3.connect(USER_DATABASE)
	connection.execute(
		"""
		CREATE TABLE IF NOT EXISTS users (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			name TEXT NOT NULL COLLATE NOCASE UNIQUE,
			embedding BLOB NOT NULL,
			created_at TEXT NOT NULL
		)
		"""
	)
	connection.commit()
	return connection


def load_users(connection):
	rows = connection.execute(
		"SELECT name, embedding FROM users ORDER BY name COLLATE NOCASE"
	).fetchall()
	return [
		{"name": name, "feature": np.frombuffer(embedding, dtype=np.float32).copy()}
		for name, embedding in rows
	]


def save_user(connection, name, feature):
	clean_name = " ".join(name.split())
	if not clean_name:
		raise ValueError("O nome do usuário não pode ficar vazio.")

	embedding = np.asarray(feature, dtype=np.float32).reshape(-1).tobytes()
	connection.execute(
		"INSERT INTO users (name, embedding, created_at) VALUES (?, ?, ?)",
		(clean_name, embedding, datetime.now().isoformat(timespec="seconds")),
	)
	connection.commit()
	return clean_name


def prompt_user_name():
	import tkinter as tk
	from tkinter import simpledialog

	root = tk.Tk()
	root.withdraw()
	root.attributes("-topmost", True)
	try:
		return simpledialog.askstring(
			"Cadastro de usuário",
			"Nome do novo usuário:",
			parent=root,
		)
	finally:
		root.destroy()


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


def draw_label(image, text, position, scale=0.55, color=COLOR_TEXT, thickness=1):
	cv2.putText(
		image,
		text,
		position,
		cv2.FONT_HERSHEY_SIMPLEX,
		scale,
		color,
		thickness,
		cv2.LINE_AA,
	)


def render_dashboard(
	frame,
	status,
	score,
	frame_count,
	timestamp=None,
	user_count=0,
	matched_user="--",
):
	if timestamp is None:
		timestamp = datetime.now()

	canvas = np.full((UI_HEIGHT, UI_WIDTH, 3), COLOR_BACKGROUND, dtype=np.uint8)
	cv2.rectangle(canvas, (0, 0), (UI_WIDTH, 76), COLOR_SURFACE, -1)
	cv2.line(canvas, (0, 75), (UI_WIDTH, 75), COLOR_BORDER, 1, cv2.LINE_AA)
	cv2.rectangle(canvas, (28, 22), (70, 54), COLOR_ACCENT, 2, cv2.LINE_AA)
	cv2.line(canvas, (36, 38), (45, 47), COLOR_ACCENT, 2, cv2.LINE_AA)
	cv2.line(canvas, (45, 47), (62, 30), COLOR_ACCENT, 2, cv2.LINE_AA)
	draw_label(canvas, "CENTRAL DE SEGURANCA", (84, 36), 0.72, COLOR_TEXT, 2)
	draw_label(canvas, "CONTROLE DE ACESSO  /  POSTO 01", (84, 58), 0.42, COLOR_MUTED)
	draw_label(canvas, "SISTEMA BIOMETRICO  |  OPERACIONAL", (834, 34), 0.42, COLOR_ACCENT, 1)
	draw_label(canvas, timestamp.strftime("%d/%m/%Y   %H:%M:%S"), (1018, 57), 0.48, COLOR_TEXT)

	video_x, video_y = 28, 104
	video_width, video_height = 800, 600
	video = cv2.resize(frame, (video_width, video_height), interpolation=cv2.INTER_AREA)
	canvas[video_y : video_y + video_height, video_x : video_x + video_width] = video
	cv2.rectangle(
		canvas,
		(video_x - 1, video_y - 1),
		(video_x + video_width, video_y + video_height),
		COLOR_BORDER,
		1,
		cv2.LINE_AA,
	)

	corner_size = 24
	for x, y, x_direction, y_direction in (
		(video_x, video_y, 1, 1),
		(video_x + video_width, video_y, -1, 1),
		(video_x, video_y + video_height, 1, -1),
		(video_x + video_width, video_y + video_height, -1, -1),
	):
		cv2.line(canvas, (x, y), (x + x_direction * corner_size, y), COLOR_ACCENT, 2, cv2.LINE_AA)
		cv2.line(canvas, (x, y), (x, y + y_direction * corner_size), COLOR_ACCENT, 2, cv2.LINE_AA)

	draw_label(canvas, "CAMERA 01  /  AO VIVO", (video_x, 735), 0.46, COLOR_MUTED)
	draw_label(canvas, f"QUADRO {frame_count:06d}", (680, 735), 0.46, COLOR_MUTED)

	panel_x, panel_width = 856, 396
	cv2.rectangle(canvas, (panel_x, 104), (panel_x + panel_width, 264), COLOR_SURFACE, -1)
	cv2.rectangle(canvas, (panel_x, 104), (panel_x + panel_width, 264), COLOR_BORDER, 1)
	draw_label(canvas, "DECISAO DE ACESSO", (panel_x + 20, 133), 0.48, COLOR_MUTED)

	status_colors = {
		"ACESSO LIBERADO": COLOR_ACCENT,
		"ACESSO NEGADO": COLOR_DANGER,
		"AGUARDANDO LEITURA": COLOR_WARNING,
		"NENHUM ROSTO DETECTADO": COLOR_WARNING,
		"ERRO DE LEITURA": COLOR_WARNING,
		"USUARIO CADASTRADO": COLOR_ACCENT,
		"NOME JA CADASTRADO": COLOR_WARNING,
		"ERRO NO CADASTRO": COLOR_WARNING,
	}
	status_color = status_colors.get(status, COLOR_WARNING)
	cv2.rectangle(canvas, (panel_x + 20, 151), (panel_x + 29, 226), status_color, -1)
	draw_label(canvas, status, (panel_x + 45, 190), 0.68, status_color, 2)
	draw_label(canvas, "ANALISE AUTOMATICA", (panel_x + 45, 220), 0.42, COLOR_MUTED)

	cv2.rectangle(canvas, (panel_x, 282), (panel_x + panel_width, 430), COLOR_SURFACE, -1)
	cv2.rectangle(canvas, (panel_x, 282), (panel_x + panel_width, 430), COLOR_BORDER, 1)
	draw_label(canvas, "VERIFICACAO BIOMETRICA", (panel_x + 20, 311), 0.48, COLOR_MUTED)
	draw_label(canvas, "USUARIO IDENTIFICADO", (panel_x + 20, 350), 0.43, COLOR_MUTED)
	draw_label(canvas, matched_user[:28], (panel_x + 20, 377), 0.48, COLOR_TEXT, 1)
	draw_label(canvas, "SIMILARIDADE", (panel_x + 20, 410), 0.40, COLOR_MUTED)
	score_text = "--" if score is None else f"{score * 100:.1f}%"
	draw_label(canvas, score_text, (panel_x + 270, 410), 0.52, COLOR_ACCENT, 1)

	cv2.rectangle(canvas, (panel_x, 448), (panel_x + panel_width, 628), COLOR_SURFACE, -1)
	cv2.rectangle(canvas, (panel_x, 448), (panel_x + panel_width, 628), COLOR_BORDER, 1)
	draw_label(canvas, "ESTADO DO SISTEMA", (panel_x + 20, 477), 0.48, COLOR_MUTED)
	cv2.circle(canvas, (panel_x + 27, 510), 5, COLOR_ACCENT, -1, cv2.LINE_AA)
	draw_label(canvas, "DETECTOR MEDIAPIPE", (panel_x + 43, 515), 0.43, COLOR_TEXT)
	cv2.circle(canvas, (panel_x + 27, 544), 5, COLOR_ACCENT, -1, cv2.LINE_AA)
	draw_label(canvas, "RECONHECIMENTO SFACE", (panel_x + 43, 549), 0.43, COLOR_TEXT)
	cv2.circle(canvas, (panel_x + 27, 578), 5, COLOR_ACCENT, -1, cv2.LINE_AA)
	draw_label(canvas, "CAMERA CONECTADA", (panel_x + 43, 583), 0.43, COLOR_TEXT)
	draw_label(canvas, "LIMIAR DE DECISAO", (panel_x + 20, 612), 0.40, COLOR_MUTED)
	draw_label(canvas, f"{COSINE_THRESHOLD:.3f}", (panel_x + 296, 612), 0.43, COLOR_TEXT)
	draw_label(canvas, f"USUARIOS CADASTRADOS: {user_count}", (panel_x + 20, 638), 0.40, COLOR_MUTED)

	cv2.rectangle(canvas, (panel_x, 656), (panel_x + panel_width, 712), COLOR_SURFACE_LIGHT, -1)
	draw_label(canvas, "N  CADASTRAR USUARIO", (panel_x + 18, 680), 0.42, COLOR_ACCENT)
	draw_label(canvas, "Q  ENCERRAR", (panel_x + 257, 680), 0.40, COLOR_TEXT)
	draw_label(canvas, "Leitura e decisao sem acao manual", (panel_x + 18, 729), 0.40, COLOR_MUTED)

	cv2.line(canvas, (28, 766), (1252, 766), COLOR_BORDER, 1, cv2.LINE_AA)
	draw_label(canvas, "PROCESSAMENTO LOCAL  |  SEM ENVIO DE IMAGENS", (28, 789), 0.39, COLOR_MUTED)
	draw_label(canvas, "SESSAO ATIVA", (1110, 789), 0.39, COLOR_ACCENT)
	return canvas


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
		user_database = initialize_user_database()
		try:
			users = load_users(user_database)
			if not users:
				save_user(user_database, "Perfil Principal", reference_feature)
				users = load_users(user_database)
		finally:
			user_database.close()
		print(f"Imagem de referência carregada. {len(users)} usuário(s) cadastrado(s).")

		camera = cv2.VideoCapture(0)
		if not camera.isOpened():
			raise RuntimeError("Não foi possível abrir a câmera. Verifique a conexão e as permissões.")

		frame_count = 0
		status_acesso = "AGUARDANDO LEITURA"
		cor_status = (0, 200, 255)
		score = None
		usuario_identificado = "--"
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
							score = None
							usuario_identificado = "--"
						else:
							feature = extract_feature(recognizer, face)
							matches = [
								(
									recognizer.match(
										user["feature"].reshape(1, -1),
										feature,
										cv2.FaceRecognizerSF_FR_COSINE,
									),
									user["name"],
								)
								for user in users
							]
							score, best_match = max(matches, default=(None, "--"))
							if score >= COSINE_THRESHOLD:
								status_acesso = "ACESSO LIBERADO"
								cor_status = (0, 255, 0)
								usuario_identificado = best_match
							else:
								status_acesso = "ACESSO NEGADO"
								cor_status = (0, 0, 255)
								usuario_identificado = "NAO IDENTIFICADO"
					except Exception:
						status_acesso = "ERRO DE LEITURA"
						cor_status = (0, 165, 255)
						score = None
						usuario_identificado = "--"

					if status_acesso != status_anterior:
						play_access_audio(status_acesso)

				interface = render_dashboard(
					frame,
					status_acesso,
					score,
					frame_count,
					user_count=len(users),
					matched_user=usuario_identificado,
				)
				cv2.imshow("Central de Seguranca | Controle de Acesso", interface)

				key = cv2.waitKey(1) & 0xFF
				if key == ord("n"):
					try:
						new_name = prompt_user_name()
						if new_name:
							face = extract_face(face_detector, frame)
							if face is None:
								status_acesso = "NENHUM ROSTO DETECTADO"
								cor_status = (0, 165, 255)
							else:
								feature = extract_feature(recognizer, face)
								user_database = initialize_user_database()
								try:
									new_name = save_user(user_database, new_name, feature)
									users = load_users(user_database)
								finally:
									user_database.close()
								usuario_identificado = new_name
								status_acesso = "USUARIO CADASTRADO"
								cor_status = COLOR_ACCENT
								print(f"Usuário cadastrado: {new_name}")
					except sqlite3.IntegrityError:
						status_acesso = "NOME JA CADASTRADO"
						cor_status = COLOR_WARNING
					except Exception as error:
						status_acesso = "ERRO NO CADASTRO"
						cor_status = COLOR_WARNING
						print(f"Não foi possível cadastrar o usuário: {error}")
				elif key == ord("q"):
					break
		finally:
			camera.release()
			cv2.destroyAllWindows()
	finally:
		face_detector.close()


if __name__ == "__main__":
	main()
