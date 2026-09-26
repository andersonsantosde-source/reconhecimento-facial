from pathlib import Path

import cv2
from deepface import DeepFace


FACE_ID = Path(__file__).resolve().parent / "IMG_0670.jpeg"


def main():
	if not FACE_ID.is_file():
		raise FileNotFoundError(f"Imagem de referência não encontrada: {FACE_ID}")

	reference_image = cv2.imread(str(FACE_ID))
	if reference_image is None:
		raise ValueError(f"Não foi possível abrir a imagem: {FACE_ID}")

	try:
		DeepFace.verify(
			img1_path=reference_image,
			img2_path=reference_image,
			enforce_detection=True,
			detector_backend="retinaface",
		)
		print("Imagem de referência carregada e rosto detectado.")
	except Exception as error:
		raise ValueError(
			"Não foi possível validar a imagem de referência. "
			"Confira se ela contém um rosto nítido."
		) from error

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
				try:
					result = DeepFace.verify(
						img1_path=reference_image,
						img2_path=frame,
						enforce_detection=True,
						detector_backend="retinaface",
					)
					if result["verified"]:
						status_acesso = "ACESSO LIBERADO"
						cor_status = (0, 255, 0)
					else:
						status_acesso = "ACESSO NEGADO"
						cor_status = (0, 0, 255)
				except Exception:
					status_acesso = "ERRO DE LEITURA"
					cor_status = (0, 165, 255)

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


if __name__ == "__main__":
	main()
