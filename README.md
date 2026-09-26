# Reconhecimento facial

Aplicação local de reconhecimento facial usando OpenCV e DeepFace. A câmera e a imagem de referência são processadas no computador.

## Requisitos

- Python 3.10 ou superior
- Webcam
- Uma imagem nítida de um rosto, salva como `face_id.jpg` na pasta do projeto

Por privacidade, `face_id.jpg` não é enviado ao GitHub. Adicione sua própria imagem localmente antes de executar.

## Instalação no Windows

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Executar

```powershell
python reconhecimento_facial.py
```

Feche a janela da câmera pressionando `Q`. Na primeira execução, o DeepFace pode baixar os modelos necessários.