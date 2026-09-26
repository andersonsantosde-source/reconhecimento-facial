# Reconhecimento facial

Aplicação local de reconhecimento facial usando MediaPipe para detectar rostos e SFace/OpenCV para compará-los. Os resultados de acesso liberado e negado também são anunciados em áudio.

## Requisitos

- Python 3.10 ou superior
- Webcam
- Uma imagem nítida de um rosto, salva como `IMG_0670.jpeg` na pasta do projeto

Por privacidade, `IMG_0670.jpeg` não é enviado ao GitHub. Adicione sua própria imagem localmente com esse nome antes de executar. Os áudios `acesso_liberado.wav` e `acesso_negado.wav` ficam no projeto.

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

Feche a janela da câmera pressionando `Q`. A aprovação é automática e não exige tecla; o áudio é tocado quando o estado muda para liberado ou negado. Na primeira execução, o programa baixa os modelos leves do MediaPipe e do SFace para a pasta do usuário.