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

Feche a janela da câmera pressionando `Q`. Para cadastrar alguém, mantenha o rosto visível na câmera, pressione `N` e informe o nome; o programa registra o rosto do quadro atual. Nomes duplicados são recusados. A aprovação é automática e não exige tecla; o áudio é tocado quando o estado muda para liberado ou negado.

O script `expressoes.py` detecta rostos com OpenCV e, a cada 30 quadros, estima emoção, idade aparente, gênero e categoria racial com DeepFace. Também conta os gêneros estimados, detecta garrafas, mouses e celulares com YOLO11n e aciona um alarme quando o FaceLandmarker detecta olhos fechados sem sorriso por pelo menos 1,5 segundo. Pressione `Q` para encerrar.

Execute esses desafios com `python expressoes.py`. O primeiro uso baixa os modelos do MediaPipe e YOLO para `%USERPROFILE%\.mediapipe\reconhecimento-facial` e pode baixar modelos adicionais do DeepFace; o modelo de classificação racial tem aproximadamente 537 MB. Idade, gênero, categoria racial e expressão são estimativas sujeitas a erro, não características verificadas. A contagem por gênero e o alerta são apenas indicadores demonstrativos, não devem decidir acesso, emprego ou tratamento de pessoas.

Os nomes e vetores biométricos são armazenados localmente em `%USERPROFILE%\.mediapipe\reconhecimento-facial\users.sqlite3`. A imagem capturada para cadastro não é guardada. Proteja a conta do Windows e permita cadastro somente em um posto confiável. Na primeira execução, o perfil existente da imagem `IMG_0670.jpeg` é adicionado como `Perfil Principal`; os modelos leves do MediaPipe e SFace são baixados para a pasta do usuário.