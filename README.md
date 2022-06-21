
# Context Search

Modification of CX_DB8 project by refactoring the code, adding user interface, and adding minor functional features.


## Features

- Asymetric/Symetric Semantic Search.
- Plug-and-Play Retriever Model & Reranker Model (Optional).
- Text, Web, and Pdf Input to Text and Pdf Output format.
- Output Highlighter.
- Processing time statistic & Score statistics.
- Raw results for inspecting.

## Acknowledgements

 - [CX_DB8](https://github.com/Hellisotherpeople/CX_DB8)

## Authors

- [@Hellisotherpeople](https://github.com/Hellisotherpeople) (Base idea and implementation)
- [@muazhari](https://github.com/muazhari) (Modification)

## Demo

[![demo](http://img.youtube.com/vi/bu93G6YesaQ/0.jpg)](http://www.youtube.com/watch?v=bu93G6YesaQ)

## Installation

1. Get your ngrok Authentication Token.
2. Deploy/create cell based on below Jupyter Notebook script in Google Colab or other alternatives.

```python
#@title Context Search App
NGROK_TOKEN = "" #@param {type:"string"} 
sh = """
cd /content
git clone https://github.com/muazhari/context-search.git
cd /content/context-search/
git fetch --all
git reset --hard origin

apt-get update
apt-get install wkhtmltopdf xvfb libopenblas-dev libomp-dev poppler-utils

cd /content/context-search/
pip install -r requirements.txt
pip install txtai[pipeline,similarity]
"""
with open('script.sh', 'w') as file:
  file.write(sh)

!bash script.sh

!nvidia-smi

%cd /content
!wget -nc https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
!unzip -n ngrok-stable-linux-amd64.zip 
get_ipython().system_raw('./ngrok authtoken {NGROK_TOKEN}'.format(NGROK_TOKEN=NGROK_TOKEN))
get_ipython().system_raw('./ngrok http 8501 &')
!apt-get install jq
print("Open public URL:")
!curl -s http://localhost:4040/api/tunnels | jq ".tunnels[0].public_url"
!streamlit run /content/context-search/app.py

!sleep 10000000
```

3. Submit your ngrok Authentication Token to `NGROK_TOKEN` coloumn in the cell form.
4. Enable GPU in the Notebook.
5. Run the cell.


    
