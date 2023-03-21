
# Semantic Search

Modification of CX_DB8 project by refactoring the code, adding user interface, and adding minor functional features.


## Features

- Asymmetric or Symmetric Supported Semantic Search by Average Attention with Sliding Window Algorithm.
- Plug-and-Play Retriever Model & Reranker Model.
- Text, Web, and Pdf Input to Text and Pdf Output format.
- Output Highlighter.
- Processing time & Score statistics.
- Caching to speedup reprocessing (Click "Git remote repository sync" button/rerun notebook cell to clear unused data in RAM after repeated unique processing).
- Raw results for inspecting.

## Acknowledgements

 - [CX_DB8](https://github.com/Hellisotherpeople/CX_DB8)

## Authors

- [@Hellisotherpeople](https://github.com/Hellisotherpeople) (Base idea and implementation)
- [@muazhari](https://github.com/muazhari) (Modification)

## Demo

[![demo](http://img.youtube.com/vi/bu93G6YesaQ/0.jpg)](http://www.youtube.com/watch?v=bu93G6YesaQ)

## Walkthrough 

1. Get your ngrok Authentication Token.
2. Create cell based on below Jupyter Notebook script in Google Colab, Kaggle, or other alternatives.

```python
#@title Semantic Search App
NGROK_TOKEN = "" #@param {type:"string"} 

%cd ~
!git clone https://github.com/muazhari/semantic-search.git
%cd ~/semantic-search/
!git fetch --all
!git reset --hard origin

!apt-get update -y
!yes | DEBIAN_FRONTEND=noninteractive apt-get install -yqq wkhtmltopdf xvfb libopenblas-dev libomp-dev poppler-utils openjdk-8-jdk jq

!pip install -r requirements.txt
!pip install pyngrok

!nvidia-smi

get_ipython().system_raw(f'ngrok authtoken {NGROK_TOKEN}')
get_ipython().system_raw('ngrok http 8501 &')
print("Open public URL:")
!curl -s http://localhost:4040/api/tunnels | jq ".tunnels[0].public_url"
!streamlit run ~/semantic-search/app.py

!sleep 10000000
```

3. Submit your ngrok Authentication Token to `NGROK_TOKEN` coloumn in the cell form.
4. Enable GPU in the Notebook.
5. Run the cell.
6. Wait until the setups are done.
7. Open ngrok public URL.
8. Use the app.

## Warning
- This repository not yet peer reviewed, so be careful when using it.


    
