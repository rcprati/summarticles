# summarticles

Summarticles is a tool for processing academic articles (pdf files) to extract information and produce visualizations and statistics. 

To run summarticles, you need:

- A Python 3.8 distribution
- [Grobid](https://grobid.readthedocs.io/en/latest/)

We recommend to use Python anaconda, with a new enviroment. To install summarticles in a new enviroment with anaconda, run the following commands into the terminal:

```
conda create -n summarticles python=3.8
conda activate summarticles
git clone https://github.com/rcprati/summarticles.git
cd summarticles
pip install -r requirements.txt 
```

To run summarticles, run 

```
docker run -t --rm --init -p 8080:8070 -p 8081:8071 --memory="9g" lfoppiano/grobid:0.7.1 &
cd notebooks
streamlit run summarticles.py
```

Summarticles will run into a new tab into your web browser

Note: you may need to have root previleges to run docker
