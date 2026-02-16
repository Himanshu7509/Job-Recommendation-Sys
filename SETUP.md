Python version: 3.10.x
Steps:
1. python -m venv venv
2. activate venv
3. pip install -r requirements.txt
4. pip install en_core_web_sm-3.7.1.whl
5. python tools/build_faiss_index.py
6. uvicorn app.main:app --reload
